import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    B, H,
    stride_A, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
    stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load scalars
    A = tl.load(A_log_ptr + h_idx * stride_A).to(tl.float32)
    a = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h).to(tl.float32)
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt).to(tl.float32)
    bb = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32)

    x = a + dt
    sp = tl.log(1.0 + tl.exp(x))  # softplus
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))  # sigmoid

    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_state_b, stride_state_v, stride_state_k,
    stride_tmp_b, stride_tmp_h,
    num_warps: tl.constexpr,
):
    # Each program computes tmp_old_v for one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load k[b, h] as [K]
    k_off = b_idx * stride_k_b + h_idx * stride_k_h
    k_vec = tl.load(k_ptr + k_off + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)  # [K]
    # Load state[b, h] as [V, K]
    state_off = b_idx * stride_state_b + h_idx * stride_state_v  # since v dimension is second in [B,H,V,K], but here we index (b,h) then load V,K
    # We need to load as [V, K] -> state_off_base = b*stride_state_b + h*stride_state_h (but our tensors are [B,H,V,K], so stride_state_v = after H, stride_state_k = last)
    # We will use state_ptr layout as contiguous [B,H,V,K]: given strides, we can load a 2D tile with two loops.
    # Build 2D indices for [V, K]
    # For each j in 0..V-1 and k in 0..K-1, load element and accumulate
    acc = 0.0
    # Triton does not support Python for with runtime bounds cleanly; we use while loops for generality
    j = 0
    while j < V:
        kk = 0
        while kk < K:
            # address = b*stride_k_b + h*stride_k_h + j*stride_state_k + kk*stride_state_k
            val = tl.load(state_ptr + b_idx * stride_state_b + h_idx * stride_state_h + j * stride_state_v + kk * stride_state_k)
            acc += val * k_vec[kk]
            kk += 1
        j += 1
    tl.store(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h, acc)


@triton.jit
def kernel_update_and_output(
    k_ptr, beta_ptr, v_ptr, state_in_ptr, q_ptr, new_state_ptr, output_ptr,
    V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_beta_b, stride_beta_h,
    stride_v_b, stride_v_h, stride_v_v,
    stride_state_b, stride_state_v, stride_state_k,
    stride_q_b, stride_q_h, stride_q_k,
    stride_new_b, stride_new_v, stride_new_k,
    stride_out_b, stride_out_v,
    scale,
    num_warps: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)
    # Load vectors
    k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0).to(tl.float32)  # [K]
    q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0).to(tl.float32)  # [K]

    # Compute tmp_old_v = dot(k, state[b,h]) via elementwise (recompute from state_in using k[b,h])
    tmp_old_v = 0.0
    j = 0
    while j < V:
        kk = 0
        acc = 0.0
        while kk < K:
            val = tl.load(state_in_ptr + b_idx * stride_state_b + h_idx * stride_state_h + j * stride_state_v + kk * stride_state_k)
            acc += val * k_vec[kk]
            kk += 1
        tmp_old_v += acc
        j += 1

    # Load v[b,h] as [V]
    j_v = 0
    v_vec = tl.zeros([V], dtype=tl.float32)
    while j_v < V:
        v_vec[j_v] = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + j_v * stride_v_v)
        j_v += 1

    # Compute new_state[b,h] elementwise: for each j in 0..V-1 and k in 0..K-1
    j_new = 0
    while j_new < V:
        kk_new = 0
        old_state_j = 0.0
        while kk_new < K:
            addr_old = b_idx * stride_state_b + h_idx * stride_state_h + j_new * stride_state_v + kk_new * stride_state_k
            old_state_j += tl.load(state_in_ptr + addr_old)
            kk_new += 1

        # scalar contributions
        # k · (beta * v + (1-beta) * tmp_old_v)
        term = beta_val * v_vec[j_new] + (1.0 - beta_val) * tmp_old_v
        # new state vector for this j
        new_vec_j = old_state_j - (k_vec.sum() * old_state_j) + (k_vec.dot(k_vec) * term)

        # Store new_state
        kk_store = 0
        while kk_store < K:
            tl.store(new_state_ptr + b_idx * stride_new_b + j_new * stride_new_v + kk_store * stride_new_k, new_vec_j[kk_store])
            kk_store += 1
        j_new += 1

    # Compute output scalar: q · new_state
    # First fill new_state rows with computed new_vec_j
    # Then compute q · new_state by summing q[k] * new_state[j,k]
    out_sum = 0.0
    j_out = 0
    while j_out < V:
        kk_out = 0
        while kk_out < K:
            new_elem = tl.load(new_state_ptr + b_idx * stride_new_b + j_out * stride_new_v + kk_out * stride_new_k)
            out_sum += new_elem * q_vec[kk_out]
            kk_out += 1
        j_out += 1
    out_vec = out_sum * scale
    # Store output per head vector (we have single V dimension here), so store as [B,H,V]
    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_v, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure everything is on same device and float32 contiguous
        device = q.device
        B = 1  # single batch per call as per get_inputs; axes vary only batch_size
        QH = q.shape[2]
        KH = k.shape[2]
        VH = v.shape[2]
        V = v.shape[3]
        K = q.shape[3]
        H = QH  # per original logic
        assert QH == 4, "num_q_heads must be 4"
        assert KH == 4, "num_k_heads must be 4"
        assert VH == 8, "num_v_heads must be 8"
        assert K == 128 and V == 128, "K and V must be 128"

        a32 = a.to(torch.float32).contiguous()
        dt_bias32 = dt_bias.to(torch.float32).contiguous()
        b32 = b.to(torch.float32).contiguous()
        A_log32 = A_log.to(torch.float32).contiguous()
        q32 = q.to(torch.float32).contiguous()
        k32 = k.to(torch.float32).contiguous()
        v32 = v.to(torch.float32).contiguous()
        state32 = state.to(torch.float32).contiguous()

        # Launch kernels
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)
        tmp_old_v = torch.empty((B, H), dtype=torch.float32, device=device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)
        output_vec = torch.empty((B, H, V), dtype=torch.float32, device=device)

        # Kernel 1: g and beta
        grid = (B, H)
        kernel_g_beta[grid](
            A_log32, a32, dt_bias32, b32,
            g, beta,
            B, H,
            A_log32.stride(0), a32.stride(0), a32.stride(1), dt_bias32.stride(0), b32.stride(0), b32.stride(1),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
            num_warps=1
        )

        # Kernel 2: tmp_old_v = k · state
        k2 = k32.view(B, H, K).contiguous()  # [B, H, K]
        state2 = state32.view(B, H, V, K).contiguous()  # [B, H, V, K]
        grid_tmp = (B, H)
        kernel_tmp_old_v[grid_tmp](
            k2, state2, tmp_old_v,
            V, K,
            k2.stride(0), k2.stride(1), k2.stride(2),
            state2.stride(0), state2.stride(1), state2.stride(2), state2.stride(3),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
            num_warps=1
        )

        # Kernel 3: update new_state and output
        # Reshape v to [B, H, V]
        v2 = v32.view(B, H, V).contiguous()
        q2 = q32.view(B, H, K).contiguous()
        k2 = k32.view(B, H, K).contiguous()
        state_in = state32.view(B, H, V, K).contiguous()
        grid3 = (B, H)
        # We need to pass strides for q and new_state properly; output is [B,H,V]
        kernel_update_and_output[grid3](
            k2, beta, v2, state_in, q2, new_state, output_vec,
            V, K,
            k2.stride(0), k2.stride(1), k2.stride(2),
            beta.stride(0), beta.stride(1),
            v2.stride(0), v2.stride(1), v2.stride(2),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            q2.stride(0), q2.stride(1), q2.stride(2),
            new_state.stride(0), new_state.stride(1), new_state.stride(2),
            output_vec.stride(0), output_vec.stride(1),
            scale,
            num_warps=1
        )

        # Return: output in bfloat16 [B, H, V], new_state in float32 [B, H, V, K]
        return (output_vec.unsqueeze(1).to(torch.bfloat16), new_state)


def run(*args):
    return ModelNew()(*args)
