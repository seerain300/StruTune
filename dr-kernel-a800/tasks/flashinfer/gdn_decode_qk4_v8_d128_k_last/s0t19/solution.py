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
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a + dt))
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))
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
    # k[b, h] is [K], state[b, h] is [V, K]
    acc = 0.0
    # Loop over K in blocks
    for kk in range(0, K, 128):
        k_off = b_idx * stride_k_b + h_idx * stride_k_h + kk + tl.arange(0, 128)
        mask = kk + tl.arange(0, 128) < K
        k_vec = tl.load(k_ptr + k_off, mask=mask, other=0.0)
        # For each i in [0..K-1], load state[h, i] vector of length V
        for i in range(0, K):
            col = b_idx * stride_state_b + h_idx * stride_state_v + i * stride_state_k  # this pattern is incorrect; fix below

# ... (middle omitted) ...


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_state_b, stride_state_h, stride_state_v, stride_state_k,
    stride_tmp_b, stride_tmp_h,
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    acc = 0.0
    # k[b, h] as [K]
    for kk in range(0, K, 128):
        offs = kk + tl.arange(0, 128)
        mask_k = offs < K
        k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + offs, mask=mask_k, other=0.0)
        # Compute dot over K by iterating elements (K is 128, so simple loop is fine)
        for i in range(0, K):
            # state[b, h, i, :] is [V]
            s_off = b_idx * stride_state_b + h_idx * stride_state_h + i * stride_state_k + tl.arange(0, V)
            s_vec = tl.load(state_ptr + s_off, mask=tl.arange(0, V) < V, other=0.0)
            acc += k_vec[i] * tl.sum(s_vec, axis=0)
    tl.store(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h, acc)


@triton.jit
def kernel_update_and_output(
    k_ptr, beta_ptr, v_ptr, state_in_ptr, q_ptr,
    new_state_ptr, out_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_beta_b, stride_beta_h,
    stride_v_b, stride_v_h, stride_v_v,
    stride_si_b, stride_si_h, stride_si_v, stride_si_k,
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
    stride_q_b, stride_q_h, stride_q_k,
    stride_out_b, stride_out_h,
    scale,  # float32 scalar
    num_warps: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load scalars
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)
    # k[b, h] as [K]
    k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + tl.arange(0, K))
    # tmp_old_v = dot(k, state[:, :]) where state[:, :] = state[b, h, :, :]
    tmp_old_v = 0.0
    for kk in range(0, K, 128):
        offs = kk + tl.arange(0, 128)
        mask_k = offs < K
        k_sub = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + offs, mask=mask_k, other=0.0)
        for i in range(0, K):
            s_off = b_idx * stride_si_b + h_idx * stride_si_h + i * stride_si_k + tl.arange(0, V)
            s_vec = tl.load(state_in_ptr + s_off, mask=tl.arange(0, V) < V, other=0.0)
            tmp_old_v += k_sub[i] * tl.sum(s_vec, axis=0)

    # Compute new state elementwise across [V, K]
    new_state_buf = tl.zeros((V, K), dtype=tl.float32)
    # Load v[b, h, :]
    v_vec = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + tl.arange(0, V))
    # old_state = beta * v + (1 - beta) * (tmp_old_v * ones)
    beta_scaled = beta_val * v_vec
    const_term = (1.0 - beta_val) * tmp_old_v
    for i in range(0, V):
        base = b_idx * stride_si_b + h_idx * stride_si_h + i * stride_si_k  # state[b,h,i,0] offset base, we need full column
        # We'll update each row i by looping over K
        for kk in range(0, K):
            old_col = tl.load(state_in_ptr + base + kk * stride_si_k)
            new_col = old_col - (k_vec[kk] * tl.sum(old_col, axis=0)) + k_vec[kk] * (beta_scaled[i] + const_term)
            new_state_buf[i, kk] = new_col

    # Store new_state_out [B, H, V, K]
    for i in range(0, V):
        row_base = b_idx * stride_ns_b + h_idx * stride_ns_h + i * stride_ns_v
        for kk in range(0, K):
            tl.store(new_state_ptr + row_base + kk * stride_ns_k, new_state_buf[i, kk])

    # Compute output = scale * (q[b, h] @ new_state_buf)
    q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + tl.arange(0, K))
    out_val = 0.0
    for kk in range(0, K):
        out_val += q_vec[kk] * tl.sum(new_state_buf[kk, :], axis=0)
    out_val *= scale
    tl.store(out_ptr + b_idx * stride_out_b + h_idx * stride_out_h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure all inputs are on the same device and float32, contiguous
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All tensors must be on CUDA for Triton kernels."
        # Shapes from inputs
        B = q.shape[0]
        QH = q.shape[2]  # query heads, typically 4
        KH = k.shape[2]  # key heads, typically 4
        VH = v.shape[2]  # value heads, typically 8
        V = v.shape[3]
        K = q.shape[3]
        H = QH  # per the original logic, H is the number of query heads (4)
        assert H == 4, "This implementation expects num_q_heads=4."
        assert QH == 4 and KH == 4, "This implementation expects num_q_heads=4 and num_k_heads=4."
        assert VH == 8 and V == 128 and K == 128, "This implementation expects num_v_heads=8, V=128, K=128."

        a32 = a.to(torch.float32).contiguous()
        dt_bias32 = dt_bias.to(torch.float32).contiguous()
        b32 = b.to(torch.float32).contiguous()
        A_log32 = A_log.to(torch.float32).contiguous()
        q32 = q.to(torch.float32).contiguous()  # [B, 1, 4, 128]
        k32 = k.to(torch.float32).contiguous()  # [B, 1, 4, 128]
        v32 = v.to(torch.float32).contiguous()  # [B, 1, 8, 128]
        state32 = state.to(torch.float32).contiguous()  # [B, 8, 128, 128]

        # Allocate g and beta [B, H]
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)

        # Kernel: g and beta
        grid = (B, H)
        kernel_g_beta[grid](
            A_log32, a32, dt_bias32, b32,
            g, beta,
            B, H,
            A_log32.stride(0), a32.stride(0), a32.stride(1), dt_bias32.stride(0), b32.stride(0), b32.stride(1),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
            num_warps=1
        )

        # tmp_old_v: [B, H]
        k2 = k32.view(B, H, K).contiguous()  # [B, H, K]
        state2 = state32.view(B, H, V, K).contiguous()  # [B, H, V, K]
        tmp_old_v = torch.empty((B, H), dtype=torch.float32, device=device)
        grid_tmp = (B, H)
        kernel_tmp_old_v[grid_tmp](
            k2, state2, tmp_old_v,
            V, K,
            k2.stride(0), k2.stride(1), k2.stride(2),
            state2.stride(0), state2.stride(1), state2.stride(2), state2.stride(3),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
            num_warps=1
        )

        # new_state_out: [B, H, V, K]
        new_state_out = torch.empty((B, H, V, K), dtype=torch.float32, device=device)
        # out: [B, H] (we'll reshape to [B, 1, H] after)
        out_vec = torch.empty((B, H), dtype=torch.float32, device=device)

        # Prepare v2 [B, H, V]
        v2 = v32.view(B, H, V).contiguous()

        # Kernel: update state and output
        grid2 = (B, H)
        kernel_update_and_output[grid2](
            k2, beta, v2, state2, q32, new_state_out, out_vec,
            B, H, V, K,
            k2.stride(0), k2.stride(1), k2.stride(2),
            beta.stride(0), beta.stride(1),
            v2.stride(0), v2.stride(1), v2.stride(2),
            state2.stride(0), state2.stride(1), state2.stride(2), state2.stride(3),
            new_state_out.stride(0), new_state_out.stride(1), new_state_out.stride(2), new_state_out.stride(3),
            q32.stride(0), q32.stride(1), q32.stride(2),
            out_vec.stride(0), out_vec.stride(1),
            float(scale),
            num_warps=2
        )

        # Return: output in bfloat16 [B, 1, H], new_state in float32 [B, H, V, K]
        output_bf16 = out_vec.view(B, 1, H).to(torch.bfloat16)
        return output_bf16, new_state_out


def run(*args):
    return ModelNew()(*args)
