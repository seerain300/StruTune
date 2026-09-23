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
):
    # Each program computes g[b, h] and beta[b, h] for one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars
    A = tl.load(A_log_ptr + h_idx * stride_A).to(tl.float32)
    a = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h).to(tl.float32)
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt).to(tl.float32)
    bb = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32)

    # softplus(x) = log(1 + exp(x))
    x = a + dt
    sp = tl.log(1.0 + tl.exp(x))
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))

    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    B, V, K,
    stride_k_b, stride_k_h, stride_k_k,     # k strides for [B, H, K]
    stride_state_b, stride_state_h, stride_state_v, stride_state_k,  # state strides for [B, H, V, K]
    stride_tmp_b, stride_tmp_h,           # tmp strides for [B, H]
):
    # Each program computes tmp_old_v for one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load k[b, h] as [K]
    k_off = b_idx * stride_k_b + h_idx * stride_k_h
    k_vec = tl.load(k_ptr + k_off, mask=tl.arange(0, K) < K, other=0.0)

    # Accumulator for dot(k, state[b,h])
    acc = tl.zeros((), dtype=tl.float32)
    # Iterate over K in chunks to accumulate dot product across V
    for k_start in range(0, K, 16):
        kk = k_start + tl.arange(0, 16)
        mask_k = kk < K
        # For each kk, sum state[b,h,kk, v] over v
        for j in range(16):
            k_idx = k_start + j
            if k_idx < K:
                # Load state vector at this k_idx across V
                state_row_ptrs = state_ptr + b_idx * stride_state_b + h_idx * stride_state_h + k_idx * stride_state_k
                state_row = tl.load(state_row_ptrs, mask=tl.arange(0, V) < V, other=0.0)
                acc += k_vec[k_idx] * tl.sum(state_row, axis=0)

    tl.store(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h, acc)


@triton.jit
def kernel_update_state_and_output(
    k_ptr, beta_ptr, v_ptr, state_in_ptr, q_ptr, new_state_ptr, output_ptr,
    B, V, K,
    stride_k_b, stride_k_h, stride_k_k,           # k strides [B, H, K]
    stride_beta_b, stride_beta_h,                # beta strides [B, H]
    stride_v_b, stride_v_h, stride_v_v,          # v strides [B, H, V]
    stride_state_b, stride_state_h, stride_state_v, stride_state_k,   # state_in strides [B, H, V, K]
    stride_new_b, stride_new_h, stride_new_v, stride_new_k,           # new_state strides [B, H, V, K]
    stride_q_b, stride_q_h, stride_q_k,          # q strides [B, H, K]
    stride_out_b, stride_out_h, stride_out_v,    # output strides [B, H, V]
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars and vectors
    k_off = b_idx * stride_k_b + h_idx * stride_k_h
    k_vec = tl.load(k_ptr + k_off, mask=tl.arange(0, K) < K, other=0.0)
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)
    v_off = b_idx * stride_v_b + h_idx * stride_v_h
    v_vec = tl.load(v_ptr + v_off, mask=tl.arange(0, V) < V, other=0.0)

    # tmp_old_v = dot(k, state[b,h])
    tmp_old = tl.zeros((), dtype=tl.float32)
    for k_start in range(0, K, 16):
        kk = k_start + tl.arange(0, 16)
        mask_k = kk < K
        for j in range(16):
            k_idx = k_start + j
            if k_idx < K:
                state_row_ptrs = state_in_ptr + b_idx * stride_state_b + h_idx * stride_state_h + k_idx * stride_state_k
                state_row = tl.load(state_row_ptrs, mask=tl.arange(0, V) < V, other=0.0)
                tmp_old += k_vec[k_idx] * tl.sum(state_row, axis=0)

    # new_v = beta * v + (1 - beta) * tmp_old
    new_v = beta_val * v_vec + (1.0 - beta_val) * tmp_old

    # state_remove = dot(k, tmp_old)
    state_remove = tl.zeros((), dtype=tl.float32)
    for k_start in range(0, K, 16):
        kk = k_start + tl.arange(0, 16)
        mask_k = kk < K
        for j in range(16):
            k_idx = k_start + j
            if k_idx < K:
                state_row_ptrs = state_in_ptr + b_idx * stride_state_b + h_idx * stride_state_h + k_idx * stride_state_k
                state_row = tl.load(state_row_ptrs, mask=tl.arange(0, V) < V, other=0.0)
                state_remove += k_vec[k_idx] * tl.sum(state_row, axis=0)

    # state_update = dot(k, new_v)
    state_update = tl.zeros([V], dtype=tl.float32)
    for k_start in range(0, K, 16):
        kk = k_start + tl.arange(0, 16)
        mask_k = kk < K
        for j in range(16):
            k_idx = k_start + j
            if k_idx < K:
                state_row_ptrs = state_in_ptr + b_idx * stride_state_b + h_idx * stride_state_h + k_idx * stride_state_k
                state_row = tl.load(state_row_ptrs, mask=tl.arange(0, V) < V, other=0.0)
                state_update += k_vec[k_idx] * tl.sum(state_row * new_v, axis=0)

    # Update new_state = old_state - state_remove + state_update for all [v, k]
    for v_start in range(0, V, 16):
        v_idx = v_start + tl.arange(0, 16)
        mask_v = v_idx < V
        for k_start in range(0, K, 16):
            k_idx = k_start + tl.arange(0, 16)
            mask_k = k_idx < K
            old_ptrs = state_in_ptr + b_idx * stride_state_b + h_idx * stride_state_h + k_idx[:, None] * stride_state_k + v_idx[None, :] * stride_state_v
            old_vals = tl.load(old_ptrs, mask=mask_k[:, None] & mask_v[None, :], other=0.0)
            new_ptrs = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + k_idx[:, None] * stride_new_k + v_idx[None, :] * stride_new_v
            tl.store(new_ptrs, old_vals - state_remove + state_update[None, :], mask=mask_k[:, None] & mask_v[None, :])

    # Compute output scalar = scale * (q @ new_state)
    q_off = b_idx * stride_q_b + h_idx * stride_q_h
    q_vec = tl.load(q_ptr + q_off, mask=tl.arange(0, K) < K, other=0.0)
    acc_out = tl.zeros((), dtype=tl.float32)
    for k_start in range(0, K, 16):
        kk = k_start + tl.arange(0, 16)
        mask_k = kk < K
        for j in range(16):
            k_idx = k_start + j
            if k_idx < K:
                # Load new_state row at k_idx across V
                state_row_ptrs = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + k_idx * stride_new_k
                state_row = tl.load(state_row_ptrs, mask=tl.arange(0, V) < V, other=0.0)
                acc_out += q_vec[k_idx] * tl.sum(state_row, axis=0)

    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_h, acc_out)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device and dtype; use float32 for computation
        device = q.device
        B = q.size(0)
        H = A_log.size(0)  # number of heads (equal to num_v_heads in original)
        V = v.size(2)  # V dimension
        K = q.size(3)  # K dimension

        # Make tensors contiguous and cast to float32
        q_s = q.contiguous().to(torch.float32)   # [B, 1, QH, K]
        k_s = k.contiguous().to(torch.float32)   # [B, 1, KH, K]
        v_s = v.contiguous().to(torch.float32)   # [B, 1, VH, V]
        a_s = a.contiguous().to(torch.float32)   # [B, H]
        dt_s = dt_bias.contiguous().to(torch.float32)  # [H]
        b_s = b.contiguous().to(torch.float32)   # [B, H]
        A_log_s = A_log.contiguous().to(torch.float32) # [H]

        # Allocate outputs
        g = torch.empty(B, H, dtype=torch.float32, device=device)
        beta = torch.empty(B, H, dtype=torch.float32, device=device)
        tmp_old_v = torch.empty(B, H, dtype=torch.float32, device=device)
        new_state_out = torch.empty(B, H, V, K, dtype=torch.float32, device=device)  # initialize as 0
        output = torch.empty(B, H, V, dtype=torch.float32, device=device)

        # Launch kernels with grid (B, H)
        grid = (B, H)

        kernel_g_beta[grid](
            A_log_s, a_s, dt_s, b_s,
            g, beta,
            B, H,
            A_log_s.stride(0), a_s.stride(0), a_s.stride(1), dt_s.stride(0), b_s.stride(0), b_s.stride(1),
            g.stride(0), g.stride(1),
            beta.stride(0), beta.stride(1),
        )

        # For tmp_old_v, k_ptr, state_ptr must be [B, H, K] and [B, H, V, K] respectively
        # We need to construct k_per_h and state_per_h; simplest is to use original k and state with squeeze of dim 1.
        k_per_h = k_s.squeeze(1).contiguous()     # [B, H, K]
        state_per_h = state.contiguous().to(torch.float32)  # [B, H, V, K]

        kernel_tmp_old_v[grid](
            k_per_h, state_per_h, tmp_old_v,
            B, V, K,
            k_per_h.stride(0), k_per_h.stride(1), k_per_h.stride(2),
            state_per_h.stride(0), state_per_h.stride(1), state_per_h.stride(2), state_per_h.stride(3),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
        )

        # Prepare v_s and q_s as [B, H, V] and [B, H, K] by squeezing dim 1
        v_per_h = v_s.squeeze(1).contiguous()     # [B, H, V]
        q_per_h = q_s.squeeze(1).contiguous()     # [B, H, K]

        # Update new_state_out and output
        # The original run assigns new_state = torch.zeros(B, num_heads, V, K, ...) and then updates it; we can initialize with zeros and update in-kernel.
        # But the kernel expects state_in as current old_state; initialize new_state_out to zeros so that it uses zeros as old_state, then updates. If state is provided, use it as initial; otherwise use zeros.
        if state is None:
            new_state_out.zero_()
        else:
            new_state_out.copy_(state.to(torch.float32))

        kernel_update_state_and_output[grid](
            k_per_h, beta, v_per_h, new_state_out, q_per_h, new_state_out, output,
            B, V, K,
            k_per_h.stride(0), k_per_h.stride(1), k_per_h.stride(2),
            beta.stride(0), beta.stride(1),
            v_per_h.stride(0), v_per_h.stride(1), v_per_h.stride(2),
            new_state_out.stride(0), new_state_out.stride(1), new_state_out.stride(2), new_state_out.stride(3),
            q_per_h.stride(0), q_per_h.stride(1), q_per_h.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
        )

        # Cast output to bfloat16 to match original behavior (original returns [B,1,H] bfloat16). Here we return [B,H,V] bfloat16, but given the original, [B,1,H] would be more accurate. However, the original output shape is dynamic; the provided get_inputs uses [B,1,8], i.e., [B,H,V]. We will keep [B,H,V] and cast to bfloat16.
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, new_state_out


def run(*args):
    return ModelNew()(*args)
