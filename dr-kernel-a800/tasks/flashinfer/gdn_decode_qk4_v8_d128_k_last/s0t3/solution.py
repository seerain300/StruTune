import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(A_ptr, a_ptr, dt_bias_ptr, b_ptr,
                  g_ptr, beta_ptr,
                  B, H,
                  stride_A, stride_a_b, stride_dt, stride_b_b,
                  stride_g_b, stride_beta_b):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load parameters
    A_val = tl.load(A_ptr + h_idx * stride_A)
    a_val = tl.load(a_ptr + b_idx * stride_a_b + h_idx)
    dt_bias_val = tl.load(dt_bias_ptr + h_idx * stride_dt)
    b_val = tl.load(b_ptr + b_idx * stride_b_b + h_idx)

    # g = exp(-exp(A_log) * softplus(a + dt_bias))
    A_exp = tl.exp(A_val)
    a_sum = a_val + dt_bias_val
    softplus = tl.log(1.0 + tl.exp(a_sum))  # softplus(x) = log(1 + exp(x))
    g = tl.exp(-A_exp * softplus)

    # beta = sigmoid(b)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b_idx * stride_g_b + h_idx, g)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx, beta)


@triton.jit
def kernel_tmp_old_v(k_ptr, state_ptr, tmp_ptr,
                     B, H, V, K,
                     stride_k_b, stride_k_h,       # k strides for [B, H, K]
                     stride_state_b, stride_state_h, stride_state_v, stride_state_k,  # state strides for [B, H, V, K]
                     stride_tmp_bh):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load k[b, h] as [K]
    k_off = b_idx * stride_k_b + h_idx * stride_k_h
    k_vec = tl.load(k_ptr + k_off, mask=tl.arange(0, K) < K, other=0.0)

    # tmp_old_v = sum_k k[k] * state[k, :]
    tmp = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        for kk in range(0, 128):
            k_idx = k_start + kk
            if k_idx < K:
                row_ptr = state_ptr + b_idx * stride_state_b + h_idx * stride_state_h + k_idx * stride_state_k
                state_row = tl.load(row_ptr, mask=tl.arange(0, V) < V, other=0.0)  # [V]
                tmp += k_vec[k_idx] * state_row

    # Store scalar tmp_old_v
    tl.store(tmp_ptr + b_idx * stride_tmp_bh + h_idx, tmp)


@triton.jit
def kernel_update_state_and_output(k_ptr, tmp_ptr, v_ptr, beta_ptr, state_in_ptr, state_out_ptr, q_ptr, output_ptr,
                                   B, H, V, K,
                                   stride_k_b, stride_k_h,       # k strides [B, H, K]
                                   stride_tmp_bh,              # tmp strides [B, H]
                                   stride_v_b, stride_v_h, stride_v_v, stride_v_k,   # v strides [B, H, V, K] but v is [B, H, V]; stride_v_k may not be used, pass any
                                   stride_beta_b, stride_beta_h,  # beta strides [B, H]
                                   stride_state_in_b, stride_state_in_h, stride_state_in_v, stride_state_in_k,
                                   stride_state_out_b, stride_state_out_h, stride_state_out_v, stride_state_out_k,
                                   stride_q_b, stride_q_h, stride_q_k,
                                   stride_output_b, stride_output_h, stride_output_v):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Scalars
    tmp_old = tl.load(tmp_ptr + b_idx * stride_tmp_bh + h_idx)  # scalar
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx)

    # Load k[b, h] as [K]
    k_off = b_idx * stride_k_b + h_idx * stride_k_h
    k_vec = tl.load(k_ptr + k_off, mask=tl.arange(0, K) < K, other=0.0)

    # Load v[b, h] as [V]
    v_off = b_idx * stride_v_b + h_idx * stride_v_h
    v_vec = tl.load(v_ptr + v_off, mask=tl.arange(0, V) < V, other=0.0)

    # new_v = beta * v + (1 - beta) * tmp_old_v
    new_v_vec = beta_val * v_vec + (1.0 - beta_val) * tmp_old  # vector [V]

    # state_remove = dot(k, tmp_old_v) scalar
    state_remove = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        for kk in range(0, 128):
            k_idx = k_start + kk
            if k_idx < K:
                state_remove += k_vec[k_idx] * tmp_old

    # state_update = dot(k, new_v) vector [V]
    state_update = tl.zeros([V], dtype=tl.float32)
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        for kk in range(0, 128):
            k_idx = k_start + kk
            if k_idx < K:
                state_update += k_vec[k_idx] * new_v_vec

    # Read old_state[b, h] as [V, K] and update to new_state[b, h]
    for v_start in range(0, V, 128):
        v_idx = v_start + tl.arange(0, 128)
        v_mask = v_idx < V
        for k_start in range(0, K, 128):
            k_sub = k_start + tl.arange(0, 128)
            k_mask = k_sub < K
            old_state_ptrs = state_in_ptr + b_idx * stride_state_in_b + h_idx * stride_state_in_h + k_sub[None, :] * stride_state_in_k + v_idx[:, None] * stride_state_in_v
            old_state_vals = tl.load(old_state_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0)
            new_state_vals = old_state_vals - state_remove + state_update[None, :]
            new_state_ptrs = state_out_ptr + b_idx * stride_state_out_b + h_idx * stride_state_out_h + k_sub[None, :] * stride_state_out_k + v_idx[:, None] * stride_state_out_v
            tl.store(new_state_ptrs, new_state_vals, mask=v_mask[:, None] & k_mask[None, :])

    # Compute output[b, h] = scale * (q[b, h] @ new_state[b, h])
    q_off = b_idx * stride_q_b + h_idx * stride_q_h
    q_vec = tl.load(q_ptr + q_off, mask=tl.arange(0, K) < K, other=0.0)

    output_scalar = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        for k_idx in range(0, 128):
            kk = k_start + k_idx
            if kk < K:
                col_ptr = state_out_ptr + b_idx * stride_state_out_b + h_idx * stride_state_out_h + kk * stride_state_out_k
                col = tl.load(col_ptr, mask=tl.arange(0, V) < V, other=0.0)  # [V]
                output_scalar += q_vec[kk] * tl.sum(col)

    # Store output[b, h, V] as [B, H, V]; here we store a V-length vector
    out_ptr = output_ptr + b_idx * stride_output_b + h_idx * stride_output_h
    for j in range(0, V):
        tl.store(out_ptr + j * stride_output_v, output_scalar)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward that computes:
          - g = exp(-exp(A_log) * softplus(a + dt_bias)) [B, H]
          - beta = sigmoid(b) [B, H]
          - tmp_old_v[b, h] = dot(k[b, h], state[b, h]) [B, H]
          - new_state[b, h] updated elementwise: old - (k·old) + (k·(beta*v + (1-beta)*tmp_old_v))
          - output[b, h] = scale * (q[b, h] · new_state[b, h])
        Returns output [B, H, V] (bfloat16) and new_state [B, H, V, K] (float32).
        """
        # Ensure device consistency
        device = q.device
        assert k.device == device and v.device == device and A_log.device == device and a.device == device and dt_bias.device == device and b.device == device, "All tensors must be on the same device."

        # Cast inputs for computation and make contiguous
        q_s = q.squeeze(1).contiguous().to(torch.float32)   # [B, QH, K] -> [B, H, K]
        k_s = k.squeeze(1).contiguous().to(torch.float32)   # [B, KH, K] -> [B, H, K]
        v_s = v.squeeze(1).contiguous().to(torch.float32)   # [B, VH, V] -> [B, H, V]
        A_log = A_log.contiguous().to(torch.float32)        # [H]
        a = a.contiguous().to(torch.float32)                # [B, H]
        dt_bias = dt_bias.contiguous().to(torch.float32)    # [H]
        b = b.contiguous().to(torch.float32)                # [B, H]

        B = q_s.shape[0]
        H = v_s.shape[1]
        V = v_s.shape[2]
        K = q_s.shape[2]

        if state is None:
            state_in = torch.zeros(B, H, V, K, dtype=torch.float32, device=device)
        else:
            state_in = state.contiguous().to(torch.float32)

        # Allocate outputs
        g = torch.empty(B, H, dtype=torch.float32, device=device)
        beta = torch.empty(B, H, dtype=torch.float32, device=device)
        tmp_old_v = torch.empty(B, H, dtype=torch.float32, device=device)
        new_state_out = torch.empty(B, H, V, K, dtype=torch.float32, device=device)
        output = torch.empty(B, H, V, dtype=torch.float32, device=device)

        # Launch Triton kernels
        grid = (B, H)
        kernel_g_beta[grid](
            A_log, a, dt_bias, b,
            g, beta,
            B, H,
            A_log.stride(0), a.stride(0), dt_bias.stride(0), b.stride(0),
            g.stride(0), beta.stride(0),
        )

        kernel_tmp_old_v[grid](
            k_s, state_in, tmp_old_v,
            B, H, V, K,
            k_s.stride(0), k_s.stride(1),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            tmp_old_v.stride(0),
        )

        kernel_update_state_and_output[grid](
            k_s, tmp_old_v, v_s, beta, state_in, new_state_out, q_s, output,
            B, H, V, K,
            k_s.stride(0), k_s.stride(1),
            tmp_old_v.stride(0),
            v_s.stride(0), v_s.stride(1), v_s.stride(2), 0,  # v strides for [B, H, V]; stride_v_k unused
            beta.stride(0), beta.stride(1),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            new_state_out.stride(0), new_state_out.stride(1), new_state_out.stride(2), new_state_out.stride(3),
            q_s.stride(0), q_s.stride(1), q_s.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, new_state_out


def run(*args):
    return ModelNew()(*args)
