import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    B, H,
    stride_Al, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
    stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
):
    # grid = (B, H)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # a[b, h] and dt_bias[h]
    a_val = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h)
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt)
    # A_log[h]
    A_val = tl.load(A_log_ptr + h_idx * stride_Al)

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt))
    # g = exp(-exp(A) * softplus(a + dt))
    g_val = tl.exp(-tl.exp(A_val) * sp)

    # beta = sigmoid(b[b, h]) = 1 / (1 + exp(-b))
    b_val = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # store
    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,         # [B, H, K]
    stride_si_b, stride_si_h, stride_si_v, stride_si_k,  # [B, H, V, K]
    stride_tmp_bh,                                # [B, H]
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # tmp_old_v[b, h] = sum_k k[b,h,k] * state[b,h,k,:]
    acc = 0.0
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        # Load k vector [128]
        k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_sub * stride_k_k, mask=k_mask, other=0.0)
        # For each k in this chunk, accumulate sum_v k * state[k, v] over v
        for kk in range(0, 128):
            k_idx = k_start + kk
            if k_idx < K:
                # For each v, load state[k_idx, v, :] and sum
                for v_start in range(0, V, 128):
                    v_sub = v_start + tl.arange(0, 128)
                    v_mask = v_sub < V
                    # state_ptr index: b*stride_si_b + h*stride_si_h + v*stride_si_v + k*stride_si_k
                    state_ptrs = state_ptr + b_idx * stride_si_b + h_idx * stride_si_h + v_sub * stride_si_v + k_idx * stride_si_k
                    state_vals = tl.load(state_ptrs, mask=v_mask, other=0.0)  # [128]
                    acc += tl.sum(k_vec[kk] * state_vals, axis=0)
    tl.store(tmp_ptr + b_idx * stride_tmp_bh + h_idx * stride_tmp_bh, acc)


@triton.jit
def kernel_update_state_and_output(
    k_ptr, tmp_ptr, v_ptr, beta_ptr, state_ptr, new_state_ptr, q_ptr, output_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,          # [B, H, K]
    stride_tmp_b, stride_tmp_h,                  # [B, H] tmp_old_v
    stride_v_b, stride_v_h, stride_v_v,          # [B, H, V]
    stride_be_b, stride_be_h,                    # [B, H] beta
    stride_si_b, stride_si_h, stride_si_v, stride_si_k,  # [B, H, V, K]
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,  # [B, H, V, K]
    stride_q_b, stride_q_h, stride_q_k,          # [B, H, K] but here [B, H, K] == [B, H, K]
    stride_out_b, stride_out_h, stride_out_v,    # [B, H, V]
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars
    tmp_old = tl.load(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h)  # scalar
    beta_val = tl.load(beta_ptr + b_idx * stride_be_b + h_idx * stride_be_h)  # scalar
    # Load vectors
    k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + tl.arange(0, K) * stride_k_k, mask=tl.arange(0, K) < K, other=0.0)  # [K]
    q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + tl.arange(0, K) * stride_q_k, mask=tl.arange(0, K) < K, other=0.0)    # [K]

    # v[b,h,:]
    v_vals = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + tl.arange(0, V) * stride_v_v, mask=tl.arange(0, V) < V, other=0.0)  # [V]
    # new_v = beta * v + (1 - beta) * tmp_old
    new_v = beta_val * v_vals + (1.0 - beta_val) * tmp_old

    # state_remove = dot(k, tmp_old) = sum_k k[k] * tmp_old
    state_remove = tl.sum(k_vec * tmp_old, axis=0)  # scalar

    # state_update = dot(k, new_v) = sum_k k[k] * new_v[k]
    state_update = tl.sum(k_vec * new_v, axis=0)    # scalar

    # Load old_state[b,h] as [V,K] (row-major), update elementwise and store to new_state
    for v_start in range(0, V, 128):
        v_sub = v_start + tl.arange(0, 128)
        v_mask = v_sub < V
        for k_start in range(0, K, 128):
            k_sub = k_start + tl.arange(0, 128)
            k_mask = k_sub < K
            old_state_ptrs = state_ptr + b_idx * stride_si_b + h_idx * stride_si_h + v_sub[:, None] * stride_si_v + k_sub[None, :] * stride_si_k
            old_state_vals = tl.load(old_state_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0)  # [128,128]
            # new_state_vals = old_state - state_remove + state_update
            new_state_vals = old_state_vals - state_remove + state_update
            new_state_ptrs = new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v_sub[:, None] * stride_ns_v + k_sub[None, :] * stride_ns_k
            tl.store(new_state_ptrs, new_state_vals, mask=v_mask[:, None] & k_mask[None, :])

    # output[b,h] = scale * (q · new_state) where new_state is [V,K] and q is [K]
    # Compute dot(q, new_state[b,h]) = sum_k q[k] * sum_v new_state[k,v]
    # We can compute per-k contribution then reduce
    dot_q = 0.0
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        # For each k in this chunk, compute sum_v new_state[k,v]
        for v_start in range(0, V, 128):
            v_sub = v_start + tl.arange(0, 128)
            v_mask = v_sub < V
            ns_ptrs = new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v_sub[:, None] * stride_ns_v + k_sub[None, :] * stride_ns_k
            ns_vals = tl.load(ns_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0)  # [128,128]
            per_k_sum = tl.sum(ns_vals, axis=0)  # [128]
            dot_q += tl.sum(q_vec[k_start:k_start+128] * per_k_sum, axis=0)
    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_h + tl.arange(0, V) * stride_out_v, dot_q, mask=tl.arange(0, V) < V)


@triton.jit
def kernel_output_scalar(
    q_ptr, new_state_ptr, output_ptr,
    B, H, V, K,
    stride_q_b, stride_q_h, stride_q_k,
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
    stride_out_b, stride_out_h, stride_out_v,
):
    # This kernel writes output[b,h,0] = scale * dot(q[b,h], new_state[b,h])
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + tl.arange(0, K) * stride_q_k, mask=tl.arange(0, K) < K, other=0.0)
    dot = 0.0
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        for v_start in range(0, V, 128):
            v_sub = v_start + tl.arange(0, 128)
            v_mask = v_sub < V
            ns_ptrs = new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v_sub[:, None] * stride_ns_v + k_sub[None, :] * stride_ns_k
            ns_vals = tl.load(ns_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0)  # [128,128]
            per_k_sum = tl.sum(ns_vals, axis=0)  # [128]
            dot += tl.sum(q_vec[k_start:k_start+128] * per_k_sum, axis=0)
    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_h, dot)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device and dtype; we'll compute in FP32
        device = q.device
        B = q.shape[0]
        # Shapes from original: q [B,1,4,K], k [B,1,4,K], v [B,1,8,V], state [B,8,V,K]
        q_ = q.squeeze(1).contiguous()       # [B, 4, K]
        k_ = k.squeeze(1).contiguous()       # [B, 4, K]
        v_ = v.squeeze(1).contiguous()       # [B, 8, V]
        if state is None:
            state_in = torch.zeros(B, 8, 128, 128, dtype=torch.float32, device=device)
        else:
            state_in = state.squeeze(1).contiguous()  # [B, 8, V, K]

        # Prepare parameters
        a_t = a.squeeze(1).to(torch.float32).contiguous()      # [B, H=8]
        dt_bias_t = dt_bias.to(torch.float32).contiguous()     # [H=8]
        b_t = b.squeeze(1).to(torch.float32).contiguous()      # [B, H=8]
        A_log_t = A_log.to(torch.float32).contiguous()         # [H=8]

        # Allocate outputs
        g = torch.empty(B, 8, dtype=torch.float32, device=device)
        beta = torch.empty(B, 8, dtype=torch.float32, device=device)
        tmp_old_v = torch.empty(B, 8, dtype=torch.float32, device=device)
        new_state_out = torch.empty(B, 8, 128, 128, dtype=torch.float32, device=device)  # [B, H, V, K]
        output = torch.empty(B, 8, 128, dtype=torch.float32, device=device)              # [B, H, V]

        # Launch kernels
        grid = (B, 8)
        kernel_g_beta[grid](
            A_log_t, a_t, dt_bias_t, b_t,
            g, beta,
            B, 8,
            A_log_t.stride(0), a_t.stride(0), a_t.stride(1), dt_bias_t.stride(0), b_t.stride(0), b_t.stride(1),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
        )
        kernel_tmp_old_v[grid](
            k_, state_in, tmp_old_v,
            B, 8, 128, 128,
            k_.stride(0), k_.stride(1), k_.stride(2),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
        )
        # Update state and compute output scalar
        kernel_update_state_and_output[grid](
            k_, tmp_old_v, v_, beta, state_in, new_state_out, q_, output,
            B, 8, 128, 128,
            k_.stride(0), k_.stride(1), k_.stride(2),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
            v_.stride(0), v_.stride(1), v_.stride(2),
            beta.stride(0), beta.stride(1),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            new_state_out.stride(0), new_state_out.stride(1), new_state_out.stride(2), new_state_out.stride(3),
            q_.stride(0), q_.stride(1), q_.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
        )

        # Return bfloat16 output and float32 new_state, matching original intent
        output_bf16 = output.to(torch.bfloat16)  # [B, 8, 128]
        return output_bf16, new_state_out


def run(*args):
    return ModelNew()(*args)
