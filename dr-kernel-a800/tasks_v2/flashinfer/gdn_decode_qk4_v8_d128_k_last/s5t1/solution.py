import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a[0,0,h] + dt_bias[h])) for h in [0..H-1]
    a_ptr: [1, 1, H] flattened to [H]
    dt_bias_ptr: [H]
    A_log_ptr: [H]
    g_ptr: [H]
    """
    h = tl.program_id(0)
    a_val = tl.load(a_ptr + h)
    dt_val = tl.load(dt_bias_ptr + h)
    A_val = tl.load(A_log_ptr + h)
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * sp)
    tl.store(g_ptr + h, g_val)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, H):
    """
    Compute beta[h] = sigmoid(b[0,0,h]) for h in [0..H-1]
    b_ptr: [1, 1, H] flattened to [H]
    beta_ptr: [H]
    """
    h = tl.program_id(0)
    b_val = tl.load(b_ptr + h)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + h, beta_val)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, out_ptr,
    B, H, V, K, scale
):
    """
    For each (b, h), update new_state[b,h] and compute output[b,h].
    q_ptr, k_ptr, v_ptr are [B, H, K]
    state_ptr is [B, H, V, K]
    g_ptr, beta_ptr are [H]
    out_ptr is [B, H, V] (we'll write per element)
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Base offsets
    q_base = b * (H * K) + h * K
    k_base = b * (H * K) + h * K
    v_base = b * (H * V * K) + h * (V * K)

    # Load vectors q_h, k_h
    q_vec = [0.0] * K
    k_vec = [0.0] * K
    for i in range(0, K):
        q_vec[i] = tl.load(q_ptr + q_base + i)
        k_vec[i] = tl.load(k_ptr + k_base + i)

    # Load g[h], beta[h]
    g_val = tl.load(g_ptr + h)
    beta_val = tl.load(beta_ptr + h)

    # Compute old_v = k @ state_old, where state_old = state[b,h] (k-last: [V,K])
    old_v = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            state_elem = tl.load(state_ptr + b * (H * V * K) + h * (V * K) + i * K + k)
            sum_k += k_vec[k] * state_elem
        old_v[i] = sum_k

    # Compute new_v = beta * v + (1 - beta) * old_v
    new_v = [0.0] * V
    for i in range(0, V):
        v_elem = tl.load(v_ptr + v_base + i * K)  # v[b,h,i]
        new_v[i] = beta_val * v_elem + (1.0 - beta_val) * old_v[i]

    # old_state = g * state_old
    old_state = [[0.0] * K for _ in range(0, V)]
    for i in range(0, V):
        for k in range(0, K):
            state_elem = tl.load(state_ptr + b * (H * V * K) + h * (V * K) + i * K + k)
            old_state[i][k] = g_val * state_elem

    # state_remove = k @ old_state (reduce over K)
    state_remove = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_vec[k] * old_state[i][k]
        state_remove[i] = sum_k

    # state_update = k @ new_v (reduce over K)
    state_update = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_vec[k] * new_v[i]
        state_update[i] = sum_k

    # Write new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None]
    new_state_base = b * (H * V * K) + h * (V * K)
    for i in range(0, V):
        for k in range(0, K):
            val = old_state[i][k] - state_remove[i] + state_update[i]
            tl.store(state_ptr + new_state_base + i * K + k, val)

    # Compute output[b,h] = scale * sum_i q[i] * new_state[b,h,i]
    out_val = 0.0
    for i in range(0, V):
        row_sum = 0.0
        for k in range(0, K):
            row_sum += q_vec[k] * tl.load(state_ptr + new_state_base + i * K + k)
        out_val += row_sum
    out_val = out_val * scale

    # Store output[b,h] to out[b,H*V + h] (we use a 1D out_ptr of size B*H*V)
    tl.store(out_ptr + b * (H * V) + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only forward:
        - q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128]
        - state: [B, 8, 128, 128]
        - A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        Returns: (output [B, H, V] cast to bfloat16), new_state [B, H, V, K] float32
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All tensors must be on CUDA for Triton."
        device = q.device

        # Squeeze T=1
        q_s = q.squeeze(1)  # [B, 4, K]
        k_s = k.squeeze(1)  # [B, 4, K]
        v_s = v.squeeze(1)  # [B, 8, V]

        # Flatten parameters for Triton
        B = q_s.shape[0]
        H = v_s.shape[1]
        V = v_s.shape[2]
        K = q_s.shape[3]

        a_flat = a[0, 0, :].contiguous()     # [H], ensure contiguous
        dt_bias_flat = dt_bias.contiguous()  # [H]
        b_flat = b[0, 0, :].contiguous()     # [H]
        A_log_flat = A_log.contiguous()      # [H]

        # Compute g and beta using Triton kernels
        g = torch.empty(H, dtype=torch.float32, device=device)
        beta = torch.empty(H, dtype=torch.float32, device=device)
        grid_g = (H,)
        _compute_g_kernel[grid_g](a_flat, dt_bias_flat, A_log_flat, g, H)
        grid_beta = (H,)
        _compute_beta_kernel[grid_beta](b_flat, beta, H)

        # Ensure q_s, k_s, v_s are float32 for compute
        q_f32 = q_s.to(torch.float32).contiguous()
        k_f32 = k_s.to(torch.float32).contiguous()
        v_f32 = v_s.to(torch.float32).contiguous()
        state_f32 = state.to(torch.float32).contiguous()

        # Output buffer [B, H, V] (float32), we will cast to bfloat16 after
        out = torch.empty(B * H * V, dtype=torch.float32, device=device)

        # Launch Triton kernel to update state and compute output for all (b,h)
        grid_up = (B * H,)
        _update_all_kernel[grid_up](
            q_f32, k_f32, v_f32, state_f32, g, beta, out,
            B, H, V, K, float(scale)
        )

        # Reshape output to [B, H, V] and cast to bfloat16
        out = out.view(B, H, V).to(torch.bfloat16)

        # new_state is updated in-place in state_f32 by the Triton kernel
        return out, state_f32


def run(*args):
    return ModelNew()(*args)
