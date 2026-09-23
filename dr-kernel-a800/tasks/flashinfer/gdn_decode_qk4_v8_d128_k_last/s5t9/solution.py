import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H: tl.constexpr):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a[0,0,h] + dt_bias[h])) for h in [0..H-1]
    a_ptr: 1D contiguous [H], elements are a[b,1,h] (dtype float32)
    dt_bias_ptr: 1D contiguous [H], elements are dt_bias[h] (dtype float32)
    A_log_ptr: 1D contiguous [H], elements are A_log[h] (dtype float32)
    g_ptr: 1D contiguous [H] to store result (dtype float32)
    """
    h = tl.program_id(0)
    a_val = tl.load(a_ptr + h)  # scalar
    dt_val = tl.load(dt_bias_ptr + h)  # scalar
    A_val = tl.load(A_log_ptr + h)  # scalar
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * sp)
    tl.store(g_ptr + h, g_val)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, H: tl.constexpr):
    """
    Compute beta[h] = sigmoid(b[0,0,h]) for h in [0..H-1]
    b_ptr: 1D contiguous [H], elements are b[b,1,h] (dtype float32)
    beta_ptr: 1D contiguous [H] to store result (dtype float32)
    """
    h = tl.program_id(0)
    b_val = tl.load(b_ptr + h)  # scalar
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + h, beta_val)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, out_ptr,
    B, H, V, K, scale
):
    """
    For each (b, h):
      - Load q_h, k_h, v_h
      - Compute old_v = k_h @ state[b,h] (reduce over K)
      - Compute new_v = beta[h] * v_h + (1 - beta[h]) * old_v
      - old_state = g[h] * state[b,h]
      - state_remove = k_h @ old_state (reduce over K)
      - state_update = k_h @ new_v (reduce over K)
      - new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None]
      - out[b,h] = scale * (q_h @ new_state[b,h]) (reduce over V)
    q_ptr: [B, H, K] float32
    k_ptr: [B, H, K] float32
    v_ptr: [B, H, V] float32
    state_ptr: [B, H, V, K] float32
    g_ptr: [H] float32
    beta_ptr: [H] float32
    out_ptr: [B, H] float32
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load q_h, k_h, v_h (1D vectors)
    q_h = tl.load(q_ptr + b * (H * K) + h * K)  # [K]
    k_h = tl.load(k_ptr + b * (H * K) + h * K)  # [K]
    v_h = tl.load(v_ptr + b * (H * V) + h * V)  # [V]

    # Load params
    g_val = tl.load(g_ptr + h)  # scalar
    beta_val = tl.load(beta_ptr + h)  # scalar

    # Load state[b,h] as [V, K]
    state_base = b * (H * V * K) + h * (V * K)
    # Initialize old_state[i, k] for i in [0..V-1], k in [0..K-1]
    # We'll build 2D arrays via row-wise accumulation.
    # Note: Triton doesn't support Python loops over tensors, but we can implement reductions manually.
    old_state = [[0.0] * K for _ in range(0, V)]
    for i in range(0, V):
        for k in range(0, K):
            old_state[i][k] = tl.load(state_ptr + state_base + i * K + k)

    # Compute old_v = k_h @ old_state (reduce over K) -> [V]
    old_v = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_h[k] * old_state[i][k]
        old_v[i] = sum_k

    # Compute new_v = beta * v_h + (1 - beta) * old_v
    new_v = [0.0] * V
    for i in range(0, V):
        new_v[i] = beta_val * v_h[i] + (1.0 - beta_val) * old_v[i]

    # Compute state_remove = k_h @ old_state (reduce over K) -> [V]
    state_remove = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_h[k] * old_state[i][k]
        state_remove[i] = sum_k

    # Compute state_update = k_h @ new_v (reduce over K) -> [V]
    state_update = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_h[k] * new_v[i]
        state_update[i] = sum_k

    # Write new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None]
    new_state_base = b * (H * V * K) + h * (V * K)
    for i in range(0, V):
        for k in range(0, K):
            val = old_state[i][k] - state_remove[i] + state_update[i]
            tl.store(state_ptr + new_state_base + i * K + k, val)

    # Compute output[b,h] = scale * (q_h @ new_state[b,h]) (reduce over V)
    out_val = 0.0
    for i in range(0, V):
        sum_i = 0.0
        for k in range(0, K):
            sum_i += new_state_ptr[b, h, i, k]
        out_val += scale * sum_i  # scale is float scalar
    tl.store(out_ptr + b * H + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original run function.
        Returns (output [B, H] cast to bfloat16, new_state [B, H, V, K] float32).
        """
        device = q.device
        # Extract B, H, V, K from shapes (num_v_heads = H = 8 as per original)
        B = q.shape[0]
        H = v.shape[1]  # num_v_heads = 8
        V = v.shape[2]  # 128
        K = q.shape[3]  # 128

        # Ensure T=1 squeeze
        q = q.squeeze(1)  # [B, 4, K]
        k = k.squeeze(1)  # [B, 4, K]
        v = v.squeeze(1)  # [B, 8, V]

        # Expand q/k to H=8 to match original run behavior (repeat_interleave)
        q_exp = q.repeat_interleave(H // q.shape[1], dim=1)  # [B, 8, K]
        k_exp = k.repeat_interleave(H // k.shape[1], dim=1)  # [B, 8, K]

        # Prepare inputs for Triton kernels
        # Flatten a, dt_bias, A_log, b to 1D contiguous vectors [H] and [B]
        a_flat = a.reshape(-1).contiguous()           # [B]
        dt_bias_flat = dt_bias.contiguous()           # [H]
        A_log_flat = A_log.contiguous()               # [H]
        b_flat = b.reshape(-1).contiguous()           # [B]
        g = torch.empty(H, dtype=torch.float32, device=device)
        beta = torch.empty(H, dtype=torch.float32, device=device)

        # Launch kernels to compute g and beta
        grid_g = (H,)
        _compute_g_kernel[grid_g](a_flat, dt_bias_flat, A_log_flat, g, H)
        grid_beta = (H,)
        _compute_beta_kernel[grid_beta](b_flat, beta, H)

        # Prepare output buffer [B, H] float32
        out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Ensure state is float32 and contiguous in [B, H, V, K] layout
        if state.dtype != torch.float32:
            state = state.float()
        state = state.contiguous()  # [B, H, V, K]

        # Launch update kernel
        grid_update = (B, H)
        _update_all_kernel[grid_update](
            q_exp, k_exp, v, state, g, beta, out,
            B, H, V, K, scale
        )

        # Return output cast to bfloat16 and new_state
        return (out.to(torch.bfloat16), state)


def run(*args):
    return ModelNew()(*args)
