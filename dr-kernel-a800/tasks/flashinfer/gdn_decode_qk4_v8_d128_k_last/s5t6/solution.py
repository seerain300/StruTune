import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H: tl.constexpr):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a[0,0,h] + dt_bias[h])) for h in [0..H-1]
    a_ptr: 1D contiguous [H] (elements are a[b,1,h])
    dt_bias_ptr: 1D contiguous [H]
    A_log_ptr: 1D contiguous [H]
    g_ptr: 1D contiguous [H]
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
def _compute_beta_kernel(b_ptr, beta_ptr, H: tl.constexpr):
    """
    Compute beta[h] = sigmoid(b[0,0,h]) for h in [0..H-1]
    b_ptr: 1D contiguous [H] (elements are b[b,1,h])
    beta_ptr: 1D contiguous [H]
    """
    h = tl.program_id(0)
    b_val = tl.load(b_ptr + h)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + h, beta_val)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, out_ptr, new_state_ptr,
    B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr, scale
):
    """
    For each (b, h), update new_state[b,h] and compute output[b,h].
    q_ptr: [B, H, K]
    k_ptr: [B, H, K]
    v_ptr: [B, H, V]
    state_ptr: [B, H, V, K] (input state)
    g_ptr: [H]
    beta_ptr: [H]
    out_ptr: [B, H] (float32)
    new_state_ptr: [B, H, V, K] (float32)
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Base offsets for q/k/v
    q_base = b * (H * K) + h * K
    k_base = b * (H * K) + h * K
    v_base = b * (H * V) + h * V

    # Load q and k vectors (K=128)
    q_vec = [0.0] * K
    k_vec = [0.0] * K
    for i in range(K):
        q_vec[i] = tl.load(q_ptr + q_base + i)
        k_vec[i] = tl.load(k_ptr + k_base + i)

    # Load params
    g_val = tl.load(g_ptr + h)
    beta_val = tl.load(beta_ptr + h)

    # Load state[b,h] as [V, K]
    state_base = b * (H * V * K) + h * (V * K)
    old_state = [[0.0] * K for _ in range(V)]
    for i in range(V):  # rows over V
        for k in range(K):  # columns over K
            old_state[i][k] = tl.load(state_ptr + state_base + i * K + k)

    # old_v = k @ state_old (reduce over K)
    old_v = [0.0] * V
    for i in range(V):
        sum_k = 0.0
        for k in range(K):
            sum_k += k_vec[k] * old_state[i][k]
        old_v[i] = sum_k

    # new_v = beta * v + (1 - beta) * old_v
    new_v = [0.0] * V
    for i in range(V):
        v_elem = tl.load(v_ptr + v_base + i)  # v[b,h,i]
        new_v[i] = beta_val * v_elem + (1.0 - beta_val) * old_v[i]

    # state_remove = k @ old_state (reduce over K) == old_v
    state_remove = [0.0] * V
    for i in range(V):
        state_remove[i] = old_v[i]

    # state_update = k @ new_v (reduce over K)
    state_update = [0.0] * V
    for i in range(V):
        sum_k = 0.0
        for k in range(K):
            sum_k += k_vec[k] * new_v[i]
        state_update[i] = sum_k

    # Compute output[b,h] = scale * sum_i q[i] * (state_remove[i] - state_update[i])
    out_val = 0.0
    for i in range(V):
        out_val += q_vec[i] * (state_remove[i] - state_update[i])

    # Write output
    tl.store(out_ptr + b * H + h, out_val)

    # Write new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None]
    new_state_base = b * (H * V * K) + h * (V * K)
    for i in range(V):
        for k in range(K):
            val = old_state[i][k] - state_remove[i] + state_update[i]
            tl.store(new_state_ptr + new_state_base + i * K + k, val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only forward:
        - Launches _compute_g_kernel, _compute_beta_kernel, _update_all_kernel.
        - Returns (output [B, H] cast to bfloat16), and new_state [B, H, V, K].
        """
        # Determine device
        device = None
        if isinstance(q, torch.Tensor):
            device = q.device
        if isinstance(k, torch.Tensor) and (device is None):
            device = k.device
        if isinstance(v, torch.Tensor) and (device is None):
            device = v.device
        if isinstance(state, torch.Tensor) and (device is None):
            device = state.device
        if isinstance(A_log, torch.Tensor) and (device is None):
            device = A_log.device
        if isinstance(a, torch.Tensor) and (device is None):
            device = a.device
        if isinstance(dt_bias, torch.Tensor) and (device is None):
            device = dt_bias.device
        if isinstance(b, torch.Tensor) and (device is None):
            device = b.device

        if device is None:
            device = torch.device("cpu")

        # Ensure all inputs are on the same device
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        state = state.to(device)
        A_log = A_log.to(device)
        a = a.to(device)
        dt_bias = dt_bias.to(device)
        b = b.to(device)

        # Normalize scalar scale if provided as tensor
        if isinstance(scale, torch.Tensor):
            scale = float(scale.item())
        else:
            scale = float(scale)

        # Shapes
        B = q.shape[0]
        # num_q_heads=4 -> expand to num_v_heads=8 via repeat_interleave
        H = 8
        # v has shape [B, 1, 8, V] -> V=128
        V = v.shape[3]  # 128
        K = q.shape[2]  # 128

        # Squeeze T=1 and expand q/k to H=8
        q = q.squeeze(1)  # [B, 4, K]
        k = k.squeeze(1)  # [B, 4, K]
        q_exp = q.repeat_interleave(H // q.shape[1], dim=1)  # [B, 8, K]
        k_exp = k.repeat_interleave(H // k.shape[1], dim=1)  # [B, 8, K]
        v = v.squeeze(1)  # [B, 8, V] already

        # Flatten parameters for Triton
        a_flat = a.contiguous().view(-1)          # [B]
        dt_bias_flat = dt_bias.contiguous().view(-1)  # [H]
        A_log_flat = A_log.contiguous().view(-1)    # [H]
        b_flat = b.contiguous().view(-1)            # [B]

        # Compute g and beta using Triton
        g = torch.empty(H, dtype=torch.float32, device=device)
        beta = torch.empty(H, dtype=torch.float32, device=device)
        grid_g = (H,)
        _compute_g_kernel[grid_g](a_flat, dt_bias_flat, A_log_flat, g, H)
        grid_beta = (H,)
        _compute_beta_kernel[grid_beta](b_flat, beta, H)

        # Output vector [B, H]
        out = torch.empty((B, H), dtype=torch.float32, device=device)
        # New state [B, H, V, K], float32
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Launch update kernel
        grid_update = (B, H)
        _update_all_kernel[grid_update](
            q_exp, k_exp, v, state, g, beta, out, new_state,
            B, H, V, K, scale
        )

        # Return output cast to bfloat16 and new_state
        return (out.to(torch.bfloat16), new_state)


def run(*args):
    return ModelNew()(*args)
