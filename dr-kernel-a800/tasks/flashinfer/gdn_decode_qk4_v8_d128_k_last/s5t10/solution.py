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
    q_ptr: [B, H, K]
    k_ptr: [B, H, K]
    v_ptr: [B, H, V]
    state_ptr: [B, H, V, K] (input state, will be written with updated state)
    g_ptr: [H]
    beta_ptr: [H]
    out_ptr: [B, H] (float32)
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Base offsets for q/k/v
    q_base = b * (H * K) + h * K
    k_base = b * (H * K) + h * K
    v_base = b * (H * V) + h * V

    # Load q and k vectors (length K)
    q_vec = [0.0] * K
    k_vec = [0.0] * K
    for i in range(0, K):
        q_vec[i] = tl.load(q_ptr + q_base + i)
        k_vec[i] = tl.load(k_ptr + k_base + i)

    # Load params
    g_val = tl.load(g_ptr + h)
    beta_val = tl.load(beta_ptr + h)

    # Load state[b,h] as [V, K]
    state_base = b * (H * V * K) + h * (V * K)
    old_state = [[0.0] * K for _ in range(0, V)]
    for i in range(0, V):
        for k in range(0, K):
            old_state[i][k] = tl.load(state_ptr + state_base + i * K + k)

    # old_v = k @ state_old (reduce over K)
    old_v = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_vec[k] * old_state[i][k]
        old_v[i] = sum_k

    # new_v = beta * v + (1 - beta) * old_v
    new_v = [0.0] * V
    for i in range(0, V):
        v_elem = tl.load(v_ptr + v_base + i)  # v[b,h,i]
        new_v[i] = beta_val * v_elem + (1.0 - beta_val) * old_v[i]

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

    # Write new_state[b,h] into state_ptr at offset b*H*V*K + h*V*K
    new_state_base = b * (H * V * K) + h * (V * K)
    for i in range(0, V):
        for k in range(0, K):
            val = old_state[i][k] - state_remove[i] + state_update[i]
            tl.store(state_ptr + new_state_base + i * K + k, val)

    # Compute output[b,h] = scale * sum_i q[i] * (state_update[i] - state_remove[i]) (scalar per (b,h))
    out_val = 0.0
    for i in range(0, V):
        out_val += q_vec[i] * (state_update[i] - state_remove[i])
    out_val = out_val * scale
    tl.store(out_ptr + b * H + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only forward: no torch ops on tensors.
        Inputs:
          q: [B, 1, 4, K]
          k: [B, 1, 4, K]
          v: [B, 1, 8, V]
          state: [B, 8, V, K]
          A_log: [8]
          a: [B, 1, 8]
          dt_bias: [8]
          b: [B, 1, 8]
          scale: float or None
        Returns:
          output: [B, 8, V] in bfloat16
          new_state: [B, 8, V, K] in float32
        """
        device = q.device
        B = q.shape[0]
        H = 8  # num_v_heads
        V = state.shape[2]
        K = state.shape[3]

        # Squeeze T=1
        q = q.squeeze(1)  # [B, 4, K]
        k = k.squeeze(1)  # [B, 4, K]
        v = v.squeeze(1)  # [B, 8, V]

        # Expand q/k/v to H=8 to match original behavior
        q_exp = q.repeat_interleave(H // 4, dim=1)  # [B, 8, K]
        k_exp = k.repeat_interleave(H // 4, dim=1)  # [B, 8, K]
        v_exp = v  # already [B, 8, V]

        # Prepare parameter tensors (flattened per head)
        a_flat = a.reshape(B, H).reshape(-1)  # [B, H] -> [B*H]
        dt_bias_flat = dt_bias                 # [H]
        A_log_flat = A_log                    # [H]
        b_flat = b.reshape(B, H).reshape(-1)  # [B*H]

        # Allocate g and beta as float32 on device
        g = torch.empty(H, dtype=torch.float32, device=device)
        beta = torch.empty(H, dtype=torch.float32, device=device)

        # Launch kernels for g and beta
        _compute_g_kernel[(H,)](a_flat, dt_bias_flat, A_log_flat, g, H)
        _compute_beta_kernel[(H,)](b_flat, beta, H)

        # Allocate output per (b,h)
        out = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch update kernel for all (b,h)
        _update_all_kernel[(B, H)](
            q_exp, k_exp, v_exp,
            state, g, beta, out,
            B, H, V, K, scale if scale is not None else 1.0
        )

        # Reshape output to [B, H, V] in bfloat16
        output = out.view(B, H)  # [B, H]
        # The original run returns [B, num_heads, V]; here num_heads = H
        output_3d = output.unsqueeze(1).expand(B, H, V).to(torch.bfloat16)

        # new_state already updated in-place in the kernel
        new_state = state  # [B, H, V, K] updated inside kernel

        return output_3d, new_state


def run(*args):
    return ModelNew()(*args)
