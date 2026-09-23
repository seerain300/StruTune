import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a[0,0,h] + dt_bias[h])) for h in [0..H-1]
    a_ptr: [B, 1, H] flattened to [H] (fp32)
    dt_bias_ptr: [H] (fp32)
    A_log_ptr: [H] (fp32)
    g_ptr: [H] (fp32)
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
    b_ptr: [B, 1, H] flattened to [H] (fp32)
    beta_ptr: [H] (fp32)
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
    For each (b, h), update new_state[b,h] (in-place in state_ptr) and compute output[b,h].
    q_ptr: [B, H, K] (fp32)
    k_ptr: [B, H, K] (fp32)
    v_ptr: [B, H, V] (fp32)
    state_ptr: [B, H, V, K] (fp32, input and updated in-place)
    g_ptr: [H] (fp32)
    beta_ptr: [H] (fp32)
    out_ptr: [B, H] (fp32)
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Base offsets for q, k, v
    q_base = b * (H * K) + h * K
    k_base = b * (H * K) + h * K
    v_base = b * (H * V) + h * V

    # Load vectors
    q_vec = [0.0] * K
    k_vec = [0.0] * K
    for i in range(0, K):
        q_vec[i] = tl.load(q_ptr + q_base + i)
        k_vec[i] = tl.load(k_ptr + k_base + i)

    # Load params
    g_val = tl.load(g_ptr + h)
    beta_val = tl.load(beta_ptr + h)

    # Load state[b,h] as [V, K] (fp32)
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

    # Write new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None]
    new_state_base = b * (H * V * K) + h * (V * K)
    for i in range(0, V):
        for k in range(0, K):
            val = old_state[i][k] - state_remove[i] + state_update[i]
            tl.store(state_ptr + new_state_base + i * K + k, val)

    # Compute output[b,h] = scale * (q @ new_state[b,h]) (reduce over V)
    out_val = 0.0
    for i in range(0, V):
        row_sum = 0.0
        for k in range(0, K):
            row_sum += q_vec[k] * (old_state[i][k] - state_remove[i] + state_update[i])
        out_val += row_sum
    out_val *= scale
    tl.store(out_ptr + (b * H) + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, K]
        k: [B, 1, 4, K]
        v: [B, 1, 8, V]
        state: [B, 8, V, K]
        A_log: [8] (fp32)
        a: [B, 1, 8] (bf16 or fp32, will be cast)
        dt_bias: [8] (fp32)
        b: [B, 1, 8] (bf16 or fp32, will be cast)
        scale: scalar or tensor
        Returns (output [B, 8] as bfloat16, updated state [B, 8, V, K])
        """
        device = q.device

        # Squeeze T=1 as original does
        q = q.squeeze(1)  # [B, 4, K]
        k = k.squeeze(1)  # [B, 4, K]
        v = v.squeeze(1)  # [B, 8, V]

        B = q.shape[0]
        H = v.shape[1]  # num_v_heads = 8
        V = v.shape[2]  # 128
        K = q.shape[2]  # 128

        # Cast to fp32 for Triton kernels
        q_fp32 = q.to(torch.float32)  # [B, 4, K]
        k_fp32 = k.to(torch.float32)  # [


def run(*args):
    return ModelNew()(*args)
