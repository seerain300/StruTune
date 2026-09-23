import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_per_h_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a_per_h[h] + dt_bias[h])) for h in [0..H-1]
    a_per_h_ptr: [H] (float32) - corresponds to a[0,0,h]
    dt_bias_ptr: [H] (float32)
    A_log_ptr: [H] (float32)
    g_ptr: [H] (float32)
    """
    h = tl.program_id(0)
    a_val = tl.load(a_per_h_ptr + h)       # float32
    dt_val = tl.load(dt_bias_ptr + h)      # float32
    A_val = tl.load(A_log_ptr + h)         # float32
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * sp)
    tl.store(g_ptr + h, g_val)


@triton.jit
def _compute_beta_kernel(b_per_h_ptr, beta_ptr, H):
    """
    Compute beta[h] = sigmoid(b_per_h[h]) for h in [0..H-1]
    b_per_h_ptr: [H] (float32) - corresponds to b[0,0,h]
    beta_ptr: [H] (float32)
    """
    h = tl.program_id(0)
    b_val = tl.load(b_per_h_ptr + h)       # float32
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + h, beta_val)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, out_ptr,
    B, H, V, K, scale
):
    """
    For each (b, h), update new_state[b,h] and compute output[b,h].
    Inputs/outputs are linearized as follows:
    - q_ptr: [B, H, K] contiguous
    - k_ptr: [B, H, K] contiguous
    - v_ptr: [B, H, V] contiguous
    - state_ptr: [B, H, V, K] contiguous (we read/write in-place updates)
    - g_ptr: [H] (float32)
    - beta_ptr: [H] (float32)
    - out_ptr: [B, H] (float32)
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Compute base indices for this (b,h)
    # q_ptr element index: ((b * H) + h) * K + i
    # k_ptr element index: ((b * H) + h) * K + i
    # v_ptr element index: ((b * H) + h) * V + i
    base_qk = (b * H + h) * K
    base_v = (b * H + h) * V

    # Load q_h, k_h, v_h vectors
    q_vec = [0.0] * K
    k_vec = [0.0] * K
    for i in range(0, K):
        q_vec[i] = tl.load(q_ptr + base_qk + i)
        k_vec[i] = tl.load(k_ptr + base_qk + i)

    # Load params g[h], beta[h]
    g_val = tl.load(g_ptr + h)
    beta_val = tl.load(beta_ptr + h)

    # Compute state[b,h] as [V, K] and update
    state_base = (b * H + h) * (V * K)
    old_state = [[0.0] * K for _ in range(0, V)]  # will hold original state[b,h]
    # Read original state[b,h] from state_ptr and store in old_state
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

    # new_v = beta * v + (1 - beta) * old_v (vector of length V)
    new_v = [0.0] * V
    for i in range(0, V):
        v_elem = tl.load(v_ptr + base_v + i)  # v[b,h,i]
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

    # Write updated new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None]
    new_state_base = (b * H + h) * (V * K)
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
    tl.store(out_ptr + (b * H + h), out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, K], k: [B, 1, 4, K], v: [B, 1, 8, V], state: [B, 8, V, K]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: scalar
        Returns: (output [B, 8] as bfloat16, updated state [B, 8, V, K])
        """
        device = q.device
        B = q.shape[0]
        H = v.shape[1]  # 8
        V = v.shape[2]  # 128
        K = q.shape[3]  # 128

        # Squeeze T=1 as original does
        q = q.squeeze(1)  # [B, 4, K]
        k = k.squeeze(1)  # [B, 4, K]
        v = v.squeeze(1)  # [B, 8, V]

        # Cast to float32 for Triton kernels (ensure dtype)
        q_fp32 = q.to(torch.float32)  # [B, 4, K]
        k_fp32 = k.to(torch.float32)  # [B, 4, K]
        v_fp32 = v.to(torch.float32)  # [B, 8, V]
        state_fp32 = state.to(torch.float32)  # [B, 8, V, K]
        A_log = A_log.to(torch.float32)       # [8]
        a_fp32 = a.to(torch.float32).squeeze(1)  # [B, 8] -> [B,8]
        dt_bias = dt_bias.to(torch.float32)    # [8]
        b_fp32 = b.to(torch.float32).squeeze(1) # [B, 8] -> [B,8]

        # Allocate outputs
        g = torch.empty(H, dtype=torch.float32, device=device)
        beta = torch.empty(H, dtype=torch.float32, device=device)
        out = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch kernels
        # 1) compute g[h]
        _compute_g_kernel[(H,)](a_fp32, dt_bias, A_log, g, H)
        # 2) compute beta[h]
        _compute_beta_kernel[(H,)](b_fp32, beta, H)
        # 3) update all (b,h) and compute output
        _update_all_kernel[(B, H)](
            q_fp32, k_fp32, v_fp32, state_fp32, g, beta, out,
            B, H, V, K, float(scale)
        )

        # Reshape output and cast to bfloat16 to match original behavior
        output = out.view(B, H).to(torch.bfloat16)  # [B, 8] in bfloat16
        # Return output and updated state (float32; original returns float32 state)
        return output, state_fp32


def run(*args):
    return ModelNew()(*args)
