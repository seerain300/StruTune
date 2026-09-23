import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bf16 or float, shape [B, 1, H]
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bf16 or float, shape [B, 1, H]
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    # One program per (b, h)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, 0, h] (second dim is 1, so index with 0)
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + 0 * a_ptr.stride(1) + h * a_ptr.stride(2)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H], contiguous
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log[h]) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b[b, 0, h]) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + 0 * b_ptr.stride(1) + h * b_ptr.stride(2)), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store to [B, 1, H]
    tl.store(g_ptr + b * g_ptr.stride(0) + 0 * g_ptr.stride(1) + h * g_ptr.stride(2), g_val)
    tl.store(beta_ptr + b * beta_ptr.stride(0) + 0 * beta_ptr.stride(1) + h * beta_ptr.stride(2), beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bf16 or float, shape [B, H, K]
    k_ptr,             # *bf16 or float, shape [B, H, K]
    v_ptr,             # *bf16 or float, shape [B, H, V]
    state_ptr,         # *float32, shape [B, H, V, K]
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    output_ptr,        # *float32, shape [B*H]
    new_state_ptr,     # *float32, shape [B, H, V, K]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # heads
    V: tl.constexpr,   # dim V (128)
    K: tl.constexpr,   # dim K (128)
    scale_inv_sqrtK: tl.float32,  # scalar float32
):
    # One program per (b, h)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load q, k, v vectors (length K or V)
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    # Compute base offsets for (b,h)
    q_base = b * K * H + h * K
    k_base = b * K * H + h * K
    v_base = b * V * H + h * V

    for j in range(0, K):
        q_vec[j] = tl.cast(tl.load(q_ptr + q_base + j), tl.float32)
        k_vec[j] = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
    for v_idx in range(0, V):
        v_vec[v_idx] = tl.cast(tl.load(v_ptr + v_base + v_idx), tl.float32)

    # Load g_val and beta_val for this (b, h)
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + 0 * g_ptr.stride(1) + h * g_ptr.stride(2))
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + 0 * beta_ptr.stride(1) + h * beta_ptr.stride(2))

    # old_state: [V, K] from state[b, h, :, :]
    old_state = tl.zeros([V, K], dtype=tl.float32)
    # state layout is [B, H, V, K] contiguous -> index as ((b*H+H)*V + v)*K + k
    state_base = b * (H * V * K) + h * (V * K)
    for v_idx in tl.static_range(0, V):
        row_base = state_base + v_idx * K
        for k_idx in tl.static_range(0, K):
            old_state[v_idx, k_idx] = tl.cast(tl.load(state_ptr + row_base + k_idx), tl.float32)

    # old_v = k @ (g * old_state) -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in tl.static_range(0, K):
        sum_val = 0.0
        for v_idx in tl.static_range(0, V):
            sum_val += old_state[v_idx, j] * g_val
        old_v_vec[j] = sum_val * k_vec[j]

    # new_v = beta * v + (1 - beta) * old_v
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in tl.static_range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update: k @ old_v and k @ new_v (scalars)
    state_remove = 0.0
    state_update = 0.0
    for j in tl.static_range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new: elementwise (g * old_state) - state_remove + state_update
    h_state_new = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in tl.static_range(0, V):
        for k_idx in tl.static_range(0, K):
            h_state_new[v_idx, k_idx] = old_state[v_idx, k_idx] * g_val - state_remove + state_update

    # Compute output scalar: scale * q_h @ h_state_new
    # q_h is q_vec[K], h_state_new is [V, K]
    output_scalar = 0.0
    for v_idx in tl.static_range(0, V):
        dot = 0.0
        for k_idx in tl.static_range(0, K):
            dot += h_state_new[v_idx, k_idx] * q_vec[k_idx]
        output_scalar += dot * v_vec[v_idx]
    output_scalar = output_scalar * scale_inv_sqrtK

    # Store output[b, h] to output_ptr[pid]
    tl.store(output_ptr + pid, output_scalar)

    # Store new_state[b, h, :, :] = h_state_new to [B, H, V, K]
    # new_state_ptr is float32, contiguous layout
    new_state_base = b * (H * V * K) + h * (V * K)
    for v_idx in tl.static_range(0, V):
        row_base = new_state_base + v_idx * K
        for k_idx in tl.static_range(0, K):
            tl.store(new_state_ptr + row_base + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation that computes:
          - g = exp(-exp(A_log) * softplus(a + dt_bias))
          - beta = sigmoid(b)
          - updates state[b,h,:] using the gated delta rule and computes a scalar output per (b,h)
        Returns:
          - output: tensor of shape [B, 1, H], dtype bfloat16
          - new_state: tensor of shape [B, H, V, K], dtype float32
        """
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "q, k, v must be [B, T, H, K] (here T=1)"
        B, T, H_q, K = q.shape
        _, _, H_k, K2 = k.shape
        _, _, H_v, V = v.shape
        # The original code asserts K=128 and V=128
        assert K == 128 and V == 128, "K and V must be 128"

        # Extract B dimension from inputs: num_v_heads = H_v = 8 in the provided inputs
        num_heads = H_v
        device = q.device

        # Prepare tensors for g and beta: [B, 1, H]
        g = torch.empty((B, 1, num_heads), dtype=torch.float32, device=device)
        beta = torch.empty((B, 1, num_heads), dtype=torch.float32, device=device)

        # Launch Triton gate/beta kernel: one program per (b, h)
        grid_g = (B * num_heads,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, num_heads
        )

        # Prepare output vector of length B*H
        output = torch.empty((B * num_heads,), dtype=torch.float32, device=device)

        # Prepare new_state tensor [B, H, V, K] as float32
        new_state = torch.empty((B, num_heads, V, K), dtype=torch.float32, device=device)

        # Compute scale * 1/sqrt(K) as float32 scalar
        if scale is None or scale == 0.0:
            scale_val = 1.0 / math.sqrt(K)
        else:
            scale_val = float(scale)

        # Launch Triton update kernel: one program per (b, h)
        grid_u = (B * num_heads,)
        triton_update_kernel[grid_u](
            q, k, v, state, g, beta, output, new_state,
            B, num_heads, V, K, scale_val
        )

        # Reshape output to (B, 1, H) and cast to bfloat16 to match original behavior
        output_reshaped = output.view(B, num_heads).unsqueeze(1)  # [B, 1, H]
        output_bf16 = output_reshaped.to(torch.bfloat16)

        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
