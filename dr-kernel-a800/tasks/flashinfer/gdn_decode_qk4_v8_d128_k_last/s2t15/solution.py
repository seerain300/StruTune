import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size (compile-time for grid)
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a and dt_bias to compute x = a + dt_bias
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)  # A_log is [H]
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + h * b_ptr.stride(1)), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H] layout
    tl.store(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1), g_val)
    tl.store(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1), beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16, [B, H, K]
    k_ptr,             # *bfloat16, [B, H, K]
    v_ptr,             # *bfloat16, [B, H, V]
    state_ptr,         # *float32, [B, H, V, K]
    g_ptr,             # *float32, [B, 1, H]
    beta_ptr,          # *float32, [B, 1, H]
    output_ptr,        # *float32, [B, H]
    new_state_ptr,     # *float32, [B, H, V, K]
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    scale: tl.constexpr,  # scalar 1/sqrt(K) passed as constexpr
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load params
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1))  # scalar
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1))  # scalar

    # Prepare vectors q_h, k_h, v_h (length K and V)
    q_base = b * H * K + h * K
    k_base = b * H * K + h * K
    v_base = b * H * V + h * V

    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_ptr + q_base + j), tl.float32)
        k_elem = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
        q_vec[j] = q_elem
        k_vec[j] = k_elem

    for v_idx in range(0, V):
        v_vec[v_idx] = tl.cast(tl.load(v_ptr + v_base + v_idx), tl.float32)

    # Load state_old: [V, K]
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + (b * H * V + h * V + v_idx) * K
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(row_base + k_idx), tl.float32)

    # Compute old_v = k @ (g * state_old)  -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = 0.0
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val
        old_v_vec[j] = sum_val

    # Compute new_v = beta * v + (1-beta) * old_v
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update as scalars: k @ old_v and k @ new_v
    state_remove = 0.0
    state_update = 0.0
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = h_state * g_val - state_remove + state_update

    # Compute output scalar: output = scale * q_h @ h_state_new
    output_scalar = 0.0
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state_new[:, j]  # [V]
        sum_j = 0.0
        for v_idx in range(0, V):
            sum_j += row_j[v_idx]
        output_scalar += qj * sum_j

    # Store output and updated state (new_state = h_state_new)
    tl.store(output_ptr + b * output_ptr.stride(0) + h * output_ptr.stride(1), output_scalar)

    # Write new state [V, K] block
    new_base = new_state_ptr + (b * H * V + h * V) * K
    for v_idx in range(0, V):
        new_row_base = new_base + v_idx * K
        for k_idx in range(0, K):
            tl.store(new_row_base + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version:
        - Compute g and beta in Triton
        - Compute per-(b,h) updates and output in Triton
        Returns:
          - output as bfloat16, unsqueezed to (B, 1, H)
          - new_state as float32 [B, H, V, K]
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All tensors must be on CUDA for Triton."
        B, H_q, K = q.shape
        _, H_k, _ = k.shape
        _, H_v, V = v.shape
        assert H_q == H_k == H_v, "num heads (H) must match across q, k, v"
        H = H_q
        num_heads = H

        # Allocate outputs for g and beta (float32)
        g = torch.empty((B, 1, H), dtype=torch.float32, device=q.device)
        beta = torch.empty((B, 1, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel to compute g and beta
        grid_g = (B * H,)
        triton_gate_beta_kernel[grid_g](A_log, a, dt_bias, b, g, beta, B, H)

        # Compute scale = 1/sqrt(K) inside Triton (pass as constexpr)
        # Note: we don't use scale argument if it's None; default to 1/sqrt(K)
        K_val = K
        scale_val = 1.0 / math.sqrt(K_val)

        # Output and new state
        output = torch.empty((B, H), dtype=torch.float32, device=q.device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=q.device)

        # Launch Triton update kernel
        grid_u = (B * H,)
        triton_update_kernel[grid_u](q, k, v, state, g, beta, output, new_state, B, H, V, K, scale_val)

        # Return outputs as expected: output bfloat16 unsqueezed, new_state float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
