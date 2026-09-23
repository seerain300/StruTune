import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16, shape [B, 1, H]
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H]
    g_ptr,             # *float32,  shape [B, 1, H]
    beta_ptr,          # *float32,  shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, 1, h] (bfloat16) and cast to float32
    a_val = tl.cast(tl.load(a_ptr + b * 1 * H + h), tl.float32)
    # dt_bias[h] is float32
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log[h]) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b[b, 1, h]) as bfloat16 -> float32
    b_val = tl.cast(tl.load(b_ptr + b * 1 * H + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results into [B, 1, H] layout (contiguous: stride along H)
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16, shape [B, H, K]
    k_ptr,             # *bfloat16, shape [B, H, K]
    v_ptr,             # *bfloat16, shape [B, H, V]
    state_ptr,         # *float32,  shape [B, H, V, K]
    g_ptr,             # *float32,  shape [B, 1, H]
    beta_ptr,          # *float32,  shape [B, 1, H]
    output_ptr,        # *float32,  shape [B, H]
    new_state_ptr,     # *float32,  shape [B, H, V, K]
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    scale_over_sqrtK: tl.constexpr,  # host-computed 1/sqrt(K) as float
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Base offsets for (b,h)
    q_base = b * H * K + h * K
    k_base = b * H * K + h * K
    v_base = b * H * V + h * V

    # Load vectors as float32
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        qj = tl.cast(tl.load(q_ptr + q_base + j), tl.float32)
        kj = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
        q_vec[j] = qj
        k_vec[j] = kj

    for v_idx in range(0, V):
        vv = tl.cast(tl.load(v_ptr + v_base + v_idx), tl.float32)
        v_vec[v_idx] = vv

    # Load old state: [V, K]
    state_old = tl.zeros([V, K], dtype=tl.float32)
    base_state = state_ptr + b * H * V * K + h * V * K
    for v_idx in range(0, V):
        row_base = base_state + v_idx * K
        for k_idx in range(0, K):
            state_old[v_idx, k_idx] = tl.cast(tl.load(state_ptr + row_base + k_idx), tl.float32)

    # Load gate and beta scalars
    g_val = tl.load(g_ptr + b * H + h)  # [B,1,H] contiguous
    beta_val = tl.load(beta_ptr + b * H + h)

    # Compute old_v = k @ (g * state_old) -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = tl.zeros([], dtype=tl.float32)
        for v_idx in range(0, V):
            sum_val += state_old[v_idx, j] * g_val
        old_v_vec[j] = tl.sum(sum_val * k_vec[j], axis=0)  # k_vec[j] is scalar

    # Compute new_v = beta * v + (1-beta) * old_v
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update: scalar
    state_remove = tl.zeros([], dtype=tl.float32)
    state_update = tl.zeros([], dtype=tl.float32)
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * state_old) - state_remove + state_update
    h_state_new = state_old * g_val - state_remove + state_update

    # Compute output scalar: output = scale * (q_h @ h_state_new) = scale * sum_j q[j] * sum_i h_state_new[i,j]
    q_dot_row = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_row = tl.zeros([], dtype=tl.float32)
        for v_idx in range(0, V):
            sum_row += h_state_new[v_idx, j]
        q_dot_row[j] = q_vec[j] * sum_row
    output_scalar = tl.sum(q_dot_row, axis=0) * scale_over_sqrtK
    tl.store(output_ptr + b * H + h, output_scalar)

    # Store new_state[b,h,:,:] back
    base_new = new_state_ptr + b * H * V * K + h * V * K
    for v_idx in range(0, V):
        row_base = base_new + v_idx * K
        for k_idx in range(0, K):
            tl.store(new_state_ptr + b * H * V * K + h * V * K + v_idx * K + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version:
        - Compute g and beta via Triton kernel.
        - Update state and compute output per (b, h) via Triton kernel.
        Returns:
        - output as bfloat16 with unsqueeze(1)
        - new_state as float32
        """
        # Shapes and device
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        device = q.device

        assert T == 1, "T must be 1"
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8
        assert K == 128 and V == 128

        # Ensure contiguous inputs
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        state_c = state.contiguous()

        # Allocate outputs
        g = torch.empty((B, 1, 8), dtype=torch.float32, device=device)
        beta = torch.empty((B, 1, 8), dtype=torch.float32, device=device)
        output = torch.empty((B, 8), dtype=torch.float32, device=device)
        new_state = torch.empty_like(state_c, dtype=torch.float32, device=device)

        # Launch Triton kernel 1: compute g and beta
        grid_g = (B * 8,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, 8
        )

        # Precompute scale_over_sqrtK on host to avoid any torch ops inside forward
        if scale is None or scale == 0.0:
            scale_over_sqrtK = 1.0 / math.sqrt(K)
        else:
            scale_over_sqrtK = float(scale) * (1.0 / math.sqrt(K))

        # Launch Triton kernel 2: update state and compute output
        grid_u = (B * 8,)
        triton_update_kernel[grid_u](
            q_c, k_c, v_c, state_c, g, beta, output, new_state,
            B, 8, 128, 128, scale_over_sqrtK
        )

        # Return outputs as expected by the original: output bfloat16 (unsqueezed), new_state float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
