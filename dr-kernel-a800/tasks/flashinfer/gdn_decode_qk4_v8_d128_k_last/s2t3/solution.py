import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,   # *float32, shape [H]
    a_ptr,       # *bfloat16, shape [B, 1, H] (we access by linear offset)
    dt_bias_ptr, # *float32, shape [H]
    b_ptr,       # *bfloat16, shape [B, 1, H] (we access by linear offset)
    g_ptr,       # *float32, shape [B, 1, H]
    beta_ptr,    # *float32, shape [B, 1, H]
    B: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, 0, h] from bfloat16 tensor and convert to float32
    a_off = b * a_ptr.stride(0) + h * a_ptr.stride(1)
    a_val = tl.cast(tl.load(a_ptr + a_off), tl.float32)

    # dt_bias[h] is float32
    dt_val = tl.load(dt_bias_ptr + h)

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    x = a_val + dt_val
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log[h]) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b[b, 0, h])
    b_off = b * b_ptr.stride(0) + h * b_ptr.stride(1)
    b_val = tl.cast(tl.load(b_ptr + b_off), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H]
    g_off = b * g_ptr.stride(0) + h * g_ptr.stride(1)
    beta_off = b * beta_ptr.stride(0) + h * beta_ptr.stride(1)
    tl.store(g_ptr + g_off, g_val)
    tl.store(beta_ptr + beta_off, beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,       # *bfloat16, shape [B, 1, H, K] (we access by linear offset)
    k_ptr,       # *bfloat16, shape [B, 1, H, K]
    v_ptr,       # *bfloat16, shape [B, 1, H, V]
    state_ptr,   # *float32,  shape [B, H, V, K]
    g_ptr,       # *float32,  shape [B, 1, H]
    beta_ptr,    # *float32,  shape [B, 1, H]
    out_ptr,     # *float32,  shape [B, H]
    newstate_ptr,# *float32,  shape [B, H, V, K]
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Prepare vectors
    q_off = b * q_ptr.stride(0) + h * q_ptr.stride(2)  # + dim1 stride=1 for T=1
    k_off = b * k_ptr.stride(0) + h * k_ptr.stride(2)
    v_off = b * v_ptr.stride(0) + h * v_ptr.stride(2)

    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    # Load q, k, v (contiguous along last dim)
    for j in range(0, K):
        q_vec[j] = tl.cast(tl.load(q_ptr + q_off + j), tl.float32)
        k_vec[j] = tl.cast(tl.load(k_ptr + k_off + j), tl.float32)
    for v_idx in range(0, V):
        v_vec[v_idx] = tl.cast(tl.load(v_ptr + v_off + v_idx), tl.float32)

    # Load state_old: [V, K]
    state_off = b * state_ptr.stride(0) + h * state_ptr.stride(1)  # + dim1 stride for H
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + state_off + v_idx * state_ptr.stride(3)  # stride(3) is K
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(row_base + k_idx), tl.float32)

    # Load g and beta scalars
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1))
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1))

    # Compute old_v = k @ (g * old_state) -> (K,)
    g_old = g_val * h_state  # [V, K]
    old_v = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = tl.zeros([], dtype=tl.float32)
        for v_idx in range(0, V):
            sum_val += g_old[v_idx, j] * k_vec[j]
        old_v[j] = sum_val

    # new_v = beta * v + (1 - beta) * old_v
    new_v = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v[v_idx]

    # Compute state_remove and state_update as scalars: k @ old_v and k @ new_v
    state_remove = tl.zeros([], dtype=tl.float32)
    state_update = tl.zeros([], dtype=tl.float32)
    for j in range(0, K):
        state_remove += old_v[j] * k_vec[j]
        state_update += new_v[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = g_old - (state_remove - state_update)  # broadcast scalar

    # Compute output scalar: output = scale * q @ h_state_new
    output_scalar = tl.zeros([], dtype=tl.float32)
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state_new[:, j]  # [V]
        sum_j = tl.zeros([], dtype=tl.float32)
        for v_idx in range(0, V):
            sum_j += row_j[v_idx]
        output_scalar += qj * sum_j

    output_scalar = output_scalar * SCALE

    # Store output[b, h]
    tl.store(out_ptr + b * out_ptr.stride(0) + h * out_ptr.stride(1), output_scalar)

    # Store new_state[b, h] as [V, K]
    ns_off = b * newstate_ptr.stride(0) + h * newstate_ptr.stride(1)
    for v_idx in range(0, V):
        row_base_ns = newstate_ptr + ns_off + v_idx * newstate_ptr.stride(3)
        for k_idx in range(0, K):
            tl.store(row_base_ns + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version:
        - Computes g and beta using Triton.
        - Performs per-(b,h) state updates and outputs using a Triton kernel.
        - Returns output as bfloat16 unsqueezed to (B, 1, H) and new_state as float32.
        """
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        num_heads = num_v_heads

        device = q.device

        # Ensure inputs are contiguous and on device
        a = a.to(device)
        dt_bias = dt_bias.to(device)
        b = b.to(device)
        A_log = A_log.to(device)
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        state = state.to(device)

        # Compute g and beta using Triton
        g = torch.empty((B, 1, num_heads), dtype=torch.float32, device=device)
        beta = torch.empty((B, 1, num_heads), dtype=torch.float32, device=device)

        grid_g = (B * num_heads,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, num_heads
        )

        # Prepare output and new_state
        output = torch.empty((B, num_heads), dtype=torch.float32, device=device)
        new_state = torch.empty_like(state, dtype=torch.float32, device=device)

        # Handle scale
        if scale is None or scale == 0.0:
            scale_val = 1.0 / math.sqrt(K)
        else:
            scale_val = float(scale)

        # Launch Triton update kernel: one program per (b,h)
        grid_u = (B * num_heads,)
        triton_update_kernel[grid_u](
            q, k, v, state, g, beta, output, new_state,
            B, num_heads, V, K, scale_val
        )

        # Return outputs as expected by the original: output as bfloat16 unsqueezed, new_state float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
