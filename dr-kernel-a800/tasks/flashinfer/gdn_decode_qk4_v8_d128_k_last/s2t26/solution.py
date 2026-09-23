import torch
import triton
import triton.language as tl


@triton.jit
def triton_softplus(sp_ptr, a_ptr, dt_bias_ptr, B: tl.constexpr, H: tl.constexpr):
    # One program per (b,h): compute softplus(a[b, h] + dt_bias[h])
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H
    a_val = tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1))  # bfloat16 or float32
    dt_val = tl.load(dt_bias_ptr + h)  # float32
    x = tl.cast(a_val, tl.float32) + dt_val  # float32
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))  # softplus in float32
    tl.store(sp_ptr + b * sp_ptr.stride(0) + h * sp_ptr.stride(1), sp)  # sp is float32


@triton.jit
def triton_gate_beta_kernel(g_ptr, beta_ptr, A_log_ptr, sp_ptr, b_ptr,
                             B: tl.constexpr, H: tl.constexpr):
    # One program per (b,h): compute g and beta using precomputed softplus
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H
    A_log_val = tl.load(A_log_ptr + h)  # float32
    sp_val = tl.load(sp_ptr + b * sp_ptr.stride(0) + h * sp_ptr.stride(1))  # float32
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + h * b_ptr.stride(1)), tl.float32)  # b is bfloat16
    g_val = tl.exp(-tl.exp(A_log_val) * sp_val)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1), g_val)
    tl.store(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1), beta_val)


@triton.jit
def triton_update_kernel(
    output_ptr, new_state_ptr, q_ptr, k_ptr, v_ptr, state_ptr,
    g_ptr, beta_ptr, B, H, V, K, scale,
    Q_strideB, Q_strideH, Q_strideK,
    K_strideB, K_strideH, K_strideK,
    V_strideB, V_strideH, V_strideV,
    State_strideB, State_strideH, State_strideV, State_strideK,
    Output_strideB, Output_strideH
):
    # One program per (b,h)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load scalars
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1))
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1))

    # Prepare vectors/arrays
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    # Base offsets for (b,h)
    q_base = b * Q_strideB + h * Q_strideH
    k_base = b * K_strideB + h * K_strideH
    v_base = b * V_strideB + h * V_strideH

    # Load q[b,h,:], k[b,h,:], v[b,h,:]
    for j in range(0, K):
        q_vec[j] = tl.cast(tl.load(q_ptr + q_base + j * Q_strideK), tl.float32)
        k_vec[j] = tl.cast(tl.load(k_ptr + k_base + j * K_strideK), tl.float32)
    for v_idx in range(0, V):
        v_vec[v_idx] = tl.cast(tl.load(v_ptr + v_base + v_idx * V_strideV), tl.float32)

    # Load old state[b,h,:,:] as [V,K]
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = b * State_strideB + h * State_strideH + v_idx * State_strideV
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(state_ptr + row_base + k_idx * State_strideK), tl.float32)

    # Compute old_v = k @ (g * state_old) -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = 0.0
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val
        old_v_vec[j] = sum_val * k_vec[j]  # k_vec[j] is scalar, broadcast

    # Compute new_v = beta * v + (1 - beta) * old_v -> (V,)
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update: k @ old_v and k @ new_v
    state_remove = 0.0
    state_update = 0.0
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = h_state * g_val - state_remove + state_update

    # Compute output scalar: output = scale * q_h @ h_state_new
    out_scalar = 0.0
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state_new[:, j]  # [V]
        for v_idx in range(0, V):
            out_scalar += qj * row_j[v_idx]
    out_scalar = out_scalar * scale  # use runtime scale

    # Store output [B,H] float32
    tl.store(output_ptr + b * Output_strideB + h * Output_strideH, out_scalar)

    # Store new_state[b,h,:,:] = h_state_new
    for v_idx in range(0, V):
        row_base_out = b * State_strideB + h * State_strideH + v_idx * State_strideV
        for k_idx in range(0, K):
            tl.store(new_state_ptr + row_base_out + k_idx * State_strideK, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation:
        - Computes softplus, g, beta in Triton.
        - Performs per-(b,h) state updates and output computation in Triton.
        Returns:
          output: [B, 1, H] in bfloat16 (to match original .unsqueeze(1).to(bfloat16))
          new_state: [B, H, V, K] float32 (same layout as input state)
        """
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        num_heads = num_v_heads
        device = q.device

        # Ensure inputs are contiguous and float32 for Triton math
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)
        state = state.contiguous().to(torch.float32)
        A_log = A_log.contiguous().to(torch.float32)
        a = a.contiguous().to(torch.float32)
        dt_bias = dt_bias.contiguous().to(torch.float32)
        b = b.contiguous().to(torch.float32)

        # Allocate intermediates in Triton
        sp = torch.empty((B, num_heads), dtype=torch.float32, device=device)
        g = torch.empty((B, num_heads), dtype=torch.float32, device=device)
        beta = torch.empty((B, num_heads), dtype=torch.float32, device=device)

        # 1) Compute softplus(a + dt_bias)
        grid_sp = (B * num_heads,)
        triton_softplus[grid_sp](sp, a, dt_bias, B, num_heads)

        # 2) Compute g and beta from A_log, softplus, and b
        grid_gb = (B * num_heads,)
        triton_gate_beta_kernel[grid_gb](g, beta, A_log, sp, b, B, num_heads)

        # 3) Update state and compute output using Triton
        new_state = torch.empty_like(state, dtype=torch.float32, device=device)
        output = torch.empty((B, num_heads), dtype=torch.float32, device=device)

        # Use the runtime scale exactly (float scalar)
        scale_val = float(scale) if scale is not None else 1.0

        grid_u = (B * num_heads,)
        triton_update_kernel[grid_u](
            output, new_state, q, k, v, state,
            g, beta, B, num_heads, V, K, scale_val,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            output.stride(0), output.stride(1),
        )

        # Return outputs as expected: output in bfloat16 unsqueezed, new_state float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
