import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h via 1D stride)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H]
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # num heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load parameters for this (b, h)
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + h * b_ptr.stride(1)), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store to [B,1,H] linear layout
    tl.store(g_ptr + pid, g_val)
    tl.store(beta_ptr + pid, beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16, shape [B, H, K]
    k_ptr,             # *bfloat16, shape [B, H, K]
    v_ptr,             # *bfloat16, shape [B, H, V]
    state_ptr,         # *float32, shape [B, H, V, K] (k-last)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    output_ptr,        # *float32, shape [B, H]
    new_state_ptr,     # *float32, shape [B, H, V, K]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # num heads
    V: tl.constexpr,   # num heads == num_v_heads, V=128 in the task
    K: tl.constexpr,   # K=128 in the task
    scale,             # float32 scalar
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load parameters
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1))  # g is [B,1,H] float32
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1))  # beta is [B,1,H] float32

    # Load vectors q[b,h,:], k[b,h,:], v[b,h,:] (cast to float32 for math)
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_ptr + b * q_ptr.stride(0) + h * q_ptr.stride(1) + j), tl.float32)
        k_elem = tl.cast(tl.load(k_ptr + b * k_ptr.stride(0) + h * k_ptr.stride(1) + j), tl.float32)
        q_vec[j] = q_elem
        k_vec[j] = k_elem

    for v_idx in range(0, V):
        v_elem = tl.cast(tl.load(v_ptr + b * v_ptr.stride(0) + h * v_ptr.stride(1) + v_idx), tl.float32)
        v_vec[v_idx] = v_elem

    # Load old state: [V, K] float32
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + b * state_ptr.stride(0) + h * state_ptr.stride(1) + v_idx * state_ptr.stride(2)
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

    # Compute scalar contributions from k
    state_remove = 0.0
    state_update = 0.0
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = h_state * g_val - state_remove + state_update

    # Compute output scalar: output = scale * q_h @ h_state_new
    inv_sqrtK = scale  # scale is passed as 1/sqrt(K)
    output_scalar = 0.0
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state_new[:, j]  # [V]
        # sum over V
        for v_idx in range(0, V):
            output_scalar += qj * row_j[v_idx]

    output_scalar = output_scalar * inv_sqrtK

    # Store output
    tl.store(output_ptr + b * output_ptr.stride(0) + h * output_ptr.stride(1), output_scalar)

    # Store updated state: [V, K] -> new_state[b,h, :, :]
    out_row_base = new_state_ptr + b * new_state_ptr.stride(0) + h * new_state_ptr.stride(1)
    for v_idx in range(0, V):
        out_row = out_row_base + v_idx * new_state_ptr.stride(2)
        for k_idx in range(0, K):
            tl.store(out_row + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the provided run function.
        - q: [B, 1, Hq, K], Hq=4 in the given test
        - k: [B, 1, Hk, K], Hk=4 in the given test
        - v: [B, 1, Hv, V], Hv=8 in the given test
        - state: [B, Hv, V, K], layout [B, H, V, K]
        - A_log: [Hv], float32
        - a: [B, 1, Hv], bfloat16
        - dt_bias: [Hv], float32
        - b: [B, 1, Hv], bfloat16
        - scale: float or None
        Returns:
        - output: [B, 1, Hv] bfloat16
        - new_state: [B, Hv, V, K] float32
        """
        device = q.device
        B = q.shape[0]
        Hq = q.shape[2]
        K = q.shape[3]
        Hk = k.shape[2]
        Hv = v.shape[2]
        V = v.shape[3]
        assert Hq == 4 and Hk == 4 and Hv == 8, "The provided test expects Hq=4, Hk=4, Hv=8."
        assert K == 128 and V == 128, "The provided test expects K=128 and V=128."

        # Ensure inputs are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()

        # Compute g and beta using Triton: shapes [B,1,Hv]
        g = torch.empty((B, 1, Hv), dtype=torch.float32, device=device)
        beta = torch.empty((B, 1, Hv), dtype=torch.float32, device=device)

        grid_g = (B * Hv,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, Hv
        )

        # Prepare output and new_state
        output = torch.empty((B, Hv), dtype=torch.float32, device=device)
        new_state = torch.empty_like(state, dtype=torch.float32, device=device)

        # Determine scale: 1/sqrt(K)
        if scale is None or scale == 0.0:
            scale_val = 1.0 / math.sqrt(K)
        else:
            scale_val = float(scale)

        # Launch Triton update kernel: one program per (b,h)
        grid_u = (B * Hv,)
        triton_update_kernel[grid_u](
            q, k, v, state, g, beta, output, new_state,
            B, Hv, V, K, scale_val
        )

        # Return outputs as expected by the original: output as bfloat16 unsqueezed, new_state float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
