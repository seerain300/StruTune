import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(
    A_log_ptr,       # [H_v] float32
    a_ptr,           # [B, H_v] float32
    dt_bias_ptr,     # [H_v] float32
    b_ptr,           # [B, H_v] float32
    g_ptr,           # [B, H_v] float32
    beta_ptr,        # [B, H_v] float32
    B: tl.int32,
    H_v: tl.int32,
):
    pid = tl.program_id(0)
    hv_idx = pid % H_v
    b_idx = pid // H_v

    a_val = tl.load(a_ptr + b_idx * H_v + hv_idx)
    dt_val = tl.load(dt_bias_ptr + hv_idx)
    b_val = tl.load(b_ptr + b_idx * H_v + hv_idx)
    A_log_val = tl.load(A_log_ptr + hv_idx)

    # softplus(x) = log1p(exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_log_val) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_ptr + b_idx * H_v + hv_idx, g_val)
    tl.store(beta_ptr + b_idx * H_v + hv_idx, beta_val)


@triton.jit
def _state_update_kernel(
    k_ptr,           # [B, H_v, D] float32
    v_ptr,           # [B, H_v, D] float32
    state_ptr,       # [H_v, D, D] float32
    g_ptr,           # [B, H_v] float32
    beta_ptr,        # [B, H_v] float32
    B: tl.int32,
    H_v: tl.int32,
    D: tl.int32,
):
    pid = tl.program_id(0)
    hv_idx = pid % H_v
    b_idx = pid // H_v  # iterate over tokens t in host loop

    # Load vectors
    i = 0
    k_vec = tl.zeros((D,), dtype=tl.float32)
    v_vec = tl.zeros((D,), dtype=tl.float32)
    while i < D:
        k_vec[i] = tl.load(k_ptr + b_idx * H_v * D + hv_idx * D + i)
        v_vec[i] = tl.load(v_ptr + b_idx * H_v * D + hv_idx * D + i)
        i += 1

    # Load gates for token b_idx
    g_val = tl.load(g_ptr + b_idx * H_v + hv_idx)
    beta_val = tl.load(beta_ptr + b_idx * H_v + hv_idx)

    # Compute old_v = sum_i k_i * state[i, i] (diagonal only)
    old_v = 0.0
    j = 0
    while j < D:
        # state_ptr layout: [hv, i, j] -> hv * D*D + i*D + j
        diag = tl.load(state_ptr + hv_idx * D * D + j * D + j)
        k_i = tl.load(k_ptr + b_idx * H_v * D + hv_idx * D + j)
        old_v += k_i * diag
        j += 1

    # new_v = beta * v + (1 - beta) * old_v (scalar per hv)
    new_v = beta_val * v_vec[0] + (1.0 - beta_val) * old_v
    # Note: v_vec[0] is a placeholder; Triton treats it as scalar. We'll use new_v scalar.

    # Compute kT_old = sum_i k_i * sum_j state[i, j]
    kT_old = 0.0
    i = 0
    while i < D:
        k_i = tl.load(k_ptr + b_idx * H_v * D + hv_idx * D + i)
        row_sum = 0.0
        j = 0
        while j < D:
            s_ij = tl.load(state_ptr + hv_idx * D * D + i * D + j)
            row_sum += s_ij
            j += 1
        kT_old += k_i * row_sum
        i += 1

    # Compute kT_newv = sum_i k_i * new_v
    kT_newv = 0.0
    i = 0
    while i < D:
        k_i = tl.load(k_ptr + b_idx * H_v * D + hv_idx * D + i)
        kT_newv += k_i * new_v
        i += 1

    # Update state: new_state[hv, :, :] = g * state - kT_old + kT_newv
    i = 0
    while i < D:
        row_sum = 0.0
        j = 0
        while j < D:
            s_ij = tl.load(state_ptr + hv_idx * D * D + i * D + j)
            row_sum += s_ij
            j += 1
        new_row = g_val * row_sum - kT_old + kT_newv
        j = 0
        while j < D:
            tl.store(state_ptr + hv_idx * D * D + i * D + j, new_row)
            j += 1
        i += 1


@triton.jit
def _output_kernel(
    q_ptr,           # [B, H_q, D] float32
    state_ptr,       # [H_v, D, D] float32
    output_ptr,      # [B, H_v, D] bfloat16
    B: tl.int32,
    H_q: tl.int32,
    H_v: tl.int32,
    D: tl.int32,
):
    pid = tl.program_id(0)
    hv_idx = pid % H_v
    b_idx = pid // H_v  # iterate over tokens t in host loop

    # Form q_exp[hv, :] as concatenation of q[b_idx, 0, :] and q[b_idx, 1, :]
    q0 = tl.zeros((D,), dtype=tl.float32)
    q1 = tl.zeros((D,), dtype=tl.float32)
    i = 0
    while i < D:
        q0[i] = tl.load(q_ptr + b_idx * H_q * D + 0 * D + i)
        q1[i] = tl.load(q_ptr + b_idx * H_q * D + 1 * D + i)
        i += 1

    # For H_v == 2 * H_q, hv < 2 -> q0, else -> q1
    q_exp = q0 if hv_idx < 2 else q1

    # Compute output_vec[hv, :] = q_exp @ state[hv, :, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    j = 0
    while j < D:
        acc = 0.0
        i = 0
        while i < D:
            s_ij = tl.load(state_ptr + hv_idx * D * D + i * D + j)
            acc += s_ij
            i += 1
        out_vec[j] = q_exp[j] * acc
        j += 1

    out_ptr_base = output_ptr + b_idx * H_v * D + hv_idx * D
    j = 0
    while j < D:
        tl.store(out_ptr_base + j, out_vec[j].to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Do not assert on shapes; accept any inputs
        device = q.device
        B = q.shape[0]
        H_q = q.shape[1]
        D = q.shape[2]
        H_v = v.shape[1]
        assert H_v == 2 * H_q, "H_v must be 2 * H_q for this implementation"

        # Cast raw params to float32 for Triton compute
        a_fp32 = a.float()
        dt_bias_fp32 = dt_bias.float()
        b_fp32 = b.float()

        # Allocate g and beta
        g = torch.empty((B, H_v), dtype=torch.float32, device=device)
        beta = torch.empty((B, H_v), dtype=torch.float32, device=device)

        # Launch _compute_g_beta_kernel
        grid_g = (B * H_v,)
        _compute_g_beta_kernel[grid_g](A_log.float(), a_fp32, dt_bias_fp32, b_fp32, g, beta, B, H_v)

        # Ensure k, v are float32
        k_fp32 = k.float()
        v_fp32 = v.float()

        # Create new_state with same shape as input state
        new_state = torch.empty_like(state, dtype=torch.float32, device=device)
        new_state.zero_()

        # Update state per token
        for t in range(B):
            grid_s = (H_v,)
            _state_update_kernel[grid_s](k_fp32[t], v_fp32[t], new_state, g[t], beta[t], B, H_v, D)

        # Output tensor [B, H_v, D], bfloat16
        output = torch.empty((B, H_v, D), dtype=torch.bfloat16, device=device)

        # Launch _output_kernel
        grid_out = (B * H_v,)
        _output_kernel[grid_out](q.float(), new_state, output, B, H_q, H_v, D)

        # Return output and new_state
        return output, new_state


def run(*args):
    return ModelNew()(*args)
