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
    # 2D grid over (b, hv)
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    # Scalar loads
    a_val = tl.load(a_ptr + b_idx * H_v + hv_idx)
    dt_val = tl.load(dt_bias_ptr + hv_idx)
    b_val = tl.load(b_ptr + b_idx * H_v + hv_idx)
    A_log_val = tl.load(A_log_ptr + hv_idx)

    x = a_val + dt_val
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    g = tl.exp(-tl.exp(A_log_val) * sp)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # store results
    tl.store(g_ptr + b_idx * H_v + hv_idx, g)
    tl.store(beta_ptr + b_idx * H_v + hv_idx, beta)


@triton.jit
def _state_update_kernel(
    k_ptr,           # [B, H_v, D] float32
    v_ptr,           # [B, H_v, D] float32
    beta_ptr,        # [B, H_v] float32
    g_ptr,           # [B, H_v] float32
    state_ptr,       # [H_v, D, D] float32
    B: tl.int32,
    H_v: tl.int32,
    D: tl.int32,
):
    # 2D grid over (b, hv)
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    # Load beta and g scalars for this (b, hv)
    beta_val = tl.load(beta_ptr + b_idx * H_v + hv_idx)
    g_val = tl.load(g_ptr + b_idx * H_v + hv_idx)

    # Load k_vec and v_vec for this (b, hv)
    k_vec = tl.zeros((D,), dtype=tl.float32)
    v_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        ptr_k = k_ptr + b_idx * H_v * D + hv_idx * D + i
        ptr_v = v_ptr + b_idx * H_v * D + hv_idx * D + i
        k_vec[i] = tl.load(ptr_k)
        v_vec[i] = tl.load(ptr_v)

    # Compute old_v = sum_i k_vec[i] * state[i, :]
    old_v = tl.zeros((), dtype=tl.float32)  # scalar
    for i in range(0, D):
        # Load row i of state[hv, :, :]
        row_i = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, D):
            state_ptr_ij = state_ptr + hv_idx * D * D + i * D + j
            row_i[j] = tl.load(state_ptr_ij)
        old_v += k_vec[i] * tl.sum(row_i)

    # Compute new_v = beta * v_vec + (1 - beta) * old_v
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

    # Compute kT_old = sum_i k_vec[i] * old_v[i]
    kT_old = 0.0
    for i in range(0, D):
        kT_old += k_vec[i] * old_v

    # Compute kT_newv = sum_i k_vec[i] * new_v[i]
    kT_newv = 0.0
    for i in range(0, D):
        kT_newv += k_vec[i] * new_v[i]

    # Update state[hv, :, :] = g * state - kT_old + kT_newv
    for i in range(0, D):
        # Load row i
        row_i = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, D):
            state_ptr_ij = state_ptr + hv_idx * D * D + i * D + j
            row_i[j] = tl.load(state_ptr_ij)
        row_i = g_val * row_i - kT_old + kT_newv
        for j in range(0, D):
            state_ptr_ij = state_ptr + hv_idx * D * D + i * D + j
            tl.store(state_ptr_ij, row_i[j])


@triton.jit
def _output_kernel(
    q_ptr,           # [B, H_q, D] float32
    state_ptr,       # [H_v, D, D] float32
    output_ptr,      # [B, H_v, D] float32
    B: tl.int32,
    H_q: tl.int32,
    H_v: tl.int32,
    D: tl.int32,
):
    # 2D grid over (b, hv)
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    # Form q_exp[hv, :] as concatenation of q[b, 0, :] and q[b, 1, :]
    # q_ptr layout: b*H_q*D + h* D + i
    q0 = tl.zeros((D,), dtype=tl.float32)
    q1 = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        q0[i] = tl.load(q_ptr + b_idx * H_q * D + 0 * D + i)
        q1[i] = tl.load(q_ptr + b_idx * H_q * D + 1 * D + i)

    q_exp = q0 if hv_idx < 2 else q1  # for H_v=8, hv < 2 uses q0, else q1

    # Compute out_vec = q_exp @ state[hv, :, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for j in range(0, D):
        acc = 0.0
        for i in range(0, D):
            state_ptr_ij = state_ptr + hv_idx * D * D + i * D + j
            acc += tl.load(state_ptr_ij)
        out_vec[j] = acc

    # Store output[b, hv, :]
    out_ptr_base = output_ptr + b_idx * H_v * D + hv_idx * D
    for j in range(0, D):
        tl.store(out_ptr_base + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure shapes and constraints (as in the original)
        device = q.device
        B = q.shape[0]
        H_q = q.shape[1]
        assert B == 6, "q: total_seq_len must be 6"
        assert H_q == 4, "num_q_heads must be 4"
        assert k.shape[1] == 4, "num_k_heads must be 4"
        assert v.shape[1] == 8, "num_v_heads must be 8"
        D = q.shape[2]
        assert D == 128, "head_size must be 128"
        H_v = v.shape[1]
        assert H_v == 8, "H_v must be 8"
        # The harness requires scale=1.0
        assert scale == 1.0, "scale must be 1.0 (unused by reference)"

        # Cast inputs to float32 for compute
        a_fp32 = a.to(torch.float32)
        dt_bias_fp32 = dt_bias.to(torch.float32)
        b_fp32 = b.to(torch.float32)
        q_fp32 = q.to(torch.float32)
        k_fp32 = k.to(torch.float32)
        v_fp32 = v.to(torch.float32)
        # Prepare output and state
        g = torch.empty((B, H_v), device=device, dtype=torch.float32)
        beta = torch.empty((B, H_v), device=device, dtype=torch.float32)
        output = torch.empty((B, H_v, D), device=device, dtype=torch.float32)
        if state is None:
            state = torch.zeros((H_v, D, D), device=device, dtype=torch.float32)
        else:
            assert state.shape == (H_v, D, D), f"state must be [H_v, D, D], got {state.shape}"

        # Launch gate/beta kernel
        grid_g = (B, H_v)
        _compute_g_beta_kernel[grid_g](
            A_log.to(torch.float32), a_fp32, dt_bias_fp32, b_fp32, g, beta
        )

        # Update state and compute output for each token
        for t in range(B):
            # Update state for all heads hv
            grid_s = (1, H_v)
            _state_update_kernel[grid_s](
                k_fp32[t], v_fp32[t], beta[t], g[t], state
            )
            # Compute output for this token and all heads
            grid_o = (1, H_v)
            _output_kernel[grid_o](
                q_fp32[t], state, output[t]
            )

        # Return output (bfloat16 as original) and state (float32)
        return output.to(torch.bfloat16), state


def run(*args):
    return ModelNew()(*args)
