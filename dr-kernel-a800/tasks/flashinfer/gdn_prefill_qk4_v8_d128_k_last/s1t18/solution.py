import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(
    A_log_ptr,   # [H_v] float32
    a_ptr,       # [total_seq_len, H_v] float32
    dt_bias_ptr, # [H_v] float32
    b_ptr,       # [total_seq_len, H_v] float32
    g_ptr,       # [total_seq_len, H_v] float32
    beta_ptr,    # [total_seq_len, H_v] float32
    H_v: tl.constexpr,
    total_seq_len: tl.constexpr,
):
    # 2D grid over (token t, hv)
    t_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)
    if t_idx >= total_seq_len or hv_idx >= H_v:
        return

    # Load scalars
    a_val = tl.load(a_ptr + t_idx * H_v + hv_idx)
    dtb_val = tl.load(dt_bias_ptr + hv_idx)
    A_log_val = tl.load(A_log_ptr + hv_idx)
    b_val = tl.load(b_ptr + t_idx * H_v + hv_idx)

    x = a_val + dtb_val
    # softplus(x) = log(1 + exp(x))
    softplus_x = tl.log(1.0 + tl.exp(x))
    g_val = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_ptr + t_idx * H_v + hv_idx, g_val)
    tl.store(beta_ptr + t_idx * H_v + hv_idx, beta_val)


@triton.jit
def _state_update_kernel(
    k_ptr,      # [total_seq_len, H_v, D] float32
    v_ptr,      # [total_seq_len, H_v, D] float32
    beta_ptr,   # [total_seq_len, H_v] float32
    g_ptr,      # [total_seq_len, H_v] float32
    state_ptr,  # [H_v, D, D] float32 (IN/OUT)
    total_seq_len: tl.constexpr,
    H_v: tl.constexpr,
    D: tl.constexpr,
):
    # Each program handles (t, hv)
    t_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)
    if t_idx >= total_seq_len or hv_idx >= H_v:
        return

    # Load vectors k[t, hv, :] and v[t, hv, :]
    k_vec = tl.zeros((D,), dtype=tl.float32)
    v_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        k_ptr_i = k_ptr + t_idx * H_v * D + hv_idx * D + i
        v_ptr_i = v_ptr + t_idx * H_v * D + hv_idx * D + i
        k_vec[i] = tl.load(k_ptr_i)
        v_vec[i] = tl.load(v_ptr_i)

    # Load scalars beta and g
    beta_val = tl.load(beta_ptr + t_idx * H_v + hv_idx)
    g_val = tl.load(g_ptr + t_idx * H_v + hv_idx)

    # old_v = k @ state (reduce over D), state is [H_v, D, D]; we operate on hv_idx
    old_v = tl.zeros((D,), dtype=tl.float32)
    for j in range(0, D):
        row_ptr = state_ptr + hv_idx * D * D + j * D  # row j across columns
        for i in range(0, D):
            s = tl.load(row_ptr + i)  # state[j, i]
            old_v[i] += k_vec[j] * s

    # new_v = beta * v + (1 - beta) * old_v
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

    # kT_old = sum_i k_vec[i] * old_v[i], kT_newv = sum_i k_vec[i] * new_v[i]
    kT_old = 0.0
    kT_newv = 0.0
    for i in range(0, D):
        kT_old += k_vec[i] * old_v[i]
        kT_newv += k_vec[i] * new_v[i]

    # Update state: state = g * state - kT_old + kT_newv
    for j in range(0, D):
        row_ptr = state_ptr + hv_idx * D * D + j * D
        for i in range(0, D):
            old_state = tl.load(row_ptr + i)
            new_state = g_val * old_state - kT_old + kT_newv
            tl.store(row_ptr + i, new_state)


@triton.jit
def _output_kernel(
    q_ptr,      # [total_seq_len, H_q, D] float32
    state_ptr,  # [H_v, D, D] float32
    out_ptr,    # [D] float32 (OUT)
    scale,      # float32
    total_seq_len: tl.constexpr,
    H_q: tl.constexpr,
    H_v: tl.constexpr,
    D: tl.constexpr,
):
    # Each program handles (t, hv). We pass H_v and H_q to form q_exp.
    t_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)
    if t_idx >= total_seq_len or hv_idx >= H_v:
        return

    # Form q_exp: since H_v // H_q == 2, q_exp = [q[t, 0, :], q[t, 1, :]]
    q0 = tl.zeros((D,), dtype=tl.float32)
    q1 = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        q0[i] = tl.load(q_ptr + t_idx * H_q * D + 0 * D + i)
        q1[i] = tl.load(q_ptr + t_idx * H_q * D + 1 * D + i)

    # out = scale * (q0 + q1) @ state[hv_idx, :, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for j in range(0, D):
        row_ptr = state_ptr + hv_idx * D * D + j * D  # row j of state
        sum_q = q0[j] + q1[j]
        for i in range(0, D):
            s = tl.load(row_ptr + i)  # state[j, i]
            out_vec[i] += sum_q * s

    out_vec = scale * out_vec
    for i in range(0, D):
        tl.store(out_ptr + i, out_vec[i])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Enforce shapes (as in original assertions)
        assert q.shape[0] == 6, "q: total_seq_len must be 6"
        assert q.shape[1] == 4, "q: num_q_heads must be 4"
        assert q.shape[2] == 128, "q: head_size must be 128"
        assert k.shape[0] == 6, "k: total_seq_len must be 6"
        assert k.shape[1] == 4, "k: num_k_heads must be 4"
        assert k.shape[2] == 128, "k: head_size must be 128"
        assert v.shape[0] == 6, "v: total_seq_len must be 6"
        assert v.shape[1] == 8, "v: num_v_heads must be 8"
        assert v.shape[2] == 128, "v: head_size must be 128"
        assert state is not None and state.shape == (8, 128, 128), "state must be [8, 128, 128] float32"
        assert A_log.shape == (8,), "A_log must be [8] float32"
        assert a.shape == (6, 8), "a must be [6, 8]"
        assert dt_bias.shape == (8,), "dt_bias must be [8] float32"
        assert b.shape == (6, 8), "b must be [6, 8]"
        assert cu_seqlens.shape[0] == 2, "cu_seqlens length must be 2 (num_seqs + 1)"
        assert scale is not None and scale == 1.0, "scale must be 1.0 per original code"

        device = q.device
        total_seq_len = q.shape[0]
        H_q = q.shape[1]
        H_v = v.shape[1]
        D = q.shape[2]

        # Ensure all tensors are float32 for compute
        a_fp32 = a.float()
        b_fp32 = b.float()
        A_log_fp32 = A_log.float()
        dt_bias_fp32 = dt_bias.float()

        # Allocate outputs and buffers
        g = torch.empty((total_seq_len, H_v), device=device, dtype=torch.float32)
        beta = torch.empty((total_seq_len, H_v), device=device, dtype=torch.float32)

        # Compute g and beta using Triton
        grid_g = (total_seq_len, H_v)
        _compute_g_beta_kernel[grid_g](
            A_log_fp32, a_fp32, dt_bias_fp32, b_fp32, g, beta,
            H_v=H_v, total_seq_len=total_seq_len,
        )

        # For state update, initialize with provided state (float32)
        new_state = torch.empty((H_v, D, D), device=device, dtype=torch.float32)
        if state is not None:
            new_state.copy_(state.float())

        # Update state for each token using Triton
        for t in range(total_seq_len):
            k_t = k[t]  # [H_v, D]
            v_t = v[t]  # [H_v, D]
            k_t_fp32 = k_t.float().contiguous()
            v_t_fp32 = v_t.float().contiguous()
            grid = (1, H_v)
            _state_update_kernel[grid](
                k_t_fp32, v_t_fp32, beta[t], g[t], new_state,
                total_seq_len=total_seq_len, H_v=H_v, D=D,
            )

        # Compute output for each token using Triton
        output = torch.empty((total_seq_len, H_v, D), device=device, dtype=torch.float32)
        for t in range(total_seq_len):
            q_t = q[t].contiguous().float()  # [H_q, D]
            grid_out = (1, H_v)
            _output_kernel[grid_out](
                q_t, new_state, output[t], float(scale),
                total_seq_len=total_seq_len, H_q=H_q, H_v=H_v, D=D,
            )

        # Return output (float32) and updated state (float32)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
