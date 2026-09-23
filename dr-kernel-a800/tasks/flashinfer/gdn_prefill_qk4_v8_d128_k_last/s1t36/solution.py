import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(
    A_log_ptr,           # [HV] float32
    a_ptr,               # [B, HV] bfloat16
    dt_bias_ptr,         # [HV] float32
    g_ptr,               # [B, HV] float32
    beta_ptr,            # [B, HV] float32
    B: tl.int32,         # total_seq_len
    HV: tl.int32,        # number of heads (H_v)
):
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)
    # bounds check
    if b_idx >= B:
        return
    # Load A_log[hv], a[b, hv], dt_bias[hv]
    A_log_val = tl.load(A_log_ptr + hv_idx)
    a_val = tl.load(a_ptr + b_idx * HV + hv_idx).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + hv_idx)

    # softplus(x) = log1p(exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_bias_val))
    g_val = tl.exp(-tl.exp(A_log_val) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-a_val - dt_bias_val))
    # Store results
    tl.store(g_ptr + b_idx * HV + hv_idx, g_val)
    tl.store(beta_ptr + b_idx * HV + hv_idx, beta_val)


@triton.jit
def _state_update_kernel(
    k_ptr,               # [B, H, D] bfloat16 (note: H could be 4 or 8; we only use H=1 in loop, but pass D)
    v_ptr,               # [B, H, D] bfloat16 (we use H_v=8)
    state_ptr,           # [H_v, D, D] float32
    g_ptr,               # [B, HV] float32
    beta_ptr,            # [B, HV] float32
    output_ptr,          # [B, H_v, D] float32
    scale: tl.float32,
    B: tl.int32,
    H_q: tl.int32,       # num_q_heads, used for q_exp mapping
    H_v: tl.int32,       # num_v_heads, also H_v
    D: tl.int32,
):
    t_idx = tl.program_id(0)  # 0..B-1
    hv_idx = tl.program_id(1) # 0..H_v-1
    if t_idx >= B:
        return

    # Load g and beta for this (t, hv)
    g_val = tl.load(g_ptr + t_idx * H_v + hv_idx)
    beta_val = tl.load(beta_ptr + t_idx * H_v + hv_idx)

    # Compute old_v = k[t, hv, :] @ state[hv, :, :]
    old_v = 0.0
    for i in range(0, D):
        # k[t, hv, i]
        k_val = tl.load(k_ptr + t_idx * H_q * D + hv_idx * D + i).to(tl.float32)
        row = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, D):
            state_ptr_ij = state_ptr + hv_idx * D * D + i * D + j
            row[j] = tl.load(state_ptr_ij)
        old_v += k_val * tl.sum(row)

    # new_v = beta * v[t, hv, :] + (1 - beta) * old_v
    new_v = beta_val * tl.load(v_ptr + t_idx * H_v * D + hv_idx * D).to(tl.float32) + (1.0 - beta_val) * old_v

    # kT_old = sum_i k[t, hv, i] * old_v[i]
    kT_old = 0.0
    for i in range(0, D):
        k_val = tl.load(k_ptr + t_idx * H_q * D + hv_idx * D + i).to(tl.float32)
        # old_v[i] = state[t, hv, i, :] dot k[t, hv, i]
        row = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, D):
            state_ptr_ij = state_ptr + hv_idx * D * D + i * D + j
            row[j] = tl.load(state_ptr_ij)
        kT_old += k_val * tl.sum(row)

    # kT_newv = sum_i k[t, hv, i] * new_v[i]
    kT_newv = 0.0
    for i in range(0, D):
        k_val = tl.load(k_ptr + t_idx * H_q * D + hv_idx * D + i).to(tl.float32)
        # new_v[i] = state[t, hv, i, :] dot new_v
        row = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, D):
            state_ptr_ij = state_ptr + hv_idx * D * D + i * D + j
            row[j] = tl.load(state_ptr_ij)
        kT_newv += k_val * tl.sum(row)

    # Update state
    state_row = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        for j in range(0, D):
            state_ptr_ij = state_ptr + hv_idx * D * D + i * D + j
            val = tl.load(state_ptr_ij)
            state_row[j] = val * g_val - kT_old + kT_newv
            tl.store(state_ptr_ij, state_row[j])

    # Optionally compute output here for this (t, hv) if needed; not required for state update.


@triton.jit
def _output_kernel(
    q_ptr,               # [B, H_q, D] bfloat16
    state_ptr,           # [H_v, D, D] float32
    output_ptr,          # [B, H_v, D] float32 (we'll use scale=1.0)
    scale: tl.float32,
    B: tl.int32,
    H_q: tl.int32,
    H_v: tl.int32,
    D: tl.int32,
):
    t_idx = tl.program_id(0)  # 0..B-1
    hv_idx = tl.program_id(1) # 0..H_v-1
    if t_idx >= B:
        return

    # q_exp mapping: for H_v == 2*H_q, hv < 2 -> q[t,0,:], else hv-2 -> q[t,1,:]
    q_exp_vec = tl.zeros((D,), dtype=tl.float32)
    if hv_idx < 2:
        for i in range(0, D):
            q_val = tl.load(q_ptr + t_idx * H_q * D + 0 * D + i).to(tl.float32)
            q_exp_vec[i] = q_val
    else:
        for i in range(0, D):
            q_val = tl.load(q_ptr + t_idx * H_q * D + 1 * D + i).to(tl.float32)
            q_exp_vec[i] = q_val

    # output_vec = scale * q_exp @ state[hv, :, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for j in range(0, D):
        acc = 0.0
        for i in range(0, D):
            state_ptr_ij = state_ptr + hv_idx * D * D + i * D + j
            acc += tl.load(state_ptr_ij)
        out_vec[j] = scale * acc

    # Store output
    out_ptr_base = output_ptr + t_idx * H_v * D + hv_idx * D
    for j in range(0, D):
        tl.store(out_ptr_base + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Shapes and asserts as in original code
        device = q.device
        total_seq_len = q.shape[0]
        assert total_seq_len == 6, "q: total_seq_len must be 6"
        assert q.shape[1] == 4, "num_q_heads must be 4"
        assert k.shape[1] == 4, "num_k_heads must be 4"
        assert v.shape[1] == 8, "num_v_heads must be 8"
        head_size = q.shape[2]
        assert head_size == 128, "head size must be 128"

        # Ensure dtypes
        a = a.to(torch.bfloat16)
        b = b.to(torch.bfloat16)
        A_log = A_log.to(torch.float32)
        dt_bias = dt_bias.to(torch.float32)

        B = total_seq_len
        H_q = 4
        H_v = 8
        D = head_size

        # Allocate g and beta tensors (float32)
        g = torch.empty((B, H_v), dtype=torch.float32, device=device)
        beta = torch.empty((B, H_v), dtype=torch.float32, device=device)

        # Launch _compute_g_beta_kernel
        grid_g = (B, H_v)
        _compute_g_beta_kernel[grid_g](
            A_log, a, dt_bias, g, beta,
            B, H_v,
        )

        # Initialize output and new state
        output = torch.empty((B, H_v, D), dtype=torch.bfloat16, device=device)
        # If state is None, initialize as zeros [H_v, D, D] float32
        if state is None:
            state_mat = torch.zeros((H_v, D, D), dtype=torch.float32, device=device)
        else:
            # state is [1, 8, 128, 128] in some tests, but we need [H_v, D, D]; take the first element
            state_mat = state[0].transpose(-1, -2).contiguous()  # [1, 8, 128, 128] -> [8, 128, 128] float32

        # Launch _state_update_kernel
        grid_state = (B, H_v)
        _state_update_kernel[grid_state](
            k, v, state_mat, g, beta, output, 1.0,
            B, H_q, H_v, D,
        )

        # Return output and updated state (keep state_mat shape [H_v, D, D])
        return output, state_mat


def run(*args):
    return ModelNew()(*args)
