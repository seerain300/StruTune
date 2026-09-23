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
    total_seq_len: tl.constexpr,
    H_v: tl.constexpr,
):
    # 2D grid: pid0 = b index, pid1 = hv index
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    # Masks
    b_mask = b_idx < total_seq_len
    hv_mask = hv_idx < H_v
    # Load scalars for current hv
    a_val = tl.load(a_ptr + b_idx * H_v + hv_idx, mask=b_mask & hv_mask, other=0.0)
    dt_val = tl.load(dt_bias_ptr + hv_idx, mask=hv_mask, other=0.0)
    b_val = tl.load(b_ptr + b_idx * H_v + hv_idx, mask=b_mask & hv_mask, other=0.0)
    A_val = tl.load(A_log_ptr + hv_idx, mask=hv_mask, other=0.0)

    x = a_val + dt_val
    softplus_x = tl.log(1 + tl.exp(x))
    g_val = tl.exp(-tl.exp(A_val) * softplus_x)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store
    tl.store(g_ptr + b_idx * H_v + hv_idx, g_val, mask=b_mask & hv_mask)
    tl.store(beta_ptr + b_idx * H_v + hv_idx, beta_val, mask=b_mask & hv_mask)


@triton.jit
def _state_update_kernel(
    k_ptr,       # [total_seq_len, H_v, D] float32
    v_ptr,       # [total_seq_len, H_v, D] float32
    beta_ptr,    # [total_seq_len, H_v] float32
    g_ptr,       # [total_seq_len, H_v] float32
    state_ptr,   # [H_v, D, D] float32, updated in-place
    total_seq_len: tl.constexpr,
    H_v: tl.constexpr,
    D: tl.constexpr,
):
    # 2D grid: pid0 = b index, pid1 = hv index
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    b_mask = b_idx < total_seq_len
    hv_mask = hv_idx < H_v

    # Load k vector [D] and v vector [D]
    k_vec = tl.zeros((D,), dtype=tl.float32)
    v_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        # k[b, hv, i]
        ptr_k = k_ptr + b_idx * H_v * D + hv_idx * D + i
        k_vec[i] = tl.load(ptr_k, mask=b_mask & hv_mask, other=0.0)
        # v[b, hv, i]
        ptr_v = v_ptr + b_idx * H_v * D + hv_idx * D + i
        v_vec[i] = tl.load(ptr_v, mask=b_mask & hv_mask, other=0.0)

    # Load scalars for gate
    g_val = tl.load(g_ptr + b_idx * H_v + hv_idx, mask=b_mask & hv_mask, other=0.0)
    beta_val = tl.load(beta_ptr + b_idx * H_v + hv_idx, mask=b_mask & hv_mask, other=0.0)

    # old_v = k @ state (reduce over D)
    old_v = tl.zeros((D,), dtype=tl.float32)
    # Loop over rows i of state
    for i_row in range(0, D):
        # Load state[hv, i_row, :] as a vector of length D
        state_row_ptr = state_ptr + hv_idx * D + i_row * D
        state_row = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, D):
            state_row[j] = tl.load(state_row_ptr + j, mask=hv_mask, other=0.0)
        old_v += k_vec[i_row] * state_row  # scalar k_vec[i_row] times vector state_row

    # new_v = beta * v + (1 - beta) * old_v
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

    # kT_old = sum_i k_vec[i] * old_v[i], kT_newv = sum_i k_vec[i] * new_v[i]
    kT_old = 0.0
    for i in range(0, D):
        kT_old += k_vec[i] * old_v[i]
    kT_newv = 0.0
    for i in range(0, D):
        kT_newv += k_vec[i] * new_v[i]

    # Update state: state = g * state - kT_old + kT_newv
    # First multiply g (scalar) with entire state
    for i_row in range(0, D):
        state_row_ptr = state_ptr + hv_idx * D + i_row * D
        for j in range(0, D):
            val = tl.load(state_row_ptr + j, mask=hv_mask, other=0.0)
            val = val * g_val - kT_old + kT_newv
            tl.store(state_row_ptr + j, val, mask=hv_mask)


@triton.jit
def _output_kernel(
    q_ptr,       # [total_seq_len, H_q, D] float32
    state_ptr,   # [H_v, D, D] float32
    out_ptr,     # [H_v, D] float32 (will be converted to bfloat16)
    scale,       # float32 scalar
    total_seq_len: tl.constexpr,
    H_q: tl.constexpr,
    H_v: tl.constexpr,
    D: tl.constexpr,
):
    # 2D grid: pid0 = b index, pid1 = hv index
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    b_mask = b_idx < total_seq_len
    hv_mask = hv_idx < H_v

    # Form q_exp: in provided setup H_v // H_q == 2, so we use q[t, 0, :] and q[t, 1, :]
    # q_exp_vec = concatenate(q[t, 0, :], q[t, 1, :]) when hv < 2, else q[t, 0, :]
    q_exp_vec = tl.zeros((D,), dtype=tl.float32)
    # Only one of these will be used due to mask on hv
    if hv_idx < 2:
        # first half from q0, second half from q1
        for i in range(0, D):
            q0_ptr = q_ptr + b_idx * H_q * D + 0 * D + i
            q1_ptr = q_ptr + b_idx * H_q * D + 1 * D + i
            q_exp_vec[i] = tl.load(q0_ptr, mask=b_mask, other=0.0)
        for i in range(D, 2 * D):
            q_exp_vec[i] = tl.load(q1_ptr - D + (i - D), mask=b_mask, other=0.0)
    else:
        for i in range(0, D):
            q0_ptr = q_ptr + b_idx * H_q * D + 0 * D + i
            q_exp_vec[i] = tl.load(q0_ptr, mask=b_mask, other=0.0)

    # out = scale * q_exp_vec @ state[hv, :, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        state_row_ptr = state_ptr + hv_idx * D + i * D
        state_row = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, D):
            state_row[j] = tl.load(state_row_ptr + j, mask=hv_mask, other=0.0)
        out_vec += q_exp_vec[i] * state_row

    out_vec = scale * out_vec
    # Store to out_ptr[hv, :]
    out_ptr[hv_idx, 0: D] = out_vec  # Triton expects pointer arithmetic; we store as 1D
    # Note: out_ptr is [H_v, D] contiguous; store via pointer arithmetic
    # We'll pass out_ptr as 1D pointer; using out_ptr[hv_idx, :] is not valid in Triton.
    # Instead, compute base for this hv and store elements
    # out_ptr is actually contiguous [H_v, D], so offset = hv_idx * D
    for j in range(0, D):
        tl.store(out_ptr + hv_idx * D + j, out_vec[j], mask=hv_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, H_q=4, H_v=8, D=128):
        super().__init__()
        self.H_q = H_q
        self.H_v = H_v
        self.D = D

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure device and dtype
        device = q.device
        total_seq_len = q.shape[0]
        H_q = self.H_q
        H_v = self.H_v
        D = self.D

        # Cast inputs to float32 for compute
        q_fp32 = q.float().contiguous()
        k_fp32 = k.float().contiguous()
        v_fp32 = v.float().contiguous()
        a_fp32 = a.float().contiguous()
        dt_bias_fp32 = dt_bias.float().contiguous()
        b_fp32 = b.float().contiguous()

        # Allocate outputs
        g = torch.empty((total_seq_len, H_v), device=device, dtype=torch.float32)
        beta = torch.empty((total_seq_len, H_v), device=device, dtype=torch.float32)

        # Compute g and beta using Triton
        _compute_g_beta_kernel[(total_seq_len, H_v)](
            A_log.float().contiguous(), a_fp32, dt_bias_fp32, b_fp32, g, beta,
            total_seq_len=total_seq_len, H_v=H_v,
        )

        # Prepare state (k-last [H, V, K] with K=D, V=H_v, H not used here)
        if state is None:
            state = torch.zeros((H_v, D, D), device=device, dtype=torch.float32)
        else:
            # Ensure contiguous float32
            state = state.float().contiguous()

        # Output buffer
        output = torch.empty((total_seq_len, H_v, D), device=device, dtype=torch.bfloat16)

        # Update state per token
        for t in range(total_seq_len):
            # Launch state update for this token t
            _state_update_kernel[(1, H_v)](
                k_fp32[t], v_fp32[t], beta[t], g[t], state,
                total_seq_len=total_seq_len, H_v=H_v, D=D,
            )
            # Compute output for this token
            _output_kernel[(1, H_v)](
                q_fp32[t], state, output[t], float(scale),
                total_seq_len=total_seq_len, H_q=H_q, H_v=H_v, D=D,
            )

        # Return output (bfloat16) and new_state (float32); state was updated in-place in kernel
        return output, state


def run(*args):
    return ModelNew()(*args)
