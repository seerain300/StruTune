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
    # 2D grid: (b, hv)
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    if b_idx >= total_seq_len or hv_idx >= H_v:
        return

    # Load scalars for this hv
    A_log_val = tl.load(A_log_ptr + hv_idx)  # float32
    dt_bias_val = tl.load(dt_bias_ptr + hv_idx)  # float32

    # Load a[b, hv] and b[b, hv]
    a_val = tl.load(a_ptr + b_idx * H_v + hv_idx)  # float32
    b_val = tl.load(b_ptr + b_idx * H_v + hv_idx)  # float32

    # Compute x = a + dt_bias
    x = a_val + dt_bias_val

    # softplus(x) = log(1 + exp(x))
    softplus_x = tl.log(1.0 + tl.exp(x))

    # g = exp(-exp(A_log) * softplus(x))
    g_val = tl.exp(-tl.exp(A_log_val) * softplus_x)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b_idx * H_v + hv_idx, g_val)
    tl.store(beta_ptr + b_idx * H_v + hv_idx, beta_val)


@triton.jit
def _state_update_kernel(
    k_ptr,      # [total_seq_len, H_v, D], float32
    v_ptr,      # [total_seq_len, H_v, D], float32
    beta_ptr,   # [total_seq_len, H_v], float32
    g_ptr,      # [total_seq_len, H_v], float32
    state_ptr,  # [H_v, D, D], float32
    total_seq_len: tl.constexpr,
    H_v: tl.constexpr,
    D: tl.constexpr,
):
    # 2D grid: (t, hv)
    t_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    if t_idx >= total_seq_len or hv_idx >= H_v:
        return

    # Load k[t, hv, :] vector (length D)
    k_vec = tl.load(k_ptr + t_idx * H_v * D + hv_idx * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)

    # Compute old_v = k @ state[hv, :, :] via loop over D
    old_v = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        row_ptr = state_ptr + hv_idx * (D * D) + i * D
        row_vec = tl.load(row_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        old_v += k_vec[i] * row_vec

    # Load beta[b, hv] and g[b, hv]
    beta_val = tl.load(beta_ptr + t_idx * H_v + hv_idx)
    g_val = tl.load(g_ptr + t_idx * H_v + hv_idx)

    # Load v[t, hv, :]
    v_vec = tl.load(v_ptr + t_idx * H_v * D + hv_idx * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

    # Compute kT_old = dot(k, old_v) and kT_newv = dot(k, new_v)
    kT_old = tl.sum(k_vec * old_v, axis=0)
    kT_newv = tl.sum(k_vec * new_v, axis=0)

    # Load current state[hv, :, :]
    curr_state = tl.zeros((D, D), dtype=tl.float32)
    for i in range(0, D):
        row_ptr = state_ptr + hv_idx * (D * D) + i * D
        row_vec = tl.load(row_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        curr_state[i, :] = row_vec

    # Update state
    new_state = curr_state * g_val - kT_old + kT_newv

    # Store updated state
    for i in range(0, D):
        row_ptr = state_ptr + hv_idx * (D * D) + i * D
        tl.store(row_ptr + tl.arange(0, D), new_state[i, :], mask=tl.arange(0, D) < D)


@triton.jit
def _output_kernel(
    q_ptr,        # [total_seq_len, H_q, D], float32
    state_ptr,    # [H_v, D, D], float32
    output_ptr,   # [total_seq_len, H_v, D], float32
    scale,        # float32 scalar
    total_seq_len: tl.constexpr,
    H_q: tl.constexpr,
    H_v: tl.constexpr,
    D: tl.constexpr,
):
    # 2D grid: (t, hv)
    t_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    if t_idx >= total_seq_len or hv_idx >= H_v:
        return

    # Form q_exp[t, hv, :] = concatenate(q[t, 0, :], q[t, 1, :]) if H_v % H_q == 2, else q[t, 0, :] duplicated
    # Given the provided setup H_v=8, H_q=4, so we use 2.
    q0 = tl.load(q_ptr + t_idx * H_q * D + 0 * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    q1 = tl.load(q_ptr + t_idx * H_q * D + 1 * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    q_exp = tl.zeros((2 * D,), dtype=tl.float32)
    q_exp[:D] = q0
    q_exp[D:] = q1

    # Compute out[t, hv, :] = scale * q_exp @ state[hv, :, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for j in range(0, D):
        col_j = tl.zeros((D,), dtype=tl.float32)
        # sum over rows i of q_exp[i] * state[i, j]
        for i in range(0, D):
            # q_exp[i] contribution to out_vec[j]
            q_i = q_exp[i] if i < (2 * D) else 0.0
            state_ij = tl.load(state_ptr + hv_idx * (D * D) + i * D + j, mask=j < D, other=0.0)
            out_vec[j] += q_i * state_ij

    out_vec = scale * out_vec

    # Store output[t, hv, :]
    tl.store(output_ptr + t_idx * H_v * D + hv_idx * D + tl.arange(0, D), out_vec, mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,    # [total_seq_len, H_q, D], bfloat16
        k: torch.Tensor,    # [total_seq_len, H_v, D], bfloat16
        v: torch.Tensor,    # [total_seq_len, H_v, D], bfloat16
        state: torch.Tensor,# [H_v, D, D], float32 or None
        A_log: torch.Tensor,  # [H_v], float32
        a: torch.Tensor,      # [total_seq_len, H_v], bfloat16
        dt_bias: torch.Tensor,# [H_v], float32
        b: torch.Tensor,      # [total_seq_len, H_v], bfloat16
        cu_seqlens: torch.Tensor, # [num_buckets+1], int64 (unused in this forward; we handle one segment per call)
        scale: float = 1.0,
    ):
        # Enforce fixed sizes per provided setup
        total_seq_len, H_q, D = q.shape
        assert H_q == 4, "H_q must be 4"
        assert D == 128, "D must be 128"
        H_v = v.shape[1]
        assert H_v == 8, "H_v must be 8"
        assert k.shape == (total_seq_len, H_v, D)
        assert v.shape == (total_seq_len, H_v, D)

        device = q.device

        # Compute g and beta in Triton
        g = torch.empty((total_seq_len, H_v), device=device, dtype=torch.float32)
        beta = torch.empty((total_seq_len, H_v), device=device, dtype=torch.float32)

        # Cast inputs to float32 for compute
        a_fp32 = a.to(torch.float32)
        b_fp32 = b.to(torch.float32)
        dt_bias_fp32 = dt_bias.to(torch.float32)
        A_log_fp32 = A_log.to(torch.float32)
        q_fp32 = q.to(torch.float32)
        k_fp32 = k.to(torch.float32)
        v_fp32 = v.to(torch.float32)

        # Launch compute_g_beta_kernel
        grid_g = (total_seq_len, H_v)
        _compute_g_beta_kernel[grid_g](
            A_log_fp32, a_fp32, dt_bias_fp32, b_fp32,
            g, beta,
            total_seq_len=total_seq_len, H_v=H_v,
        )

        # Prepare output and new state
        output = torch.empty((total_seq_len, H_v, D), device=device, dtype=torch.bfloat16)
        new_state = torch.empty((H_v, D, D), device=device, dtype=torch.float32)

        if state is None:
            # Initialize state to zeros
            state = torch.zeros((H_v, D, D), device=device, dtype=torch.float32)

        # Launch state_update_kernel for each token
        for t in range(total_seq_len):
            _state_update_kernel[(1, H_v)](
                k_fp32[t], v_fp32[t], beta[t], g[t], state,
                total_seq_len=total_seq_len, H_v=H_v, D=D,
            )
            # After update, write to new_state
            new_state.copy_(state)

        # Compute output for each token using output_kernel; also loop over t
        for t in range(total_seq_len):
            _output_kernel[(1, H_v)](
                q_fp32[t], new_state, output[t],
                float(scale),
                total_seq_len=total_seq_len, H_q=H_q, H_v=H_v, D=D,
            )

        return output, new_state


def run(*args):
    return ModelNew()(*args)
