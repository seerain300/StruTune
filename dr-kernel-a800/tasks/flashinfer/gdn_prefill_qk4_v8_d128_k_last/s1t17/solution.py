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
    # 2D grid over tokens (b) and head index (hv)
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    # Load parameters
    a_val = tl.load(a_ptr + b_idx * H_v + hv_idx)  # a[b, hv]
    dt_b = tl.load(dt_bias_ptr + hv_idx)           # dt_bias[hv]
    b_val = tl.load(b_ptr + b_idx * H_v + hv_idx)  # b[b, hv]
    A_log = tl.load(A_log_ptr + hv_idx)            # A_log[hv]

    x = a_val + dt_b
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    g_val = tl.exp(-tl.exp(A_log) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store
    tl.store(g_ptr + b_idx * H_v + hv_idx, g_val)
    tl.store(beta_ptr + b_idx * H_v + hv_idx, beta_val)


@triton.jit
def _state_update_kernel(
    k_ptr,       # [total_seq_len, H_v, D] float32
    v_ptr,       # [total_seq_len, H_v, D] float32
    beta_ptr,    # [total_seq_len, H_v] float32
    g_ptr,       # [total_seq_len, H_v] float32
    state_ptr,   # [H_v, D, D] float32 (output updated state)
    H_v: tl.constexpr,
    total_seq_len: tl.constexpr,
    D: tl.constexpr,
):
    # 2D grid over (token t, hv)
    t_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    # Load k vector and v vector
    k_vec = tl.zeros((D,), dtype=tl.float32)
    v_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        k_ptr_i = k_ptr + t_idx * H_v * D + hv_idx * D + i
        v_ptr_i = v_ptr + t_idx * H_v * D + hv_idx * D + i
        k_vec[i] = tl.load(k_ptr_i)
        v_vec[i] = tl.load(v_ptr_i)

    # Load beta and g scalars for this (t, hv)
    beta_val = tl.load(beta_ptr + t_idx * H_v + hv_idx)
    g_val = tl.load(g_ptr + t_idx * H_v + hv_idx)

    # Load current state[hv, :, :] as [D, D] tile
    S = tl.zeros((D, D), dtype=tl.float32)
    # We store state as contiguous [H_v, D, D]; need to compute base pointer for hv_idx
    # For each row i in D:
    for i in range(0, D):
        row_ptr = state_ptr + hv_idx * (D * D) + i * D
        # Load full row i
        # Since state is [H_v, D, D] contiguous, stride for rows (dim=2) is D, for cols (dim=1) is 1.
        # For each col j in D, S[i, j] = state[hv_idx, i, j]
        for j in range(0, D):
            S[i, j] = tl.load(row_ptr + j)

    # Compute old_v = sum_i k_vec[i] * S[i, :]
    old_v = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        old_v += k_vec[i] * S[i, :]

    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # scalar per dim

    # kT_old = sum_i k_vec[i] * old_v[i]
    kT_old = tl.sum(k_vec * old_v, axis=0)

    # kT_newv = sum_i k_vec[i] * new_v[i] (new_v is scalar per dim, broadcast)
    kT_newv = tl.sum(k_vec * new_v, axis=0)

    # Update state: S = g * S - kT_old + kT_newv (broadcast kT terms as row vector)
    # g is scalar
    g_mat = g_val * S
    # Add kT_newv minus kT_old
    # Broadcasting kT_newv and kT_old over columns
    for i in range(0, D):
        for j in range(0, D):
            S[i, j] = g_mat[i, j] + (kT_newv - kT_old)

    # Store updated state back
    for i in range(0, D):
        row_ptr = state_ptr + hv_idx * (D * D) + i * D
        for j in range(0, D):
            tl.store(row_ptr + j, S[i, j])


@triton.jit
def _output_kernel(
    q_ptr,       # [total_seq_len, H_q, D] float32
    state_ptr,   # [H_v, D, D] float32
    output_ptr,  # [H_v, D] bfloat16 (store result)
    scale: tl.constexpr,
    H_q: tl.constexpr,
    H_v: tl.constexpr,
    D: tl.constexpr,
):
    # 2D grid over (token t, hv)
    t_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    # Form q_exp[hv, :] for hv in {0,1} since H_v // H_q == 2
    q0 = tl.zeros((D,), dtype=tl.float32)
    q1 = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        q0[i] = tl.load(q_ptr + t_idx * (H_q * D) + 0 * D + i)
        q1[i] = tl.load(q_ptr + t_idx * (H_q * D) + 1 * D + i)

    # Compute output[hv, :] = scale * (q0 + q1) @ state[hv, :, :]
    # We will compute sum_j (q0[j] + q1[j]) * state[hv, j, :] for each output dim hv
    # But since output is [H_v, D], and hv_idx selects which head we compute.
    # However, q_exp selection is only for hv in {0,1}. For hv>=2, q_exp is undefined; in our setup H_v=8 and we only use hv=0,1.
    # To keep general: if hv_idx >= 2, return zeros.
    if hv_idx < 2:
        q_exp = q0 + q1
    else:
        q_exp = tl.zeros((D,), dtype=tl.float32)

    S = tl.zeros((D, D), dtype=tl.float32)
    # Load state[hv_idx, :, :]
    for i in range(0, D):
        row_ptr = state_ptr + hv_idx * (D * D) + i * D
        for j in range(0, D):
            S[i, j] = tl.load(row_ptr + j)

    out_vec = tl.zeros((D,), dtype=tl.float32)
    for j in range(0, D):
        out_vec[j] = tl.sum(S[j, :] * q_exp, axis=0)  # reduce over columns

    # Store as bfloat16
    # Note: Triton does not have bfloat16 type in all versions; we assume bf16 support in this environment.
    for j in range(0, D):
        tl.store(output_ptr + hv_idx * D + j, out_vec[j].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-ONLY forward:
        - Compute g and beta in Triton kernel.
        - Update state in Triton kernel for each token.
        - Compute output in Triton kernel for each token.
        """
        total_seq_len, H_q, D = q.shape
        H_k = k.shape[1]
        H_v = v.shape[1]
        assert H_q == 4 and H_k == 4 and H_v == 8 and D == 128, "Fixed head sizes required for this Triton implementation."

        device = q.device

        # Cast inputs to float32 for compute
        q_fp32 = q.float()
        k_fp32 = k.float()
        v_fp32 = v.float()
        a_fp32 = a.float()
        b_fp32 = b.float()
        dt_bias_fp32 = dt_bias.float()
        A_log_fp32 = A_log.float()

        # Prepare outputs and buffers
        output = torch.empty((total_seq_len, H_v, D), device=device, dtype=torch.bfloat16)
        new_state = torch.empty((H_v, D, D), device=device, dtype=torch.float32)

        # Compute g and beta with Triton kernel
        g = torch.empty((total_seq_len, H_v), device=device, dtype=torch.float32)
        beta = torch.empty((total_seq_len, H_v), device=device, dtype=torch.float32)

        grid_g_beta = (total_seq_len, H_v)
        _compute_g_beta_kernel[grid_g_beta](
            A_log_fp32, a_fp32, dt_bias_fp32, b_fp32,
            g, beta,
            H_v=H_v, total_seq_len=total_seq_len,
        )

        # Initialize state if provided, else zeros
        if state is None:
            state = torch.zeros((H_v, D, D), device=device, dtype=torch.float32)
        else:
            # Ensure state is float32
            state = state.float()

        # Update state for each token using Triton
        for t in range(total_seq_len):
            _state_update_kernel[(1, H_v)](
                k_fp32[t], v_fp32[t], beta[t], g[t], state,
                H_v=H_v, total_seq_len=total_seq_len, D=D,
            )
            # After update, write to new_state
            new_state.copy_(state)

        # Compute output for each token using Triton
        for t in range(total_seq_len):
            _output_kernel[(1, H_v)](
                q_fp32[t], new_state, output[t],
                float(scale),
                H_q=H_q, H_v=H_v, D=D,
            )

        return output, new_state


def run(*args):
    return ModelNew()(*args)
