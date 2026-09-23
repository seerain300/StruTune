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
    # Each program handles one hv (0..H_v-1)
    pid_hv = tl.program_id(0)
    if pid_hv >= H_v:
        return

    # Preload dt_bias for this hv
    dt = tl.load(dt_bias_ptr + pid_hv)

    # Loop over tokens b in [0, total_seq_len) and compute g[b, pid_hv], beta[b, pid_hv]
    for b in range(0, total_seq_len):
        # Load a[b, pid_hv] and b[b, pid_hv]
        a_val = tl.load(a_ptr + b * H_v + pid_hv)
        bb_val = tl.load(b_ptr + b * H_v + pid_hv)

        # x = a + dt
        x = a_val + dt
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(x))
        # Load A_log[pid_hv]
        A_val = tl.load(A_log_ptr + pid_hv)
        # g = exp(-exp(A) * softplus(x))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        # beta = sigmoid(b) = 1 / (1 + exp(-b))
        beta_val = 1.0 / (1.0 + tl.exp(-bb_val))

        # Store results
        tl.store(g_ptr + b * H_v + pid_hv, g_val)
        tl.store(beta_ptr + b * H_v + pid_hv, beta_val)


@triton.jit
def _state_update_kernel(
    k_ptr,       # [total_seq_len, H_v, D] float32
    v_ptr,       # [total_seq_len, H_v, D] float32
    state_ptr,   # [H_v, D, D] float32 (contiguous, strides not needed if flattened)
    g_ptr,       # [total_seq_len, H_v] float32
    beta_ptr,    # [total_seq_len, H_v] float32
    new_state_ptr,  # [H_v, D, D] float32 (output updated state)
    total_seq_len: tl.constexpr,
    H_v: tl.constexpr,
    D: tl.constexpr,
):
    # One program per (t, hv). We use program_id(0) to iterate over t and pid_hv over hv (but grid is 1D so we encode both).
    # Launch as grid (total_seq_len, H_v) for clarity.
    pid_t = tl.program_id(0)
    pid_hv = tl.program_id(1)
    if pid_t >= total_seq_len or pid_hv >= H_v:
        return

    # Load k[t, pid_hv, :] and v[t, pid_hv, :]
    k_vec = tl.load(k_ptr + pid_t * (H_v * D) + pid_hv * D + tl.arange(0, D))  # [D]
    v_vec = tl.load(v_ptr + pid_t * (H_v * D) + pid_hv * D + tl.arange(0, D))  # [D]

    # Load old state and g, beta for this (t, hv)
    old_state = tl.load(state_ptr + pid_hv * (D * D) + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :])  # [D, D]
    g_val = tl.load(g_ptr + pid_t * H_v + pid_hv)
    beta_val = tl.load(beta_ptr + pid_t * H_v + pid_hv)

    # old_v = k @ state_old
    old_v = tl.sum(k_vec[:, None] * old_state, axis=0)  # [D]

    # new_v = beta * v + (1 - beta) * old_v
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [D]

    # Compute k^T @ old_v and k^T @ new_v as scalars (sum of k over D)
    k_sum = tl.sum(k_vec, axis=0)
    kT_old = tl.sum(k_vec * old_v, axis=0)  # sum_i k[i] * old_v[i]
    kT_newv = tl.sum(k_vec * new_v, axis=0)

    # new_state = g * state_old - k^T @ old_v + k^T @ new_v
    new_state_mat = g_val * old_state - kT_old + kT_newv

    # Store updated state
    tl.store(new_state_ptr + pid_hv * (D * D) + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :], new_state_mat)


@triton.jit
def _output_kernel(
    q_ptr,       # [total_seq_len, H_q, D] float32
    state_ptr,   # [H_v, D, D] float32
    out_ptr,     # [total_seq_len, H_v, D] float32
    scale,       # float32
    H_q: tl.constexpr,
    H_v: tl.constexpr,
    D: tl.constexpr,
):
    # Grid: (total_seq_len, H_v). Each program handles one (t, hv).
    pid_t = tl.program_id(0)
    pid_hv = tl.program_id(1)
    if pid_t >= total_seq_len or pid_hv >= H_v:
        return

    # Form q_exp vector for this (t, hv): original code repeats q along head dimension. Here H_v//H_q == 2 for H_v=8, H_q=4.
    # We need q_exp_vec of length D*2. We'll load q[t, h, :] for h in [0..H_q-1], and duplicate into q_exp_vec.
    q_exp_vec = tl.zeros((D * 2,), dtype=tl.float32)
    for h in range(H_q):
        q_vec = tl.load(q_ptr + pid_t * (H_q * D) + h * D + tl.arange(0, D))  # [D]
        q_exp_vec[h * D : (h + 1) * D] = q_vec
        q_exp_vec[(h + H_q) * D : (h + H_q + 1) * D] = q_vec

    # Compute output vector: scale * q_exp_vec @ state[hv, :, :]
    state_mat = tl.load(state_ptr + pid_hv * (D * D) + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :])  # [D, D]
    out_vec = tl.sum(q_exp_vec[:, None] * state_mat, axis=0)  # [D]
    out_vec = out_vec * scale

    # Store to out[t, hv, :]
    tl.store(out_ptr + pid_t * (H_v * D) + pid_hv * D + tl.arange(0, D), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Cast inputs to float32 for numerical stability and Triton support
        device = q.device
        total_seq_len, H_q, D = q.shape
        H_v = v.shape[1]
        assert H_q == 4 and H_v == 8 and D == 128, "This Triton implementation assumes H_q=4, H_v=8, D=128"

        q_f = q.float().contiguous()
        k_f = k.float().contiguous()
        v_f = v.float().contiguous()

        # Prepare buffers
        g = torch.empty((total_seq_len, H_v), dtype=torch.float32, device=device)
        beta = torch.empty((total_seq_len, H_v), dtype=torch.float32, device=device)

        # Launch gate computation kernel: one program per hv
        _compute_g_beta_kernel[(H_v,)](A_log.float(), a.float(), dt_bias.float(), b.float(), g, beta, H_v=H_v, total_seq_len=total_seq_len)

        # new_state: [H_v, D, D], initialize zeros or use provided state (float32)
        new_state = torch.zeros((H_v, D, D), dtype=torch.float32, device=device)
        if state is not None:
            new_state.copy_(state.float().contiguous())

        # Update state per token using Triton kernel: grid (total_seq_len, H_v)
        _state_update_kernel[(total_seq_len, H_v)](
            k_f, v_f, new_state, g, beta, new_state, total_seq_len=total_seq_len, H_v=H_v, D=D
        )

        # Output: allocate [total_seq_len, H_v, D] float32 and fill via Triton kernel
        out = torch.empty((total_seq_len, H_v, D), dtype=torch.float32, device=device)

        # Scale: use provided or default
        scale_val = float(scale) if scale is not None else 1.0 / math.sqrt(D)

        # Launch output kernel: grid (total_seq_len, H_v)
        _output_kernel[(total_seq_len, H_v)](
            q_f, new_state, out, scale_val, H_q=H_q, H_v=H_v, D=D
        )

        # Return output as bfloat16 and updated state (converted to float32 as in original). The original returns (output, new_state).
        out_bf16 = out.to(torch.bfloat16)
        new_state_f32 = new_state  # keep float32 as original uses float32 for state
        return out_bf16, new_state_f32


def run(*args):
    return ModelNew()(*args)
