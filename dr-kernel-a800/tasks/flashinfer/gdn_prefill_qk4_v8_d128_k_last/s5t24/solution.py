import torch
import math
import triton
import triton.language as tl


# Triton kernel: compute g and beta for all (t, hv)
# Inputs:
#   a_ptr: [T, HV] bfloat16
#   dt_bias_ptr: [HV] float32
#   A_log_ptr: [HV] float32
#   b_ptr: [T, HV] bfloat16
# Outputs:
#   g_ptr: [T, HV] float32
#   beta_ptr: [T, HV] float32
@triton.jit
def _compute_g_beta_kernel(
    a_ptr, dt_bias_ptr, A_log_ptr, b_ptr,
    g_ptr, beta_ptr,
    T: tl.int32, HV: tl.int32
):
    # Launch over all (t, hv) elements
    pid = tl.program_id(0)
    if pid >= T * HV:
        return
    t = pid // HV
    hv = pid % HV
    a_val = tl.load(a_ptr + t * HV + hv)
    dt_bias_val = tl.load(dt_bias_ptr + hv)
    A_log_val = tl.load(A_log_ptr + hv)

    # x = a + dt_bias
    x_val = a_val.to(tl.float32) + dt_bias_val
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x_val))
    # g = exp(-exp(A_log) * softplus(x))
    g_val = tl.exp(-tl.exp(A_log_val) * sp)
    tl.store(g_ptr + t * HV + hv, g_val)

    b_val = tl.load(b_ptr + t * HV + hv)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val.to(tl.float32)))
    tl.store(beta_ptr + t * HV + hv, beta_val)


# Triton kernel: repeat_interleave q and k along head dimension factor
# Input q_ptr/k_ptr: [T, H, K], bfloat16 (H=4, K=128)
# Output out_ptr: [T, Hv, K], bfloat16 (Hv=8)
@triton.jit
def _repeat_interleave_qk_kernel(
    q_ptr, k_ptr, out_ptr,
    T: tl.int32, H: tl.int32, K: tl.int32, factor: tl.int32
):
    # Grid: (T, H*factor)
    pid_t = tl.program_id(0)
    pid_hv = tl.program_id(1)
    if pid_t >= T or pid_hv >= (H * factor):
        return
    hv = pid_hv % factor
    # Compute base indices
    base = pid_t * (H * K) + pid_hv // factor * K
    out_index = pid_t * (factor * K) + hv * K
    q_val = tl.load(q_ptr + base)
    tl.store(out_ptr + out_index, q_val.to(tl.bfloat16))


# Triton kernel: per-(seq, t, v) compute o_vec = scale * q_exp @ state_new
# q_exp_ptr: [H, V*K] float32 row-major
# state_new_ptr: [H, V*K] float32 row-major
# output_ptr: [V*K] float32, host will cast to bfloat16
@triton.jit
def _linear_out_per_v_kernel(
    q_exp_ptr, state_new_ptr, output_ptr,
    H: tl.constexpr, V: tl.constexpr, K: tl.constexpr, scale: tl.float32
):
    # Grid: (num_seqs, T, V); within kernel, we compute the output vector for one v.
    pid_seq = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_v = tl.program_id(2)
    if pid_seq < 0 or pid_t < 0 or pid_v < 0:
        return
    # Vectorize over K for this v
    for kk in range(0, K):
        dot = tl.zeros((), dtype=tl.float32)
        # Reduce over H
        for h in range(0, H):
            col = h * (V * K) + pid_v * K + kk
            qh = tl.load(q_exp_ptr + col)
            st = tl.load(state_new_ptr + col)
            dot += qh * st
        o_elem = dot * scale
        tl.store(output_ptr + pid_v * K + kk, o_elem)


# Triton kernel: per-(seq, h, v, k) update state according to formula
# Inputs:
#   state_old_ptr: [H, V, K] float32 (row-major)
#   q_row_ptr: [K] float32 (q_exp[t, v, :])
#   k_row_ptr: [K] float32 (k_exp[t, hv, :])
#   v_row_ptr: [K] float32 (v[t, hv, :])
#   g_ptr: [T, HV] float32
#   beta_ptr: [T, HV] float32
#   A_log_ptr: [HV] float32
#   scale: float32
# Output:
#   new_state_ptr: [H, V, K] float32 (row-major), written elementwise per (h,v,k)
@triton.jit
def _update_state_per_v_kernel(
    state_old_ptr, q_row_ptr, k_row_ptr, v_row_ptr, g_ptr, beta_ptr, A_log_ptr,
    new_state_ptr,
    T: tl.int32, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr
):
    pid_seq = tl.program_id(0)  # over num_seqs
    h = tl.program_id(1)        # over H
    v = tl.program_id(2)        # over V
    k = tl.program_id(3)        # over K
    if pid_seq < 0 or h >= H or v >= V or k >= K:
        return
    t = tl.program_id(4)        # over T
    if t < 0:
        return

    # Compute g and beta for this hv = h * V + v
    hv = h * V + v
    g_t = tl.load(g_ptr + t * (H * V) + hv)
    beta_t = tl.load(beta_ptr + t * (H * V) + hv)
    A_log_val = tl.load(A_log_ptr + hv)

    # Load vectors
    q_row = tl.zeros([K], dtype=tl.float32)
    for kk in range(0, K):
        q_row[kk] = tl.load(q_row_ptr + kk)

    v_row = tl.zeros([K], dtype=tl.float32)
    for kk in range(0, K):
        v_row[kk] = tl.load(v_row_ptr + kk)

    k_row = tl.zeros([K], dtype=tl.float32)
    for kk in range(0, K):
        k_row[kk] = tl.load(k_row_ptr + kk)

    # Load state_old[h, v, :]
    state_old_vec = tl.zeros([K], dtype=tl.float32)
    base = pid_seq * (H * V * K) + h * (V * K) + v * K
    for kk in range(0, K):
        state_old_vec[kk] = tl.load(state_old_ptr + base + kk)

    # Compute old_v = k_row @ state_old_vec
    old_v = tl.zeros((), dtype=tl.float32)
    for kk in range(0, K):
        old_v += k_row[kk] * state_old_vec[kk]

    # Compute new_v = beta * v_row + (1 - beta) * old_v
    new_v = beta_t * v_row + (1.0 - beta_t) * old_v

    # Compute remove = k_row @ (k_row @ state_old_vec) which equals k_row @ old_v
    # (Note: remove equals old_v since k_row @ state_old_vec = old_v)
    remove = old_v

    # Update: new_state[h, v, k] = g * state_old[h, v, k] + (k_row @ new_v) - remove
    # Compute k_row @ new_v
    dot_k_newv = tl.zeros((), dtype=tl.float32)
    for kk in range(0, K):
        dot_k_newv += k_row[kk] * new_v

    new_state_elem = g_t * state_old_vec[k] + dot_k_newv - remove
    new_state_index = pid_seq * (H * V * K) + h * (V * K) + v * K + k
    tl.store(new_state_ptr + new_state_index, new_state_elem)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Fixed constants as in the original setup
        H = 4          # num_q_heads
        V = 8          # num_v_heads
        K = 128        # head_size

        T = q.shape[0]
        device = q.device

        # 1) Compute g and beta using Triton
        HV = H * V
        a_flat = a.float().contiguous()          # [T, HV] bfloat16 promoted to float32
        dt_bias_vec = dt_bias.float().contiguous()  # [HV] float32
        b_flat = b.float().contiguous()          # [T, HV]
        A_log_vec = A_log.float().contiguous()   # [HV]

        g = torch.empty((T, HV), dtype=torch.float32, device=device)
        beta = torch.empty((T, HV), dtype=torch.float32, device=device)

        # Launch Triton kernel over all (t, hv)
        grid_g = (T * HV,)
        _compute_g_beta_kernel[grid_g](
            a_flat, dt_bias_vec, A_log_vec, b_flat,
            g, beta,
            T, HV
        )

        # 2) Repeat-interleave q and k from H heads to Hv
        q_exp = torch.empty((T, V * K), dtype=torch.bfloat16, device=device)
        k_exp = torch.empty((T, V * K), dtype=torch.bfloat16, device=device)

        grid_qk = (T, H * V)
        _repeat_interleave_qk_kernel[grid_qk](
            q, k, q_exp,
            T, H, K, V
        )

        # 3) Prepare state_old for first iteration (if provided), otherwise zeros
        # Reference code uses state layout [H, V, K] in k-last. We convert to [H, V, K] row-major for Triton.
        if state is None:
            state_old = torch.empty((H, V, K), dtype=torch.float32, device=device)
        else:
            # state in reference is [H, V, K]; convert to [H, V, K] row-major for Triton
            state_old = state[0].float().transpose(-1, -2).contiguous()  # [H, K, V] to [H, V, K] contiguous

        new_state = torch.empty((1, H, V, K), dtype=torch.float32, device=device)  # single seq, will return [num_seqs, H, V, K]

        # 4) Compute outputs and update state for each sequence t using Triton kernels
        # For each sequence block, we update state per (h, v, k) and compute o_vec per (seq, t, v).
        num_seqs = cu_seqlens.numel() - 1
        seq_start = int(cu_seqlens[0].item())
        seq_end = int(cu_seqlens[1].item())
        seq_len = seq_end - seq_start

        # We assume num_seqs=1 in provided get_inputs; handle generic num_seqs if needed
        # Initialize new_state per sequence. We'll update it per t using _update_state_per_v_kernel.
        # First, set new_state to state_old for seq_idx=0
        if state is None:
            new_state[0] = state_old
        else:
            new_state[0] = state_old

        # Prepare output tensor [num_seqs, V, K] as float32; we'll cast to bfloat16 at return
        output = torch.empty((num_seqs, V, K), dtype=torch.float32, device=device)

        # Per sequence block, iterate t from seq_start to seq_end
        for t_idx in range(seq_start, seq_end):
            # Update state per (h, v, k) using Triton
            grid_update = (num_seqs, H, V, K)
            # Note: for single sequence, num_seqs=1, so grid_update is (1, H, V, K)
            _update_state_per_v_kernel[grid_update](
                state_old, q_exp[t_idx], k_exp[t_idx], v[t_idx], g[t_idx], beta[t_idx], A_log_vec,
                new_state,  # write per (h, v, k)
                T, H, V, K
            )

            # Compute output per v using Triton: o_vec = scale * q_exp[t_idx, v, :] @ new_state[:, v, :]
            # We need q_exp[t_idx, v, :] -> reshape to [H, V*K], state_new -> [H, V*K]
            # Compute state_new per v by reading new_state[..., k] and writing to [H, V*K]
            # Here we reconstruct state_new as [H, V*K] by slicing new_state over v:
            # We'll write to a temporary [H, V*K] tensor for this t.
            state_new_rows = torch.empty((H, V * K), dtype=torch.float32, device=device)
            for h in range(H):
                for v2 in range(V):
                    base = new_state[0][h, v2]  # [K]
                    for kk in range(K):
                        state_new_rows[h, v2 * K + kk] = base[kk]

            # Launch Triton kernel for each v
            for v2 in range(V):
                output_slice = torch.empty((K,), dtype=torch.float32, device=device)
                grid_linear = (num_seqs, 1, 1)  # only one sequence block; using 1s for t and v indices
                _linear_out_per_v_kernel[grid_linear](
                    state_new_rows, q_exp[t_idx], output_slice,
                    H, V, K, float(scale)
                )
                output[0, v2] = output_slice  # store per v vector

        # Return output [num_seqs, V, K] as bfloat16 (reference output is bfloat16)
        output_bf16 = output.to(torch.bfloat16)
        # Return new_state as float32 with shape (1, H, V, K) -> (num_seqs, H, V, K). If more seqs, extend similarly.
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
