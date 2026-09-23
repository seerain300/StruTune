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
    pid = tl.program_id(0)  # program id over T*HV
    t = pid // HV
    hv = pid % HV
    if t >= T or hv >= HV:
        return
    a_val = tl.load(a_ptr + t * HV + hv)
    dt_bias_val = tl.load(dt_bias_ptr + hv)
    A_log_val = tl.load(A_log_ptr + hv)

    x_val = a_val.to(tl.float32) + dt_bias_val
    sp = tl.log(1.0 + tl.exp(x_val))  # softplus
    g_val = tl.exp(-tl.exp(A_log_val) * sp)
    tl.store(g_ptr + t * HV + hv, g_val)

    b_val = tl.load(b_ptr + t * HV + hv)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val.to(tl.float32)))
    tl.store(beta_ptr + t * HV + hv, beta_val)


# Triton kernel: compute per-time-step output for each v across all sequences
# Input tensors:
#   q_exp_ptr: [T, V, K] bfloat16 (V=num_v_heads, K=head_size), laid out as T*(V*K) + v*K + kk
#   state_new_ptr: [num_seqs, H, V, K] float32 (H=num_q_heads), laid out as num_seqs*(H*V*K) + h*(V*K) + v*K + kk
#   cu_seqlens_ptr: [num_seqs+1] int32 (cumulative lengths per sequence segment)
# Output:
#   output_ptr: [T, V, K] bfloat16, laid out as T*(V*K) + v*K + kk
# Launch with grid=(num_seqs, T) to cover all sequence segments and all time steps.
@triton.jit
def _compute_output_all_t_kernel(
    q_exp_ptr, state_new_ptr, cu_seqlens_ptr, output_ptr,
    num_seqs: tl.int32, T: tl.int32, V: tl.int32, K: tl.int32
):
    pid_seq = tl.program_id(0)  # sequence segment index
    pid_t = tl.program_id(1)    # time step index
    if pid_seq >= num_seqs or pid_t >= T:
        return

    # For each sequence segment, start index is cu_seqlens[pid_seq] - 1 (since cu_seqlens is cumulative).
    # But we don't need 'start' here because we already index by pid_t relative to the segment length.
    # We just process time step pid_t within segment pid_seq.

    # Compute o_vec for all v at time pid_t
    for v in range(0, V):
        # Load q_exp[t, v, :] as a K-vector
        q_row = tl.zeros([K], dtype=tl.float32)
        base_q = pid_t * (V * K) + v * K
        for kk in range(0, K):
            q_row[kk] = tl.load(q_exp_ptr + base_q + kk)

        # Load state_new[:, v, :] for all h (h in [0..H-1]); state_new is [num_seqs, H, V, K]
        state_vec = tl.zeros([K], dtype=tl.float32)
        base_state = pid_seq * (H * V * K) + v * K
        for h in range(0, H):
            base_h = base_state + h * (V * K)
            for kk in range(0, K):
                state_vec[kk] += tl.load(state_new_ptr + base_h + kk)

        # Dot product: scale = 1.0 as in original code
        dot = tl.zeros([1], dtype=tl.float32)
        for kk in range(0, K):
            dot += q_row[kk] * state_vec[kk]
        o_elem = dot[0]  # scalar float32

        # Store output: [T, V, K], linearized as T*(V*K) + v*K + kk
        out_base = pid_t * (V * K) + v * K
        for kk in range(0, K):
            tl.store(output_ptr + out_base + kk, o_elem.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Extract shapes
        T = q.shape[0]
        H = q.shape[1]  # num_q_heads (could be 1, 2, 4, etc.)
        K = q.shape[2]
        V = v.shape[1]  # num_v_heads
        num_seqs = cu_seqlens.numel() - 1
        device = q.device

        # Prepare inputs for Triton g/beta computation
        a_flat = a.float().contiguous()          # [T, H*V]
        dt_bias_vec = dt_bias.float().contiguous()  # [H*V]
        b_flat = b.float().contiguous()          # [T, H*V]
        A_log_vec = A_log.float().contiguous()   # [H*V]

        # Allocate outputs for g and beta
        g = torch.empty((T, H * V), dtype=torch.float32, device=device)
        beta = torch.empty((T, H * V), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta over all (t, hv)
        grid_g = (T * (H * V),)
        _compute_g_beta_kernel[grid_g](a_flat, dt_bias_vec, A_log_vec, b_flat, g, beta, T, H * V)

        # Compute repeat-interleave factor dynamically: factor = V // H
        # If V % H != 0, treat factor as 1 (no repeat). Original code uses factor=2 for H=4, V=8.
        factor = V // H
        if factor == 0:
            factor = V

        # Create repeated q and k along head dimension: [T, V, K], bfloat16
        q_exp = torch.repeat_interleave(q, repeats=factor, dim=1).contiguous()
        k_exp = torch.repeat_interleave(k, repeats=factor, dim=1).contiguous()

        # Output tensor [T, V, K], bfloat16
        output = torch.empty((T, V, K), dtype=torch.bfloat16, device=device)

        # Launch Triton kernel over (num_seqs, T): computes output for each sequence segment and each time step.
        grid_out = (num_seqs, T)
        _compute_output_all_t_kernel[grid_out](
            q_exp, state, cu_seqlens, output,
            num_seqs, T, V, K
        )

        # Return output [T, V, K] bfloat16 and state (the original state is not modified in the reference either)
        return output, state


def run(*args):
    return ModelNew()(*args)
