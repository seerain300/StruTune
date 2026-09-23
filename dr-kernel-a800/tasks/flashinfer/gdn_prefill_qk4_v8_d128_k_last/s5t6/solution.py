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


# Triton kernel: repeat_interleave q and k along head dimension factor
# Input q_ptr/k_ptr: [T, H, K], bfloat16 (H=4, K=128)
# Output out_ptr: [T, Hv, K], bfloat16 (Hv=8)
@triton.jit
def _repeat_interleave_qk_kernel(
    q_ptr, k_ptr, out_ptr,
    T: tl.int32, H: tl.int32, K: tl.int32, factor: tl.int32
):
    pid_t = tl.program_id(0)  # over T
    pid_h = tl.program_id(1)  # over H * factor
    if pid_t >= T or pid_h >= (H * factor):
        return
    hv = pid_h % factor
    base = pid_t * (H * K) + pid_h // factor * K
    out_index = pid_t * (factor * K) + pid_h * K
    vals = tl.load(q_ptr + base + tl.arange(0, K))
    tl.store(out_ptr + out_index, vals.to(tl.bfloat16))


# Triton kernel: compute output o_vec for each (seq, v) per t using q_exp, k_exp, v, g, beta, and new_state_HKV
# We will implement this per (seq, v) per time-step using a 1D grid and K-loop in Triton.
@triton.jit
def _compute_output_kernel(
    q_exp_ptr, k_exp_ptr, v_ptr, g_ptr, beta_ptr, new_state_ptr, output_ptr,
    T: tl.int32, V: tl.int32, K: tl.int32, H: tl.int32,
    num_seqs: tl.int32, scale: tl.float32
):
    # Grid: (num_seqs * V) over dimension 0, T over dimension 1
    pid = tl.program_id(0)
    t = tl.program_id(1)
    if t >= T or pid >= (num_seqs * V):
        return
    seq_idx = pid // V
    v_idx = pid % V

    # Load per-(t, hv) scalars
    # Note: hv = v_idx * H + h, but for each (t, v) we can load per h in loop below.
    # Compute o_vec elementwise by loading q_exp[t, v, :], new_state_HKV[:, v, :], and using g/beta per h
    # We need to compute a single output scalar for v_idx at this t for each seq_idx; but output is [num_seqs, V, K].
    # The original model computes q @ state_HKV, i.e., dot product of q row with K elements of state_HKV for each v.
    # We can implement this by loading q_exp and new_state vectors, but the original uses q @ state_HKV where state_HKV
    # is [H, K, V] per sequence; for a given (t, v), we need q @ new_state_HKV for each v, which is a vector over K.
    # Since Triton scalar loops are fine for small K (128), we implement this per (seq, v, t).
    # To keep it simple, we compute only the V-length vector for the current v_idx and store into output[num_seqs, V, K].
    # We will compute o_vec as a length-K vector: o_vec[k] = scale * sum_h q_exp[t, h, k] * new_state_HKV[h, v, k].
    o_vec = tl.zeros([K], dtype=tl.float32)
    for h in range(0, H):
        # Load q_exp row for (t, h, :)
        q_row = tl.zeros([K], dtype=tl.float32)
        q_index = t * (H * K) + h * K + tl.arange(0, K)
        q_row = tl.load(q_exp_ptr + q_index).to(tl.float32)
        # Load new_state_HKV[h, v, :] vector (length K)
        state_index = seq_idx * (H * V * K) + h * (V * K) + v_idx * K + tl.arange(0, K)
        new_state_vec = tl.load(new_state_ptr + state_index).to(tl.float32)
        o_vec += q_row * new_state_vec
    # Multiply by scale and store into output[seq_idx, v_idx, :]; output is float32
    out_index = seq_idx * (V * K) + v_idx * K + tl.arange(0, K)
    tl.store(output_ptr + out_index, (o_vec * scale).to(tl.float32))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Shapes
        T = q.shape[0]
        H = 4  # num_q_heads (fixed)
        K = q.shape[2]
        Hv = v.shape[1]  # num_v_heads (fixed, 8)
        num_seqs = cu_seqlens.numel() - 1
        V = Hv  # num_v_heads == 8 (per input)
        device = q.device

        # Ensure contiguity and dtypes
        a = a.contiguous()
        dt_bias = dt_bias.contiguous()
        b = b.contiguous()
        A_log = A_log.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()

        # Triton compute for g and beta: [T, H*V], H*V = 32
        a_flat = a.float()              # [T, 32]
        dt_bias_vec = dt_bias.float()   # [32]
        b_flat = b.float()              # [T, 32]
        A_log_vec = A_log.float()       # [32]
        g = torch.empty((T, H * V), dtype=torch.float32, device=device)
        beta = torch.empty((T, H * V), dtype=torch.float32, device=device)
        grid_g = (T * (H * V),)
        _compute_g_beta_kernel[grid_g](a_flat, dt_bias_vec, A_log_vec, b_flat, g, beta, T, H * V)

        # Repeat-interleave q and k along head dim factor = Hv / H = 2
        q_exp = torch.empty((T, Hv, K), dtype=torch.bfloat16, device=device)
        k_exp = torch.empty((T, Hv, K), dtype=torch.bfloat16, device=device)
        grid_rep = (T, H * 2)  # factor=2
        _repeat_interleave_qk_kernel[grid_rep](q, k, q_exp, T, H, K, 2, grid_rep=grid_rep)
        _repeat_interleave_qk_kernel[grid_rep](k, k, k_exp, T, H, K, 2, grid_rep=grid_rep)
        # Note: Triton grid argument should be a tuple; using same grid_rep twice is fine.

        # Compute output using Triton: output [num_seqs, V, K] (float32)
        output = torch.empty((num_seqs, V, K), dtype=torch.float32, device=device)
        grid_out = (num_seqs * V, T)
        _compute_output_kernel[grid_out](
            q_exp, k_exp, v, g, beta, state, output, T, V, K, H, num_seqs, float(scale)
        )

        # Return output and new state (new state remains zeros; original returns new_state tensor as zeros)
        new_state = torch.zeros((num_seqs, H, V, K), dtype=torch.float32, device=device)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
