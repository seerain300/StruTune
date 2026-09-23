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
    pid = tl.program_id(0)  # program id over T * HV
    t = pid // HV
    hv = pid % HV
    if t >= T or hv >= HV:
        return
    a_val = tl.load(a_ptr + t * HV + hv)
    dt_bias_val = tl.load(dt_bias_ptr + hv)
    A_log_val = tl.load(A_log_ptr + hv)

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
    pid_t = tl.program_id(0)  # over T
    pid_h = tl.program_id(1)  # over H * factor
    if pid_t >= T or pid_h >= (H * factor):
        return
    hv = pid_h % factor
    # Compute base indices
    base = pid_t * (H * K) + pid_h // factor * K
    out_index = pid_t * (Hv * K) + hv * K
    q_val = tl.load(q_ptr + base)
    tl.store(out_ptr + out_index, q_val.to(tl.bfloat16))


# Triton kernel: per-(seq, t) linear for o = scale * q_exp @ state_new
# q_exp_ptr: [H, V*K] float32 (row-major: H*rows = H*V*K)
# state_new_ptr: [H, V*K] float32
# output_ptr: [V*K] float32 (we'll cast to bfloat16 on host)
# Grid: (1, 1) specialized per call; we launch once per (seq, t) via host loop.
@triton.jit
def _linear_bf16_kernel(
    q_exp_ptr, state_new_ptr, output_ptr,
    H: tl.constexpr, V: tl.constexpr, K: tl.constexpr, scale: tl.float32
):
    # This kernel computes o = scale * q_exp @ state_new for all columns (V*K).
    # We use a simple vectorized dot per column: for col in range(V*K), o[col] = sum_h q_exp[h, col] * state_new[h, col].
    # Launch with grid=(1, 1) to keep it simple; host will call once per (seq, t).
    for col in range(0, V * K):
        dot = tl.zeros((), dtype=tl.float32)
        # Reduce over H dimension
        for h in range(0, H):
            qh = tl.load(q_exp_ptr + h * (V * K) + col)
            st = tl.load(state_new_ptr + h * (V * K) + col)
            dot += qh * st
        o_elem = dot * scale
        tl.store(output_ptr + col, o_elem)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Shapes
        T = q.shape[0]
        H = 4


def run(*args):
    return ModelNew()(*args)
