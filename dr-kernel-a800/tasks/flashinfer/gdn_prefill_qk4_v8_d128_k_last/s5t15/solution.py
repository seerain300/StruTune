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
    a_val = tl.load(a_ptr + t * HV + hv)  # bfloat16
    dt_bias_val = tl.load(dt_bias_ptr + hv)  # float32
    A_log_val = tl.load(A_log_ptr + hv)  # float32

    x_val = a_val.to(tl.float32) + dt_bias_val
    sp = tl.log(1.0 + tl.exp(x_val))  # softplus
    g_val = tl.exp(-tl.exp(A_log_val) * sp)
    tl.store(g_ptr + t * HV + hv, g_val)

    b_val = tl.load(b_ptr + t * HV + hv)  # bfloat16
    beta_val = 1.0 / (1.0 + tl.exp(-b_val.to(tl.float32)))
    tl.store(beta_ptr + t * HV + hv, beta_val)


# Triton kernel: repeat-interleave q and k along head dimension (factor = Hv // H)
# Input q_ptr/k_ptr: [T, H, K], bfloat16 (H=4, K=128 in the reference)
# Output out_ptr: [T, Hv, K], bfloat16 (Hv=8 in the reference)
@triton.jit
def _repeat_interleave_qk_kernel(
    q_ptr, k_ptr, out_ptr,
    T: tl.int32, H: tl.int32, K: tl.int32, factor: tl.int32
):
    pid_t = tl.program_id(0)  # over T
    pid_hv = tl.program_id(1)  # over H * factor
    if pid_t >= T or pid_hv >= (H * factor):
        return
    hv = pid_hv % factor
    h = pid_hv // factor
    in_index = pid_t * (H * K) + h * K + tl.arange(0, K)
    out_index = pid_t * (factor * K) + pid_hv * K + tl.arange(0, K)
    val = tl.load(q_ptr + in_index)
    tl.store(out_ptr + out_index, val)


# Triton kernel: compute per (t, v) output vector:
#   output[t, v, :] = scale * q_exp[t, v, :] @ state_new[:, v, :]
#   q_exp is [T, Hv, K], state_new is [H, V, K], K=128, V=8
# Inputs:
#   q_exp_ptr: [T, Hv, K] bfloat16
#   state_new_ptr: [H, V, K] float32
#   scale: float32 scalar
# Output:
#   out_ptr: [T, V, K] float32
@triton.jit
def _compute_output_per_tv_kernel(
    q_exp_ptr, state_new_ptr, out_ptr,
    T: tl.int32, Hv: tl.int32, V: tl.int32, K: tl.int32, scale: tl.float32
):
    pid_t = tl.program_id(0)  # over T
    pid_v = tl.program_id(1)  # over V
    if pid_t >= T or pid_v >= V:
        return
    for kk in range(0, K):
        acc = tl.zeros([1], dtype=tl.float32)
        for h in range(0, 4):  # H = 4 in reference
            hv_idx = h * V + pid_v
            q_elem = tl.load(q_exp_ptr + pid_t * (Hv * K) + hv_idx * K + kk)
            state_elem = tl.load(state_new_ptr + h * (V * K) + pid_v * K + kk)
            acc += q_elem.to(tl.float32) * state_elem
        out_index = pid_t * (V * K) + pid_v * K + kk
        tl.store(out_ptr + out_index, acc[0] * scale)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Shapes: q: [T, 4, 128], k: [T, 4, 128], v: [T, 8, 128]
        T = q.shape[0]
        H = 4  # num_q_heads
        K = q.shape[2]
        Hv = v.shape[1]  # num_v_heads = 8
        V = Hv  # num_sab_heads == 8
        device = q.device

        # Ensure contiguity and dtypes
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        a = a.contiguous()
        b = b.contiguous()
        A_log = A_log.contiguous()
        dt_bias = dt_bias.contiguous()

        # Triton compute g and beta
        a_flat = a.to(torch.float32).contiguous()          # [T, H*V]
        dt_bias_vec = dt_bias.to(torch.float32).contiguous()  # [H*V]
        b_flat = b.to(torch.float32).contiguous()          # [T, H*V]
        A_log_vec = A_log.to(torch.float32).contiguous()   # [H*V]
        g = torch.empty((T, H * V), dtype=torch.float32, device=device)
        beta = torch.empty((T, H * V), dtype=torch.float32, device=device)
        grid_g_beta = (T * (H * V),)
        _compute_g_beta_kernel[grid_g_beta](
            a_flat, dt_bias_vec, A_log_vec, b_flat, g, beta, T, H * V
        )

        # Repeat-interleave q and k along head dimension to get q_exp and k_exp
        # factor = Hv // H = 2
        q_exp = torch.empty((T, Hv, K), dtype=torch.bfloat16, device=device)
        k_exp = torch.empty((T, Hv, K), dtype=torch.bfloat16, device=device)
        grid_rep = (T, Hv * H)
        _repeat_interleave_qk_kernel[grid_rep](
            q, k, q_exp, T, H, K, Hv // H
        )
        _repeat_interleave_qk_kernel[grid_rep](
            k, k, k_exp, T, H, K, Hv // H
        )

        # Compute output per (t, v) using Triton
        state_new = torch.empty((H, V, K), dtype=torch.float32, device=device)
        output = torch.empty((T, V, K), dtype=torch.float32, device=device)
        grid_out = (T, V)
        _compute_output_per_tv_kernel[grid_out](
            q_exp, state_new, output, T, Hv, V, K, scale if scale is not None else 1.0
        )

        # Return output with expected shape (T, 8, 128) in bfloat16
        return output.to(torch.bfloat16), None


def run(*args):
    return ModelNew()(*args)
