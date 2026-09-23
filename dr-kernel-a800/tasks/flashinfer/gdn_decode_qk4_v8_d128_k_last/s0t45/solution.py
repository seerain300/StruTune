import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    B, H,
    stride_A, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
    stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load parameters
    A = tl.load(A_log_ptr + h_idx * stride_A).to(tl.float32)  # A_log[h]
    a = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h).to(tl.float32)  # a[b, h]
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt).to(tl.float32)  # dt_bias[h]
    bb = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32)  # b[b, h]
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a + dt))
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))
    # Store
    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_tmp_b, stride_tmp_h,
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # k[b, h] is [K]
    k_offs = tl.arange(0, K)
    k_vals = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_offs * stride_k_k)
    # state[b, h] is [V, K]
    v = tl.arange(0, V)
    k_dim = tl.arange(0, K)
    s_ptrs = state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v[:, None] * stride_s_v + k_dim[None, :] * stride_s_k
    mask = (v[:, None] < V) & (k_dim[None, :] < K)
    state_block = tl.load(s_ptrs, mask=mask, other=0.0)
    # tmp_old_v[b, h] = sum_k k_vals[k] * state_block[k, :]
    tmp_vec = state_block * k_vals[None, :]
    tmp_val = tl.sum(tmp_vec, axis=1)  # sum over K for each v
    tmp_val = tl.sum(tmp_val, axis=0)  # sum over V to scalar
    tl.store(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h, tmp_val)


@triton.jit
def kernel_output_reduce(
    q_ptr, new_state_ptr, output_ptr,
    B, H, V, K,
    stride_q_b, stride_q_h, stride_q_k,
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
    stride_out_b, stride_out_h,
    scale,  # float32 scalar
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h) and computes scalar output
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # q[b, h] is [K]
    q_offs = tl.arange(0, K)
    q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + q_offs * stride_q_k)
    # new_state[b, h] is [V, K]
    v = tl.arange(0, V)
    k_dim = tl.arange(0, K)
    ns_ptrs = new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v[:, None] * stride_ns_v + k_dim[None, :] * stride_ns_k
    mask = (v[:, None] < V) & (k_dim[None, :] < K)
    new_state_block = tl.load(ns_ptrs, mask=mask, other=0.0)
    # output[b, h] = scale * sum_{i} q[i] * new_state[i, :]
    prod = new_state_block * q_vec[None, :]
    prod_sum = tl.sum(prod, axis=1)  # sum over K for each v
    out_val = tl.sum(prod_sum, axis=0)  # sum over V to scalar
    out_val = scale * out_val
    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_h, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure inputs are contiguous and float32
        device = q.device
        B


def run(*args):
    return ModelNew()(*args)
