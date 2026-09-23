import math
import torch
import triton
import triton.language as tl


# Kernel: compute g = exp(-exp(A_log[h]) * softplus(x)), where x = a[b,h] + dt_bias[h]
# softplus(x) = log(1 + exp(x))
@triton.jit
def softplus_and_exp_kernel(
    dt_bias_ptr,    # float32 [H]
    a_ptr,          # float32 [B*H]
    A_log_ptr,      # float32 [H]
    g_out_ptr,      # float32 [B*H]
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    h = pid % H
    a_val = tl.load(a_ptr + pid)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val
    s = tl.log(1.0 + tl.exp(x))
    e = tl.exp(tl.load(A_log_ptr + h))
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Kernel: compute beta = sigmoid(b[b,h])
@triton.jit
def sigmoid_kernel(
    b_ptr,          # float32 [B*H]
    beta_out_ptr,   # float32 [B*H]
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Kernel: scalar dot old_v = k_vec @ state_mat, state_mat flattened [V*K]
@triton.jit
def dot_k_state_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    old_v_ptr,      # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j
    tl.store(old_v_ptr, acc)


# Kernel: scalar dot state_remove = k_vec @ (g * state_mat)
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    g_ptr,          # float32 [1] (scalar g for this b,h)
    state_remove_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    g_val = tl.load(g_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j * g_val
    tl.store(state_remove_ptr, acc)


# Kernel: scalar dot state_update = k_vec @ (beta * v_vec + (1 - beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    old_v_ptr,      # float32 [1] (old_v)
    beta_ptr,       # float32 [1] (beta scalar for this b,h)
    state_update_ptr,   # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    beta_val = tl.load(beta_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (beta_val * v_i + (1.0 - beta_val) * tl.load(old_v_ptr))
    tl.store(state_update_ptr, acc)


# Kernel: vector h_state_vec[i] = sum_j state[b,h,i,j] * g - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_ptr,          # float32 [1] scalar g for this b,h
    state_remove_ptr,   # float32 [1]
    state_update_ptr,   # float32 [1]
    h_state_ptr,         # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    g_val = tl.load(g_ptr)
    state_remove_val = tl.load(state_remove_ptr)
    state_update_val = tl.load(state_update_ptr)
    for i in range(V):
        total = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            total += state_ij * g_val
        h_state_vec_i = total - state_remove_val + state_update_val
        tl.store(h_state_ptr + i, h_state_vec_i)


# Kernel: scalar output[b,h] = scale * q_vec @ h_state_vec
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    output_ptr,     # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        q_j = tl.load(q_ptr + j)
        for i in range(V):
            h_i = tl.load(h_state_ptr + i)
            acc += q_j * h_i
    tl.store(output_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # No torch ops in __init__; capture args if needed (not used in forward)

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Inputs shapes: q: [B,1,4,128], k: [B,1,4,128], v: [B,1,8,128], state: [B,8,128,128]
        device = q.device
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v


def run(*args):
    return ModelNew()(*args)
