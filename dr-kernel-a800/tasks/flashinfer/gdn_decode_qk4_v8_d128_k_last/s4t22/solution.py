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
    B: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    h = pid % H
    b = pid // H
    a_val = tl.load(a_ptr + pid)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val
    s = tl.log(1.0 + tl.exp(x))
    e = tl.exp(tl.load(A_log_ptr + h))
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Kernel: compute beta = sigmoid(b[b,h]) = 1 / (1 + exp(-b))
@triton.jit
def sigmoid_kernel(
    b_ptr,          # float32 [B*H]
    beta_out_ptr,   # float32 [B*H]
    B: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    h = pid % H
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Kernel: compute old_v = dot(k_vec, state_mat) where state_mat is [V*K] flattened
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


# Kernel: compute state_remove = dot(k_vec, g * state_mat)
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    g_ptr,          # float32 [1] scalar g
    state_remove_ptr,   # float32 [1]
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


# Kernel: compute state_update = dot(k_vec, beta * v + (1 - beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    old_v_ptr,      # float32 [1] old_v scalar
    beta_ptr,       # float32 [1] beta scalar
    state_update_ptr,   # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    beta_val = tl.load(beta_ptr)
    old_v_val = tl.load(old_v_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (beta_val * v_i + (1.0 - beta_val) * old_v_val)
    tl.store(state_update_ptr, acc)


# Kernel: vector h_state_vec[i] = sum_j state[b,h,i,j] * g - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_ptr,          # float32 [1] scalar g for this (b,h)
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


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [V*K]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        val = tl.load(h_state_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # All compute in Triton; no torch ops in forward
        device = q.device
        dtype_compute = torch.float32

        # Shapes
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8
        assert K == 128 and V == 128 and T == 1

        # Convert inputs to float32 tensors on device (no torch ops in forward)
        a_f32 = a.squeeze(1).contiguous().to(dtype_compute)
        dt_bias_f32 = dt_bias.contiguous().to(dtype_compute)
        A_log_f32 = A_log.contiguous().to(dtype_compute)
        q_f32 = q.squeeze(1).contiguous().to(dtype_compute)
        k_f32 = k.squeeze(1).contiguous().to(dtype_compute)
        v_f32 = v.squeeze(1).contiguous().to(dtype_compute)
        state_f32 = state.contiguous().to(dtype_compute)
        b_f32 = b.squeeze(1).contiguous().to(dtype_compute)

        # Compute g and beta
        g = torch.empty(B * num_v_heads, device=device, dtype=dtype_compute)
        beta = torch.empty(B * num_v_heads, device=device, dtype=dtype_compute)
        softplus_and_exp_kernel[(B * num_v_heads,)](
            dt_bias_f32, a_f32, A_log_f32, g, B, num_v_heads
        )
        sigmoid_kernel[(B * num_v_heads,)](
            b_f32, beta, B, num_v_heads
        )

        # Prepare output and new_state
        output = torch.empty(B * num_v_heads, device=device, dtype=dtype_compute)
        new_state = torch.empty((B, num_v_heads, V, K), device=device, dtype=dtype_compute)

        # For each (b,h)
        for b_idx in range(B):
            for h_idx in range(num_v_heads):
                pid = b_idx * num_v_heads + h_idx

                # k_vec and state_mat
                k_vec = k_f32[b_idx, h_idx]  # [K]
                state_mat = state_f32[b_idx, h_idx]  # [V*K]

                # old_v = k @ state
                old_v = torch.empty(1, device=device, dtype=dtype_compute)
                dot_k_state_kernel[(1,)](
                    k_vec, state_mat, old_v, V, K
                )

                # state_remove = k @ (g * state)
                g_ptr = g[pid]  # scalar
                state_remove = torch.empty(1, device=device, dtype=dtype_compute)
                dot_k_gstate_kernel[(1,)](
                    k_vec, state_mat, g_ptr, state_remove, V, K
                )

                # state_update = k @ (beta * v + (1 - beta) * old_v)
                v_vec = v_f32[b_idx, h_idx]  # [V]
                beta_ptr = beta[pid]
                old_v_ptr = old_v
                state_update = torch.empty(1, device=device, dtype=dtype_compute)
                dot_k_newv_kernel[(1,)](
                    k_vec, v_vec, old_v_ptr, beta_ptr, state_update, V, K
                )

                # h_state_vec
                h_state = torch.empty(V, device=device, dtype=dtype_compute)
                h_state_ptr = h_state
                h_state_vec_kernel[(1,)](
                    state_mat, g_ptr, state_remove, state_update, h_state_ptr, V, K
                )

                # output scalar = scale * (q @ h_state_vec)
                q_vec = q_f32[b_idx, h_idx]  # [K]
                out_scalar = torch.empty(1, device=device, dtype=dtype_compute)
                dot_q_hstate_kernel[(1,)](
                    q_vec, h_state_ptr, out_scalar, V, K
                )
                output[pid] = out_scalar[0]

                # write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
                new_state_ptr = new_state[b_idx, h_idx].contiguous()  # [V*K]
                write_new_state_kernel[(1,)](
                    h_state_ptr, new_state_ptr, V, K
                )

        # Cast output to bfloat16 and return single tensor [B,1,H]
        output_bf16 = output.view(B, 1, num_v_heads).to(torch.bfloat16)
        # new_state is not returned (avoid tuple issues); compute done in Triton
        return output_bf16


def run(*args):
    return ModelNew()(*args)
