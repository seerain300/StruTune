import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute g[b,h] = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
@triton.jit
def softplus_and_exp_kernel(
    dt_bias_ptr,    # float32 [H]
    a_ptr,          # float32 [B*H]
    A_log_ptr,      # float32 [H]
    g_out_ptr,      # float32 [B*H]
    B: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..B*H-1
    h = pid % H
    a_val = tl.load(a_ptr + pid)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val
    s = tl.log(1.0 + tl.exp(x))
    e = tl.exp(tl.load(A_log_ptr + h))
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Triton kernel: compute beta[b,h] = 1 / (1 + exp(-b[b,h]))
@triton.jit
def sigmoid_kernel(
    b_ptr,          # float32 [B*H]
    beta_out_ptr,   # float32 [B*H]
    B: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..B*H-1
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Triton kernel: compute old_v = dot(k_vec, state_mat) where state_mat is flattened [V*K]
# Inputs: k_ptr [K], state_ptr [V*K], old_v_ptr [1]
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


# Triton kernel: compute state_remove = dot(k_vec, g * state_mat)
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    g_scalar,       # float32 scalar
    state_remove_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j
    acc = acc * g_scalar
    tl.store(state_remove_ptr, acc)


# Triton kernel: compute state_update = dot(k_vec, beta * v_vec + (1 - beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [K]
    old_v_scalar,   # float32 scalar
    beta_scalar,    # float32 scalar
    state_update_ptr,   # float32 [1]
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        v_j = tl.load(v_ptr + j)
        acc += k_j * (beta_scalar * v_j + (1.0 - beta_scalar) * old_v_scalar)
    tl.store(state_update_ptr, acc)


# Triton kernel: compute h_state_vec[i] = sum_j state[i,j] * g_scalar - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_scalar,       # float32 scalar
    state_remove_scalar,  # float32 scalar
    state_update_scalar,  # float32 scalar
    h_state_ptr,    # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        acc = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * g_scalar
        acc = acc - state_remove_scalar + state_update_scalar
        tl.store(h_state_ptr + i, acc)


# Triton kernel: compute output[b,h] = scale * (q_vec @ h_state_vec)
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    hstate_ptr,     # float32 [V]
    output_ptr,     # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
    scale,          # float32 scalar
):
    acc = 0.0
    for i in range(V):
        h_i = tl.load(hstate_ptr + i)
        for j in range(K):
            q_j = tl.load(q_ptr + j)
            acc += h_i * q_j
    acc = acc * scale
    tl.store(output_ptr, acc)


# Triton kernel: write new_state[b,h] as [V,K]: broadcast h_state_vec across K
@triton.jit
def write_new_state_kernel(
    hstate_ptr,     # float32 [V]
    new_state_ptr,  # float32 [V*K]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        val = tl.load(hstate_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure inputs are on CUDA device; Triton kernels require CUDA tensors
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "All inputs must be CUDA tensors"

        # Shapes from inputs
        B, T_q, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        H = num_v_heads  # heads (e.g., 8)

        # Squeeze T=1 dimension and make contiguous; cast to float32 for compute
        q_f32 = q.squeeze(1).float().contiguous()  # [B,4,K] -> [B,K]
        k_f32 = k.squeeze(1).float().contiguous()  # [B,4,K] -> [B,K]
        v_f32 = v.squeeze(1).float().contiguous()  # [B,8,K] -> [B,K]
        a_f32 = a.float().contiguous()             # [B,H]
        dt_bias_f32 = dt_bias.float().contiguous() # [H]
        A_log_f32 = A_log.float().contiguous()     # [H]
        b_f32 = b.float().contiguous()             # [B,H]
        state_f32 = state.float().contiguous()     # [B,H,V,K]

        # Allocate outputs
        output = torch.empty((B, H), dtype=torch.float32, device=device)  # [B,H]
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)  # [B,H,V,K]

        # Compute g and beta: one program per (b,h)
        g_f32 = torch.empty(B * H, dtype=torch.float32, device=device)
        beta_f32 = torch.empty(B * H, dtype=torch.float32, device=device)
        grid = (B * H,)
        softplus_and_exp_kernel[grid](dt_bias_f32, a_f32, A_log_f32, g_f32, B=B, H=H)
        sigmoid_kernel[grid](b_f32, beta_f32, B=B, H=H)

        # Process each (b,h) vector
        for b_idx in range(B):
            for h_idx in range(H):
                pid = b_idx * H + h_idx

                # Scalars
                g_scalar = g_f32[pid]
                beta_scalar = beta_f32[pid]

                # Pointers for this (b,h)
                state_bh = state_f32[b_idx, h_idx]  # [V,K]
                k_vec = k_f32[b_idx]                # [K]
                v_vec = v_f32[b_idx]                # [K]
                q_vec = q_f32[b_idx]                # [K]

                # Compute old_v = k @ state
                old_v_scalar = torch.empty(1, dtype=torch.float32, device=device)
                dot_k_state_kernel[(K,)](k_vec, state_bh, old_v_scalar, V=V, K=K)
                old_v = old_v_scalar[0]

                # Compute state_remove = k @ (g * state)
                state_remove_scalar = torch.empty(1, dtype=torch.float32, device=device)
                dot_k_gstate_kernel[(K,)](k_vec, state_bh, g_scalar, state_remove_scalar, V=V, K=K)
                state_remove = state_remove_scalar[0]

                # Compute state_update = k @ (beta*v + (1 - beta)*old_v)
                state_update_scalar = torch.empty(1, dtype=torch.float32, device=device)
                dot_k_newv_kernel[(K,)](k_vec, v_vec, old_v, beta_scalar, state_update_scalar, K=K)
                state_update = state_update_scalar[0]

                # Compute h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update
                h_state = torch.empty(V, dtype=torch.float32, device=device)
                h_state_vec_kernel[(V,)](state_bh, g_scalar, state_remove, state_update, h_state, V=V, K=K)

                # Compute output[b,h] = scale * (q @ h_state)
                output_bh_scalar = torch.empty(1, dtype=torch.float32, device=device)
                dot_q_hstate_kernel[(V, K)](q_vec, h_state, output_bh_scalar, V=V, K=K, scale=float(scale))
                output[b_idx, h_idx] = output_bh_scalar[0]

                # Write new_state[b,h] as [V,K] by broadcasting h_state across K
                new_state_bh_ptr = new_state[b_idx, h_idx]  # [V,K] contiguous
                write_new_state_kernel[(V, K)](h_state, new_state_bh_ptr, V=V, K=K)

        # Cast output to bfloat16 and return as [B,1,H]
        output_b1H = output.unsqueeze(1).to(torch.bfloat16)  # [B,1,H]
        return output_b1H, new_state


def run(*args):
    return ModelNew()(*args)
