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
    e = tl.exp(tl.load(A_log_ptr + h))  # A_log is same for all b
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Kernel: compute beta = sigmoid(b[b,h]) using b_ptr of length B*H
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


# Kernel: compute h_state_vec[i] = (sum_j state[i,j] * k[j]) - state_remove + state_update
# state_ptr: [V*K] flattened, k_ptr: [K], g_scalar: g[b,h], state_remove: [1], state_update: [1], h_state_ptr: [V]
@triton.jit
def h_state_vec_kernel_with_k(
    k_ptr,          # float32 [K]
    g_scalar,       # float32 scalar g[b,h]
    state_ptr,      # float32 [V*K]
    state_remove_ptr,    # float32 [1]
    state_update_ptr,    # float32 [1]
    h_state_ptr,         # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    # For each row i, compute sum_j state[i,j] * k[j], then adjust
    for i in range(V):
        total = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            k_j = tl.load(k_ptr + j)
            total += state_ij * k_j
        # Apply scalar adjustments
        state_remove_val = tl.load(state_remove_ptr)
        state_update_val = tl.load(state_update_ptr)
        total = total - state_remove_val + state_update_val
        tl.store(h_state_ptr + i, total)


# Kernel: output[b,h] = scale * (q[b,h] @ h_state_vec)
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    output_ptr,     # float32 [B*H]
    V: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    acc = 0.0
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        for j in range(K):
            q_j = tl.load(q_ptr + j)
            acc += h_i * q_j
    tl.store(output_ptr + pid, acc)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [*] (we pass flattened [B*H*V*K] region for this (b,h))
    V: tl.constexpr,
    K: tl.constexpr,
    stride_v: tl.constexpr,  # typically K
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    # For this pid, we write one [V,K] block. We need to map pid to b and h to get base pointer.
    # Since we launch per (b,h), we can assume base offset for new_state is pid * V*K. But new_state is [B,H,V,K].
    # We need to compute b_idx and h_idx from pid: not straightforward in Triton kernel. Instead, forward will pass new_state_ptr as [B,H,V,K] row pointer for this (b,h).
    # To keep things simple, forward will call this kernel with new_state_ptr pointing to new_state[b,h].view(-1).
    # Implement generic store: forward ensures the pointer points to [V*K] for this (b,h).
    for i in range(V):
        val = tl.load(h_state_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, val)


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
    g_scalar,       # float32 scalar
    state_ptr,      # float32 [V*K]
    state_remove_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * g_scalar * k_j
    tl.store(state_remove_ptr, acc)


# Kernel: compute state_update = dot(k_vec, beta * v_vec + (1-beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    beta_scalar,    # float32 scalar
    v_ptr,          # float32 [V]
    old_v_scalar,   # float32 scalar
    state_update_ptr,   # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    # Precompute new_v contribution per j
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        # new_v_i depends on i; we can compute it inside the j loop by iterating i and multiplying. But simpler: compute new_v per i and then dot with k_j.
        # Instead, compute total contribution directly:
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            new_v_i = beta_scalar * v_i + (1.0 - beta_scalar) * old_v_scalar
            acc += new_v_i * k_j
    tl.store(state_update_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        # We accept arbitrary args to satisfy the caller; forward uses the first 9.

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure CUDA tensors and float32 for computation
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "All tensors must be CUDA tensors"
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape

        # Cast inputs to float32 for computation
        device = q.device
        qf = q.float()   # [B, T, num_q_heads, K]
        kf = k.float()   # [B, T, num_k_heads, K]
        vf = v.float()   # [B, T, num_v_heads, V]
        af = a.float()   # [B, 1, num_v_heads]
        dt_bias_f = dt_bias.float()   # [num_v_heads]
        A_log_f = A_log.float()       # [num_v_heads]
        state_f = state.float()       # [B, num_v_heads, V, K]

        # Compute g[b,h] and beta[b,h] using Triton
        BpH = B * num_v_heads
        g = torch.empty(BpH, dtype=torch.float32, device=device)  # [B*H]
        beta = torch.empty(BpH, dtype=torch.float32, device=device)  # [B*H]

        # Launch softplus_and_exp_kernel
        grid_g = (BpH,)
        softplus_and_exp_kernel[grid_g](dt_bias_f, af.view(-1), A_log_f, g, H=num_v_heads, num_warps=1)

        # Launch sigmoid_kernel
        bf = b.float()
        grid_beta = (BpH,)
        sigmoid_kernel[grid_beta](bf.view(-1), beta, H=num_v_heads, num_warps=1)

        # Output and new state buffers
        output = torch.empty((B, num_v_heads, V), dtype=torch.float32, device=device)  # per (b,h,v)
        new_state = torch.empty((B, num_v_heads, V, K), dtype=torch.float32, device=device)  # [B,H,V,K]

        # Process each (b,h)
        for b_idx in range(B):
            for h_idx in range(num_v_heads):
                pid = b_idx * num_v_heads + h_idx

                # Vectors and matrix
                q_vec = qf[b_idx, 0, h_idx]  # [K]
                k_vec = kf[b_idx, 0, h_idx]  # [K]
                v_vec = vf[b_idx, 0, h_idx]  # [V]
                state_mat = state_f[b_idx, h_idx]  # [V,K], contiguous

                # Scalars
                g_val = g[pid]     # scalar
                beta_val = beta[pid]  # scalar

                # Temporary buffers
                old_v = torch.empty((), dtype=torch.float32, device=device)  # [1]
                state_remove = torch.empty((), dtype=torch.float32, device=device)  # [1]
                state_update = torch.empty((), dtype=torch.float32, device=device)  # [1]
                h_state_vec = torch.empty(V, dtype=torch.float32, device=device)  # [V]

                # Compute old_v = k @ state
                dot_k_state_kernel[(1,)](k_vec, state_mat, old_v, V, K, num_warps=1)

                # Compute state_remove = k @ (g * state)
                dot_k_gstate_kernel[(1,)](k_vec, g_val, state_mat, state_remove, V, K, num_warps=1)

                # Compute state_update = k @ (beta * v + (1-beta) * old_v)
                dot_k_newv_kernel[(1,)](k_vec, beta_val, v_vec, old_v, state_update, V, K, num_warps=1)

                # Compute h_state_vec[i] = sum_j state[i,j] * k[j] - state_remove + state_update
                h_state_vec_kernel_with_k[(1,)](k_vec, g_val, state_mat, state_remove, state_update, h_state_vec, V, K, num_warps=1)

                # Output scalar: output[b,h] = scale * (q @ h_state_vec)
                out_scalar = torch.empty((), dtype=torch.float32, device=device)
                dot_q_hstate_kernel[(1,)](q_vec, h_state_vec, out_scalar, V, K, num_warps=1)
                output[b_idx, h_idx] = out_scalar * scale

                # Write new_state[b,h] as [V,K]: broadcast h_state_vec across K
                new_state_row = new_state[b_idx, h_idx].view(-1)  # [V*K]
                write_new_state_kernel[(1,)](h_state_vec, new_state_row, V, K, stride_v=K, num_warps=1)

        # Return output as [B,1,H] in bfloat16 and new_state as [B,H,V,K] float32
        output_bf16 = output.view(B, 1, num_v_heads).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
