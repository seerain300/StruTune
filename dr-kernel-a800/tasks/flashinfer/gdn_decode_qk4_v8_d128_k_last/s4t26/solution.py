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


# Kernel: compute old_v = dot(k_vec[K], state_mat[V*K]) -> scalar (single element tensor)
@triton.jit
def dot_k_state_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K] (flattened)
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


# Kernel: compute state_remove = dot(k_vec, g * state_mat) -> scalar
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K] (flattened)
    g_scalar_ptr,   # float32 [1]
    state_remove_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    g = tl.load(g_scalar_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j * g
    tl.store(state_remove_ptr, acc)


# Kernel: compute state_update = dot(k_vec, beta * v_vec + (1 - beta) * old_v) -> scalar
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    old_v_ptr,      # float32 [1]
    beta_scalar_ptr,  # float32 [1]
    state_update_ptr, # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    beta = tl.load(beta_scalar_ptr)
    old_v = tl.load(old_v_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (beta * v_i + (1.0 - beta) * old_v)
    tl.store(state_update_ptr, acc)


# Kernel: compute h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update, vector of length V
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K] (flattened)
    g_scalar_ptr,   # float32 [1]
    state_remove_ptr,  # float32 [1]
    state_update_ptr,  # float32 [1]
    h_state_ptr,   # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    g = tl.load(g_scalar_ptr)
    state_remove = tl.load(state_remove_ptr)
    state_update = tl.load(state_update_ptr)
    for i in range(V):
        acc = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * g
        h_state_vec_i = acc - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_vec_i)


# Kernel: compute output_scalar[b,h] = scale * dot(q_vec[K], h_state_vec[V])
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    scale,          # float32 scalar
    output_ptr,     # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        for j in range(K):
            q_j = tl.load(q_ptr + j)
            acc += h_i * q_j
    acc = acc * scale
    tl.store(output_ptr, acc)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [V*K] flattened
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, h_i)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward. All computation happens in Triton kernels.
        Returns:
          - output: [B, 1, H] in bfloat16
          - new_state: [B, H, V, K] in float32
        """
        # Shapes
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        device = q.device

        # Prepare inputs as float32 on device
        q_f32 = q.contiguous().float()
        k_f32 = k.contiguous().float()
        v_f32 = v.contiguous().float()
        a_f32 = a.contiguous().float()
        b_f32 = b.contiguous().float()
        A_log_f32 = A_log.contiguous().float()
        dt_bias_f32 = dt_bias.contiguous().float()
        state_f32 = state.contiguous().float()

        H = num_v_heads

        # Allocate outputs for g and beta
        g_f32 = torch.empty(B * H, dtype=torch.float32, device=device)
        beta_f32 = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch softplus_and_exp_kernel and sigmoid_kernel
        grid_g = (B * H,)
        softplus_and_exp_kernel[grid_g](dt_bias_f32, a_f32, A_log_f32, g_f32, B, H)
        grid_beta = (B * H,)
        sigmoid_kernel[grid_beta](b_f32, beta_f32, H)

        # Prepare outputs
        output = torch.empty((B, H), dtype=torch.float32, device=device)  # [B, H], we will cast to bfloat16 later
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # For each (b,h), compute everything in Triton
        for b_idx in range(B):
            for h_idx in range(H):
                # Current pointers for this (b,h)
                base_q = q_f32[b_idx, 0, h_idx]  # [K]
                base_k = k_f32[b_idx, 0, h_idx]  # [K]
                base_v = v_f32[b_idx, 0, h_idx]  # [V]
                base_state = state_f32[b_idx, h_idx]  # [V, K], contiguous

                # Scalars
                g_scalar = g_f32[b_idx * H + h_idx]
                beta_scalar = beta_f32[b_idx * H + h_idx]

                # Temporary buffers (1-element)
                old_v = torch.empty(1, dtype=torch.float32, device=device)
                state_remove = torch.empty(1, dtype=torch.float32, device=device)
                state_update = torch.empty(1, dtype=torch.float32, device=device)
                h_state_vec = torch.empty(V, dtype=torch.float32, device=device)

                # Launch dot kernels for scalars
                dot_k_state_kernel[(K,)](base_k, base_state.reshape(V * K), old_v, V, K)
                dot_k_gstate_kernel[(K,)](base_k, base_state.reshape(V * K), g_scalar, state_remove, V, K)
                dot_k_newv_kernel[(K,)](base_k, base_v, old_v, beta_scalar, state_update, V, K)

                # Compute h_state_vec
                h_state_vec_kernel[(V,)](base_state.reshape(V * K), g_scalar, state_remove, state_update, h_state_vec, V, K)

                # Compute output[b,h] = scale * (q @ h_state_vec)
                output[b_idx, h_idx] = dot_q_hstate_kernel[(K,)](base_q, h_state_vec, float(scale), output[b_idx, h_idx], V, K)

                # Write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
                new_state[b_idx, h_idx] = write_new_state_kernel[(V, K)](h_state_vec, new_state[b_idx, h_idx].reshape(V * K), V, K)

        # Return as required: [output, new_state], cast output to bfloat16
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H]
        return [output_bf16, new_state]


def run(*args):
    return ModelNew()(*args)
