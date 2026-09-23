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


# Kernel: compute old_v = dot(k_vec[K], state_mat[V*K]) -> scalar
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


# Kernel: compute state_update = dot(k_vec, beta * v_vec + (1-beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    beta_scalar_ptr,   # float32 [1]
    old_v_ptr,      # float32 [1]
    state_update_ptr,  # float32 [1]
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


# Kernel: compute h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_scalar_ptr,   # float32 [1]
    state_remove_ptr,  # float32 [1]
    state_update_ptr,  # float32 [1]
    h_state_ptr,    # float32 [V]
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
        h_state_val = acc - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_val)


# Kernel: compute output_scalar = q_vec @ h_state_vec
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


# Kernel: compute scale = 1.0 / sqrt(K) and store to scale_ptr[0] as float32
@triton.jit
def scale_sqrt_kernel(
    K,              # int32 scalar
    scale_ptr,      # float32 [1]
):
    inv_sqrt = 1.0 / tl.sqrt(K)
    tl.store(scale_ptr, inv_sqrt)


class ModelNew(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        # No parameters; capture args for signature compatibility

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation. Returns (output (bfloat16), new_state (float32)).
        """
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and state.dim() == 4
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        assert T == 1
        device = q.device

        # Ensure dtype and contiguity for compute
        q_f32 = q.float().contiguous()
        k_f32 = k.float().contiguous()
        v_f32 = v.float().contiguous()
        a_f32 = a.float().contiguous()
        b_f32 = b.float().contiguous()
        A_log_f32 = A_log.float().contiguous()
        dt_bias_f32 = dt_bias.float().contiguous()
        state_f32 = state.float().contiguous()

        # Compute g and beta using Triton
        B_flat = B * num_v_heads
        g = torch.empty(B_flat, dtype=torch.float32, device=device)
        beta = torch.empty(B_flat, dtype=torch.float32, device=device)

        # Launch softplus_and_exp_kernel
        grid_g = (B_flat,)
        softplus_and_exp_kernel[grid_g](dt_bias_f32, a_f32, A_log_f32, g, B, num_v_heads)

        # Launch sigmoid_kernel
        beta_kernel = torch.empty_like(a_f32, dtype=torch.float32, device=device)
        grid_beta = (B_flat,)
        sigmoid_kernel[grid_beta](b_f32, beta_kernel, num_v_heads)
        beta = beta_kernel.view(B, num_v_heads)
        g = g.view(B, num_v_heads)

        # Prepare outputs
        output = torch.empty(B, num_v_heads, dtype=torch.float32, device=device)
        new_state = torch.empty(B, num_v_heads, V, K, dtype=torch.float32, device=device)

        # Compute scale in Triton: scale = 1/sqrt(K)
        scale_buf = torch.empty(1, dtype=torch.float32, device=device)
        scale_sqrt_kernel[(1,)](K, scale_buf)
        scale = scale_buf[0] if scale is None or scale == 0.0 else float(scale)

        # For each (b,h), compute h_state_vec and new_state
        for b_idx in range(B):
            for h_idx in range(num_v_heads):
                # Flatten state[b,h] to [V*K]
                state_bh = state_f32[b_idx, :, h_idx].contiguous()  # [V,K]
                state_bh_flat = state_bh.reshape(V * K).contiguous()

                # Vectors
                k_vec = k_f32[b_idx, 0, h_idx].contiguous()  # [K]
                q_vec = q_f32[b_idx, 0, h_idx].contiguous()  # [K]
                v_vec = v_f32[b_idx, 0, h_idx].contiguous()  # [V]

                # 1) old_v = dot(k_vec, state_bh_flat)
                old_v = torch.empty(1, dtype=torch.float32, device=device)
                dot_k_state_kernel[(1,)](k_vec, state_bh_flat, old_v, V, K)

                # 2) state_remove = dot(k_vec, g[b,h] * state_bh_flat)
                g_val = g[b_idx, h_idx]
                state_remove = torch.empty(1, dtype=torch.float32, device=device)
                dot_k_gstate_kernel[(1,)](k_vec, state_bh_flat, g_val, state_remove, V, K)

                # 3) state_update = dot(k_vec, beta[b,h] * v_vec + (1 - beta) * old_v)
                beta_val = beta[b_idx, h_idx]
                state_update = torch.empty(1, dtype=torch.float32, device=device)
                dot_k_newv_kernel[(1,)](k_vec, v_vec, beta_val, old_v, state_update, V, K)

                # 4) h_state_vec[i] = sum_j state_bh[i,j] * g - state_remove + state_update
                h_state = torch.empty(V, dtype=torch.float32, device=device)
                h_state_vec_kernel[(1,)](state_bh_flat, g_val, state_remove, state_update, h_state, V, K)

                # 5) output[b,h] = scale * dot(q_vec, h_state)
                output_scalar = torch.empty(1, dtype=torch.float32, device=device)
                dot_q_hstate_kernel[(1,)](q_vec, h_state, output_scalar, V, K)
                output[b_idx, h_idx] = output_scalar[0] * scale

                # 6) write new_state[b,h] as [V,K] by broadcasting h_state across K
                new_state_bh = new_state[b_idx, h_idx].reshape(V * K)
                write_new_state_kernel[(1,)](h_state, new_state_bh, V, K)

        # Return as expected by harness: (output (bfloat16), new_state (float32))
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H]
        return (output_bf16, new_state)


def run(*args):
    return ModelNew()(*args)
