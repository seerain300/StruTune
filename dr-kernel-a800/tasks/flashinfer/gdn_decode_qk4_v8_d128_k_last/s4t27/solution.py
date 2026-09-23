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
    B: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Kernel: compute old_v = dot(k_vec[K], state_mat[V*K]) -> scalar written to old_v_ptr[0]
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


# Kernel: compute state_remove = dot(k_vec[K], g_scalar * state_mat[V*K]) -> scalar written to state_remove_ptr[0]
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


# Kernel: compute state_update = dot(k_vec[K], beta_scalar * v_vec[K] + (1 - beta_scalar) * old_v_scalar) -> scalar written to state_update_ptr[0]
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [K]
    beta_scalar_ptr,   # float32 [1]
    old_v_ptr,      # float32 [1]
    state_update_ptr,  # float32 [1]
    K: tl.constexpr,
):
    beta = tl.load(beta_scalar_ptr)
    old_v = tl.load(old_v_ptr)
    acc = 0.0
    for j in range(K):
        v_j = tl.load(v_ptr + j)
        k_j = tl.load(k_ptr + j)
        acc += k_j * (beta * v_j + (1.0 - beta) * old_v)
    tl.store(state_update_ptr, acc)


# Kernel: compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update, write to h_state_ptr[V]
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
        sum_j = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            sum_j += state_ij
        h_state_val = sum_j * g - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_val)


# Kernel: compute output_scalar = scale * dot(q_vec[K], h_state_vec[V]) -> write to out_ptr[0]
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    scale,          # float32 scalar
    out_ptr,        # float32 [1]
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
    tl.store(out_ptr, acc)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec[V] across K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [V*K] flattened row-major
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, h_i)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Shapes
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        H = num_v_heads  # number of heads for v
        assert T == 1
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8
        assert K == 128 and V == 128

        device = q.device

        # Convert to float32 for compute
        q_f32 = q.squeeze(1).contiguous().to(torch.float32)      # [B, H, K]
        k_f32 = k.squeeze(1).contiguous().to(torch.float32)      # [B, H, K]
        v_f32 = v.squeeze(1).contiguous().to(torch.float32)      # [B, H, K]
        a_f32 = a.contiguous().to(torch.float32)                 # [B, H]
        b_f32 = b.contiguous().to(torch.float32)                 # [B, H]
        state_f32 = state.contiguous().to(torch.float32)         # [B, H, V, K]
        A_log_f32 = A_log.contiguous().to(torch.float32)         # [H]
        dt_bias_f32 = dt_bias.contiguous().to(torch.float32)     # [H]

        # Allocate outputs for g and beta
        g_bH = torch.empty(B * H, dtype=torch.float32, device=device)  # [B*H]
        beta_bH = torch.empty(B * H, dtype=torch.float32, device=device)  # [B*H]

        # Compute g[b,h]
        softplus_and_exp_kernel[(B * H,)](
            dt_bias_f32, a_f32, A_log_f32, g_bH,
            B=B, H=H
        )

        # Compute beta[b,h]
        sigmoid_kernel[(B * H,)](
            b_f32.reshape(-1), beta_bH,
            B=B, H=H
        )

        # Outputs
        output = torch.empty((B, H), dtype=torch.bfloat16, device=device)  # [B, H]
        new_state_f32 = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Per (b,h) compute
        for b_idx in range(B):
            for h_idx in range(H):
                pid = b_idx * H + h_idx

                # Vectors/matrices
                q_vec = q_f32[b_idx, h_idx]     # [K]
                k_vec = k_f32[b_idx, h_idx]     # [K]
                v_vec = v_f32[b_idx, h_idx]     # [K]
                state_mat = state_f32[b_idx, h_idx]  # [V, K] row-major: index i*K + j

                # 1) old_v = k @ state
                old_v = torch.empty((), dtype=torch.float32, device=device)
                dot_k_state_kernel[(1,)](
                    k_vec, state_mat.reshape(-1), old_v,
                    V=V, K=K
                )

                # 2) state_remove = k @ (g * state)
                state_remove = torch.empty((), dtype=torch.float32, device=device)
                dot_k_gstate_kernel[(1,)](
                    k_vec, state_mat.reshape(-1), g_bH[pid:pid+1], state_remove,
                    V=V, K=K
                )

                # 3) state_update = k @ (beta * v + (1 - beta) * old_v)
                state_update = torch.empty((), dtype=torch.float32, device=device)
                dot_k_newv_kernel[(1,)](
                    k_vec, v_vec, beta_bH[pid:pid+1], old_v, state_update,
                    K=K
                )

                # 4) h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
                h_state_vec = torch.empty(V, dtype=torch.float32, device=device)
                h_state_vec_kernel[(1,)](
                    state_mat.reshape(-1), g_bH[pid:pid+1], state_remove, state_update, h_state_vec,
                    V=V, K=K
                )

                # 5) output_scalar = scale * dot(q, h_state_vec)
                out_scalar = torch.empty((), dtype=torch.float32, device=device)
                dot_q_hstate_kernel[(1,)](
                    q_vec, h_state_vec, float(scale), out_scalar,
                    V=V, K=K
                )
                output[b_idx, h_idx] = out_scalar  # store scalar per (b,h)

                # 6) write new_state[b,h] = [V,K] broadcasting h_state_vec across K
                new_row = torch.empty(V * K, dtype=torch.float32, device=device)
                write_new_state_kernel[(1,)](
                    h_state_vec, new_row,
                    V=V, K=K
                )
                new_state_f32[b_idx, h_idx] = new_row.view(V, K)

        # Return [output [B,1,H], new_state [B,H,V,K]]
        output = output.unsqueeze(1)  # [B, 1, H]
        return [output.to(torch.bfloat16), new_state_f32]


def run(*args):
    return ModelNew()(*args)
