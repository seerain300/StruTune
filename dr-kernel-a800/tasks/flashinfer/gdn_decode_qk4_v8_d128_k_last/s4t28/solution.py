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


# Kernel: compute state_remove = dot(k_vec, g * state_mat) -> scalar
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


# Kernel: compute state_update = dot(k_vec, beta * v_vec + (1 - beta) * old_v) -> scalar
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    beta_scalar_ptr,  # float32 [1]
    old_v_scalar_ptr, # float32 [1]
    state_update_ptr, # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    beta = tl.load(beta_scalar_ptr)
    old_v = tl.load(old_v_scalar_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (v_i * beta + (1.0 - beta) * old_v)
    tl.store(state_update_ptr, acc)


# Kernel: compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update, length V
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_scalar_ptr,   # float32 [1]
    state_remove_ptr,  # float32 [1]
    state_update_ptr,  # float32 [1]
    h_state_out_ptr,   # float32 [V]
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
        h_state_out_ptr[i] = acc - state_remove + state_update


# Kernel: output_scalar[b,h] = scale * dot(q_vec[K], h_state_vec[V])
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    scale_scalar_ptr,  # float32 [1]
    output_ptr,     # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    scale = tl.load(scale_scalar_ptr)
    acc = 0.0
    for j in range(K):
        q_j = tl.load(q_ptr + j)
        for i in range(V):
            h_i = tl.load(h_state_ptr + i)
            acc += q_j * h_i
    acc *= scale
    tl.store(output_ptr, acc)


# Kernel: write new_state[b,h] as [V,K], broadcasting h_state_vec across K
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
            new_state_ptr[i * K + j] = val


class ModelNew(torch.nn.Module):
    def forward(
        self,
        q, k, v, state, A_log, a, dt_bias, b, scale,
    ):
        """
        Compute:
          g = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
          beta = sigmoid(b[b,h])
          h_state[b,h] = (sum_j state[b,h,i,j] * g[b,h]) - (k[b,h] @ (g[b,h] * state[b,h])) + (k[b,h] @ (beta[b,h] * v[b,h] + (1 - beta[b,h]) * (k[b,h] @ state[b,h])))
          output[b,h] = scale * (q[b,h] @ h_state[b,h])
          new_state[b,h] = [V,K] with h_state broadcast along K

        Returns:
          [output [B,1,H] bfloat16, new_state [B,H,V,K] float32]
        """
        # Ensure inputs are float32 on CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "All inputs must be CUDA tensors"

        B_q, T_q, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        # From original asserts (kept for correctness): num_q_heads == 4, num_k_heads == 4, num_v_heads == 8
        # But we will not hardcode; use actual shapes.

        # Prepare pointers
        a_f32 = a.float().reshape(-1)               # [B*H]
        dt_bias_f32 = dt_bias.float()               # [H]
        A_log_f32 = A_log.float()                   # [H]
        b_f32 = b.float().reshape(-1)               # [B*H]
        q_f32 = q.float().contiguous().reshape(B_q, -1)  # [B_q, K] (T_q is 1 in original)
        k_f32 = k.float().contiguous().reshape(B_q, -1)  # [B_q, K]
        v_f32 = v.float().contiguous().reshape(B_q, num_v_heads, V)  # [B_q, H, V]
        state_f32 = state.float().contiguous().reshape(B_q, num_v_heads, V, K)  # [B_q, H, V, K]

        # Compute g and beta using Triton kernels
        g_out = torch.empty(a_f32.shape[0], dtype=torch.float32, device=a_f32.device)
        grid_g = (a_f32.shape[0],)
        softplus_and_exp_kernel[grid_g](dt_bias_f32, a_f32, A_log_f32, g_out, B_q, num_v_heads)

        beta_out = torch.empty(b_f32.shape[0], dtype=torch.float32, device=b_f32.device)
        grid_beta = (b_f32.shape[0],)
        sigmoid_kernel[grid_beta](b_f32, beta_out, num_v_heads)

        # We need to compute per (b,h). Since T_q == 1, B_q == B and H == num_v_heads.
        # Output is [B_q, 1, H]. New state is [B_q, H, V, K].
        output = torch.empty((B_q, num_v_heads), dtype=torch.float32, device=a_f32.device)
        new_state = torch.empty((B_q, num_v_heads, V, K), dtype=torch.float32, device=a_f32.device)

        # Scale: if provided, use it; else use 1/sqrt(K)
        scale_tensor = torch.tensor(float(scale) if scale is not None else (1.0 / math.sqrt(K)), dtype=torch.float32, device=a_f32.device)

        # Loop over b and h; Triton kernels handle per-(b,h) logic
        for b_idx in range(B_q):
            for h_idx in range(num_v_heads):
                # k_vec: [K]
                k_vec = k_f32[b_idx]                      # [K]
                # state_mat: [V,K]
                state_mat = state_f32[b_idx, h_idx].contiguous().view(V * K)  # [V*K]

                # old_v = k @ state
                old_v = torch.empty((), dtype=torch.float32, device=a_f32.device)
                dot_k_state_kernel[(1,)](
                    k_vec, state_mat, old_v,
                    V=V, K=K
                )

                # state_remove = k @ (g * state)
                g_scalar = torch.empty((), dtype=torch.float32, device=a_f32.device)
                g_scalar.copy_(g_out[b_idx * num_v_heads + h_idx])  # g[b,h]
                state_remove = torch.empty((), dtype=torch.float32, device=a_f32.device)
                dot_k_gstate_kernel[(1,)](
                    k_vec, state_mat, g_scalar, state_remove,
                    V=V, K=K
                )

                # state_update = k @ (beta * v + (1 - beta) * old_v)
                beta_scalar = torch.empty((), dtype=torch.float32, device=a_f32.device)
                beta_scalar.copy_(beta_out[b_idx * num_v_heads + h_idx])
                state_update = torch.empty((), dtype=torch.float32, device=a_f32.device)
                v_vec = v_f32[b_idx, h_idx].contiguous().view(V)  # [V]
                dot_k_newv_kernel[(1,)](
                    k_vec, v_vec, beta_scalar, old_v, state_update,
                    V=V, K=K
                )

                # h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update
                h_state_vec = torch.empty(V, dtype=torch.float32, device=a_f32.device)
                h_state_vec_kernel[(1,)](
                    state_mat, g_scalar, state_remove, state_update, h_state_vec,
                    V=V, K=K
                )

                # output_scalar[b,h] = scale * (q[b,h] @ h_state_vec)
                q_vec = q_f32[b_idx]  # [K]
                output_scalar = torch.empty((), dtype=torch.float32, device=a_f32.device)
                dot_q_hstate_kernel[(1,)](
                    q_vec, h_state_vec, scale_tensor, output_scalar,
                    V=V, K=K
                )
                output[b_idx, h_idx] = output_scalar

                # new_state[b,h] = [V,K], broadcast h_state_vec across K
                new_row = torch.empty(V * K, dtype=torch.float32, device=a_f32.device)
                write_new_state_kernel[(1,)](
                    h_state_vec, new_row,
                    V=V, K=K
                )
                new_state[b_idx, h_idx] = new_row.view(V, K)

        # Return [output [B_q, H], new_state [B_q, H, V, K]]
        output = output.unsqueeze(1)  # [B_q, 1, H]
        return [output.to(torch.bfloat16), new_state]


def run(*args):
    return ModelNew()(*args)
