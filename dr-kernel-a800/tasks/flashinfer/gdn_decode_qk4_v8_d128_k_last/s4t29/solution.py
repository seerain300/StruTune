import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute g[b,h] = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
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


# Triton kernel: compute beta[b,h] = sigmoid(b[b,h])
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


# Triton kernel: compute old_v = dot(k_vec[K], state_mat[V*K]) -> scalar
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


# Triton kernel: compute state_remove = dot(k_vec, g * state_mat) -> scalar
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


# Triton kernel: compute state_update = dot(k_vec, beta * v_vec + (1 - beta) * old_v) -> scalar
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    old_v_ptr,      # float32 [1]
    beta_ptr,       # float32 [1]
    state_update_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    beta = tl.load(beta_ptr)
    old_v = tl.load(old_v_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (beta * v_i + (1.0 - beta) * old_v)
    tl.store(state_update_ptr, acc)


# Triton kernel: compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
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
        h_state_i = acc - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_i)


# Triton kernel: compute output_scalar = scale * dot(q_vec[K], h_state_vec[V])
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    scale_ptr,      # float32 [1]
    output_ptr,     # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    scale = tl.load(scale_ptr)
    acc = 0.0
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        for j in range(K):
            q_j = tl.load(q_ptr + j)
            acc += h_i * q_j
    tl.store(output_ptr, acc * scale)


# Triton kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
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
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        state: torch.Tensor, A_log: torch.Tensor, a: torch.Tensor,
        dt_bias: torch.Tensor, b: torch.Tensor, scale: float
    ) -> list:
        """
        Triton-optimized forward. Returns [output [B,1,H], new_state [B,H,V,K]].
        output dtype: bfloat16; new_state dtype: float32
        """
        # Shapes from inputs
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        device = q.device

        # Ensure float32 for numerical stability; make contiguous
        q_f32 = q.float().contiguous()
        k_f32 = k.float().contiguous()
        v_f32 = v.float().contiguous()
        # a,b: [B,1,H] -> flatten to [B*H]
        a_f32 = a.squeeze(1).float().contiguous()
        b_f32 = b.squeeze(1).float().contiguous()
        A_log_f32 = A_log.float().contiguous()
        dt_bias_f32 = dt_bias.float().contiguous()
        state_f32 = state.float().contiguous()

        # Output and new state
        output_f32 = torch.empty((B, H), dtype=torch.float32, device=device)
        new_state_f32 = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # 1) Compute g and beta vectors [B*H]
        g_vec = torch.empty(B * H, dtype=torch.float32, device=device)
        beta_vec = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch kernels for g and beta
        grid_g = (B * H,)
        softplus_and_exp_kernel[grid_g](dt_bias_f32, a_f32, A_log_f32, g_vec, H=H)
        sigmoid_kernel[grid_g](b_f32, beta_vec, H=H)

        # 2) For each batch b and head h, compute outputs and new_state
        for b_idx in range(B):
            for h_idx in range(H):
                # Compute per-(b,h) scalars and vectors
                # old_v = k @ state
                old_v = torch.empty((), dtype=torch.float32, device=device)
                k_bh = k_f32[b_idx, 0, h_idx].contiguous()  # [K]
                state_bh = state_f32[b_idx, h_idx].contiguous()  # [V*K]
                dot_k_state_kernel[(K,)](k_bh, state_bh, old_v, V=V, K=K)

                # state_remove = k @ (g * state)
                g_val = torch.empty((), dtype=torch.float32, device=device)
                g_scalar = torch.empty((), dtype=torch.float32, device=device)
                g_scalar[:] = g_vec[b_idx * H + h_idx]
                dot_k_gstate_kernel[(K,)](k_bh, state_bh, g_scalar, state_remove, V=V, K=K)

                # state_update = k @ (beta * v + (1 - beta) * old_v)
                beta_val = torch.empty((), dtype=torch.float32, device=device)
                beta_scalar = torch.empty((), dtype=torch.float32, device=device)
                beta_scalar[:] = beta_vec[b_idx * H + h_idx]
                dot_k_newv_kernel[(K,)](k_bh, v_f32[b_idx, 0, h_idx].contiguous(), old_v, beta_scalar, state_update, V=V, K=K)

                # h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update
                h_state_vec = torch.empty(V, dtype=torch.float32, device=device)
                h_state_vec_kernel[(V,)](state_bh, g_scalar, state_remove, state_update, h_state_vec, V=V, K=K)

                # output_scalar = scale * (q @ h_state_vec)
                output_scalar = torch.empty((), dtype=torch.float32, device=device)
                q_bh = q_f32[b_idx, 0, h_idx].contiguous()  # [K]
                scale_t = torch.empty((), dtype=torch.float32, device=device)
                scale_t[:] = float(scale)  # original uses scale=1.0; keep as scalar
                dot_q_hstate_kernel[(V,)](q_bh, h_state_vec, scale_t, output_scalar, V=V, K=K)

                # Store output as [B, H]
                output_f32[b_idx, h_idx] = output_scalar.item()  # save scalar

                # write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
                new_row = torch.empty(V * K, dtype=torch.float32, device=device)
                write_new_state_kernel[(V,)](h_state_vec, new_row, V=V, K=K)
                new_state_f32[b_idx, h_idx] = new_row.view(V, K)

        # Return [output [B,1,H], new_state [B,H,V,K]]
        output = output_f32.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H]
        return [output, new_state_f32]


def run(*args):
    return ModelNew()(*args)
