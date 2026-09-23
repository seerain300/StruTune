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


# Triton kernel: compute beta[b,h] = 1 / (1 + exp(-b[b,h]))
@triton.jit
def sigmoid_kernel(
    b_ptr,          # float32 [B*H]
    beta_out_ptr,   # float32 [B*H]
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Triton kernel: compute scalar dot(old_v) = dot(k_vec, state_mat) where state_mat is [V*K] flattened
# Inputs:
#   k_ptr:          float32 [K]
#   state_ptr:      float32 [V*K]
#   old_v_ptr:      float32 [1]
#   V: tl.constexpr
#   K: tl.constexpr
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


# Triton kernel: compute scalar state_remove = dot(k_vec, g_scalar * state_mat)
# state_mat [V*K], k_vec [K], g_scalar is passed as 1-element tensor and loaded inside
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    g_scalar_ptr,   # float32 [1] (g[b,h])
    state_remove_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    gs = tl.load(g_scalar_ptr)  # scalar
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * gs * k_j
    tl.store(state_remove_ptr, acc)


# Triton kernel: compute scalar state_update = dot(k_vec, beta_scalar * v_vec + (1 - beta_scalar) * old_v_scalar)
# v_vec [V], old_v_scalar is passed as 1-element tensor and loaded inside
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    beta_scalar_ptr,   # float32 [1]
    old_v_scalar_ptr,  # float32 [1]
    state_update_ptr,  # float32 [1]
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
            acc += k_j * (beta * v_i + (1.0 - beta) * old_v)
    tl.store(state_update_ptr, acc)


# Triton kernel: compute vector h_state_vec[V] = sum_j state[b,h,i,j] * g[b,h] - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_scalar_ptr,   # float32 [1]
    state_remove_ptr,  # float32 [1]
    state_update_ptr,  # float32 [1]
    h_state_vec_ptr,   # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    gs = tl.load(g_scalar_ptr)
    state_rm = tl.load(state_remove_ptr)
    state_upd = tl.load(state_update_ptr)
    for i in range(V):
        acc = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * gs
        h_state_vec_ptr[i] = acc - state_rm + state_upd


# Triton kernel: compute scalar output[b,h] = scale * dot(q_vec[K], h_state_vec[V])
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    scale_ptr,      # float32 [1]
    out_ptr,        # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    scale = tl.load(scale_ptr)
    acc = 0.0
    for j in range(K):
        q_j = tl.load(q_ptr + j)
        for i in range(V):
            h_i = tl.load(h_state_ptr + i)
            acc += q_j * h_i
    tl.store(out_ptr, acc * scale)


# Triton kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
@triton.jit
def write_new_state_kernel(
    h_state_vec_ptr,   # float32 [V]
    new_state_ptr,     # float32 [V*K] flattened
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        val = tl.load(h_state_vec_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, val)


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Shapes from the benchmark (fixed constants)
        B, _, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        num_heads = num_v_heads

        assert num_q_heads == 4
        assert num_k_heads == 4
        assert num_v_heads == 8
        assert K == 128 and V == 128
        assert q.shape[0] == k.shape[0] == v.shape[0] == B
        assert state is not None
        device = q.device

        # Cast inputs to float32 for compute (reference uses .float())
        q_f32 = q.squeeze(1).contiguous().float()   # [B,4,K] -> [B,K]
        k_f32 = k.squeeze(1).contiguous().float()   # [B,4,K] -> [B,K]
        v_f32 = v.squeeze(1).contiguous().float()   # [B,8,V] -> [B,V]
        state_f32 = state.contiguous().float()      # [B,H,V,K] -> [B,V,K]

        A_log_f32 = A_log.contiguous().float()      # [H]
        a_f32 = a.squeeze(1).contiguous().float()   # [B,H]
        dt_bias_f32 = dt_bias.contiguous().float()  # [H]
        b_f32 = b.squeeze(1).contiguous().float()   # [B,H]

        H = a_f32.shape[1]

        # 1) Compute g[b,h] and beta[b,h]
        g_out = torch.empty(B * H, dtype=torch.float32, device=device)
        softplus_and_exp_kernel[(B * H,)](dt_bias_f32, a_f32.view(-1), A_log_f32, g_out, H=H)
        g = g_out.view(B, H)                        # [B,H]

        beta_out = torch.empty(B * H, dtype=torch.float32, device=device)
        sigmoid_kernel[(B * H,)](b_f32.view(-1), beta_out, H=H)
        beta = beta_out.view(B, H)                 # [B,H]

        # Output buffer [B,H]
        output = torch.empty((B, H), dtype=torch.float32, device=device)
        # New state buffer [B,H,V,K]
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # 2) For each (b,h), compute vector h_state and output scalar
        for b_idx in range(B):
            for h_idx in range(H):
                # k_vec: [K], state_mat: [V,K], v_vec: [V]
                k_vec = k_f32[b_idx, h_idx]            # [K]
                state_mat = state_f32[b_idx, h_idx]    # [V,K]
                v_vec = v_f32[b_idx, h_idx]            # [V]

                # Allocate 1-element buffers for scalars
                old_v_buf = torch.empty(1, dtype=torch.float32, device=device)
                state_remove_buf = torch.empty(1, dtype=torch.float32, device=device)
                state_update_buf = torch.empty(1, dtype=torch.float32, device=device)

                # 2.1) old_v = dot(k_vec, state_mat)
                dot_k_state_kernel[(1,)](k_vec, state_mat, old_v_buf, V=V, K=K)

                # 2.2) state_remove = dot(k_vec, g[b,h] * state_mat)
                dot_k_gstate_kernel[(1,)](k_vec, state_mat, g[b_idx, h_idx], state_remove_buf, V=V, K=K)

                # 2.3) state_update = dot(k_vec, beta[b,h] * v_vec + (1 - beta[b,h]) * old_v)
                beta_scalar = torch.empty(1, dtype=torch.float32, device=device)
                beta_scalar[0] = beta[b_idx, h_idx]
                old_v_scalar = torch.empty(1, dtype=torch.float32, device=device)
                old_v_scalar[0] = old_v_buf[0]
                dot_k_newv_kernel[(1,)](k_vec, v_vec, beta_scalar, old_v_scalar, state_update_buf, V=V, K=K)

                # 2.4) h_state_vec[i] = sum_j state_mat[i,j] * g[b,h] - state_remove + state_update
                h_state_vec = torch.empty(V, dtype=torch.float32, device=device)
                h_state_vec_kernel[(1,)](state_mat, g[b_idx, h_idx], state_remove_buf, state_update_buf, h_state_vec, V=V, K=K)

                # 2.5) output[b,h] = scale * dot(q_vec, h_state_vec)
                q_vec = q_f32[b_idx, h_idx]  # [K]
                out_buf = torch.empty(1, dtype=torch.float32, device=device)
                if scale is None or scale == 0.0:
                    # Triton does not have torch.sqrt; compute 1/sqrt(K) in host and pass as scale (or leave default).
                    # Here we assume scale is provided; the harness should pass a valid scale. We use the provided scale.
                    scale_scalar = scale if (scale is not None and scale != 0.0) else (1.0 / (K ** 0.5))
                else:
                    scale_scalar = scale
                scale_buf = torch.empty(1, dtype=torch.float32, device=device)
                scale_buf[0] = scale_scalar
                dot_q_hstate_kernel[(1,)](q_vec, h_state_vec, scale_buf, out_buf, V=V, K=K)
                output[b_idx, h_idx] = out_buf[0]

                # 2.6) write new_state[b,h] as [V,K] broadcasting h_state_vec across K
                new_state_ptr = new_state[b_idx, h_idx]  # flattened [V*K]
                write_new_state_kernel[(V,)](h_state_vec, new_state_ptr, V=V, K=K)

        # Assemble output [B,1,H] bfloat16 and return (output, new_state)
        output_final = output.view(B, 1, H).to(torch.bfloat16)
        return output_final, new_state


def run(*args):
    return ModelNew()(*args)
