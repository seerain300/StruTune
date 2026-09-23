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


# Kernel: compute old_v = dot(k_vec, state_mat) where state_mat is flattened [V*K]
# Inputs:
#   k_ptr:          float32 [K]
#   state_ptr:      float32 [V*K] (row-major: row i starts at i*K)
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
    k_vec = tl.load(k_ptr)  # load entire k vector as a 1D vector
    for j in range(K):
        k_j = k_vec[j]
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j
    tl.store(old_v_ptr, acc)


# Kernel: compute state_remove = dot(k_vec, g_scalar * state_mat)
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,           # float32 [K]
    state_ptr,       # float32 [V*K]
    g_scalar_ptr,    # float32 [1] (scalar)
    state_remove_ptr,# float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    gs = tl.load(g_scalar_ptr)  # scalar g
    k_vec = tl.load(k_ptr)
    for j in range(K):
        k_j = k_vec[j]
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * gs * k_j
    tl.store(state_remove_ptr, acc)


# Kernel: compute state_update = dot(k_vec, beta_scalar * v_vec + (1 - beta_scalar) * old_v_scalar)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,             # float32 [K]
    v_ptr,             # float32 [V]
    beta_scalar_ptr,   # float32 [1]
    old_v_scalar_ptr,  # float32 [1]
    state_update_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    beta = tl.load(beta_scalar_ptr)
    old_v = tl.load(old_v_scalar_ptr)
    k_vec = tl.load(k_ptr)
    for j in range(K):
        k_j = k_vec[j]
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            new_v_i = beta * v_i + (1.0 - beta) * old_v
            acc += new_v_i * k_j
    tl.store(state_update_ptr, acc)


# Kernel: compute h_state_vec[i] = (sum_j state[i,j] * g_scalar) - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    state_ptr,        # float32 [V*K]
    g_scalar_ptr,     # float32 [1]
    state_remove_ptr, # float32 [1]
    state_update_ptr, # float32 [1]
    h_state_ptr,      # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    gs = tl.load(g_scalar_ptr)
    state_rem = tl.load(state_remove_ptr)
    state_upd = tl.load(state_update_ptr)
    for i in range(V):
        acc = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * gs
        h_val = acc - state_rem + state_upd
        tl.store(h_state_ptr + i, h_val)


# Kernel: compute output_scalar = scale * dot(q_vec, h_state_vec)
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,            # float32 [V]
    h_state_ptr,      # float32 [V]
    scale_ptr,        # float32 [1]
    out_ptr,          # float32 [1]
    V: tl.constexpr,
):
    scale = tl.load(scale_ptr)
    acc = 0.0
    for i in range(V):
        q_i = tl.load(q_ptr + i)
        h_i = tl.load(h_state_ptr + i)
        acc += q_i * h_i
    tl.store(out_ptr, acc * scale)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,      # float32 [V]
    new_state_ptr,    # float32 [V*K]
    V: tl.constexpr,
    K: tl.constexpr,
):
    i = tl.program_id(axis=0)  # one program per row i
    for j in range(K):
        val = tl.load(h_state_ptr + i)
        tl.store(new_state_ptr + i * K + j, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B,1,4,128], bfloat16
        k: [B,1,4,128], bfloat16
        v: [B,1,8,128], bfloat16
        state: [B,8,128,128], float32 (k-last)
        A_log: [8], float32
        a: [B,1,8], bfloat16
        dt_bias: [8], float32
        b: [B,1,8], bfloat16
        scale: float or None
        returns: (output [B,1,8] bfloat16, new_state [B,8,128,128] float32)
        """
        # Shapes
        B_q, _, num_q_heads, K_q = q.shape
        B_k, _, num_k_heads, K_k = k.shape
        B_v, _, num_v_heads, V_v = v.shape
        B, H, V, K = state.shape
        assert B_q == B_k == B_v == B
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8
        assert K_q == K_k == K == 128 and V_v == V == 128

        # Make tensors contiguous and float32 for kernels
        q32 = q.squeeze(1).contiguous().to(torch.float32)   # [B,4,128]
        k32 = k.squeeze(1).contiguous().to(torch.float32)   # [B,4,128]
        v32 = v.squeeze(1).contiguous().to(torch.float32)   # [B,8,128]
        state32 = state.contiguous().to(torch.float32)      # [B,H,V,K]
        A_log32 = A_log.contiguous().to(torch.float32)      # [H]
        a32 = a.squeeze(1).contiguous().to(torch.float32)   # [B,H]
        dt_bias32 = dt_bias.contiguous().to(torch.float32)  # [H]
        b32 = b.squeeze(1).contiguous().to(torch.float32)   # [B,H]

        # Allocate outputs for g and beta
        g = torch.empty(B * H, dtype=torch.float32, device=q.device)
        beta = torch.empty(B * H, dtype=torch.float32, device=q.device)

        # Launch softplus_and_exp_kernel
        grid_g = (B * H,)
        softplus_and_exp_kernel[grid_g](dt_bias32, a32, A_log32, g, H=H)

        # Launch sigmoid_kernel
        grid_beta = (B * H,)
        sigmoid_kernel[grid_beta](b32, beta, H=H)

        # Initialize output and new_state
        output = torch.empty(B, H, dtype=torch.float32, device=q.device)  # will be [B,1,H] later
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=q.device)

        # Process each (b,h)
        for b_idx in range(B):
            for h_idx in range(H):
                # Extract vectors
                q_vec = q32[b_idx, h_idx]                      # [K]
                k_vec = k32[b_idx, h_idx]                      # [K]
                v_vec = v32[b_idx, h_idx]                      # [V]
                state_mat = state32[b_idx, h_idx]              # [V,K]

                # Compute g_bh and beta_bh via scalar kernels
                g_bh_buf = torch.empty(1, dtype=torch.float32, device=q.device)
                softplus_and_exp_kernel[(1,)](dt_bias32[h_idx], a32[b_idx, h_idx], A_log32[h_idx], g_bh_buf, H=1)  # scalar
                g_bh = g_bh_buf[0]
                beta_bh_buf = torch.empty(1, dtype=torch.float32, device=q.device)
                sigmoid_kernel[(1,)](b32[b_idx, h_idx], beta_bh_buf, H=1)
                beta_bh = beta_bh_buf[0]

                # 1) old_v = dot(k_vec, state_mat)
                old_v_buf = torch.empty(1, dtype=torch.float32, device=q.device)
                dot_k_state_kernel[(1,)](k_vec, state_mat, old_v_buf, V=V, K=K)

                # 2) state_remove = dot(k_vec, g_bh * state_mat)
                state_remove_buf = torch.empty(1, dtype=torch.float32, device=q.device)
                g_scalar_buf = torch.empty(1, dtype=torch.float32, device=q.device)
                g_scalar_buf[0] = g_bh
                dot_k_gstate_kernel[(1,)](k_vec, state_mat, g_scalar_buf, state_remove_buf, V=V, K=K)

                # 3) state_update = dot(k_vec, beta_bh * v_vec + (1 - beta_bh) * old_v)
                old_v_scalar = old_v_buf[0]
                state_update_buf = torch.empty(1, dtype=torch.float32, device=q.device)
                beta_scalar_buf = torch.empty(1, dtype=torch.float32, device=q.device)
                beta_scalar_buf[0] = beta_bh
                dot_k_newv_kernel[(1,)](k_vec, v_vec, beta_scalar_buf, old_v_scalar, state_update_buf, V=V, K=K)

                # 4) h_state_vec[i] = sum_j state[i,j] * g_bh - state_remove + state_update
                h_state_vec = torch.empty(V, dtype=torch.float32, device=q.device)
                g_scalar_buf[0] = g_bh
                h_state_vec_kernel[(1,)](state_mat, g_scalar_buf, state_remove_buf, state_update_buf, h_state_vec, V=V, K=K)

                # 5) output_scalar[b,h] = scale * dot(q_vec, h_state_vec)
                out_buf = torch.empty(1, dtype=torch.float32, device=q.device)
                scale_scalar = 1.0 / math.sqrt(K) if (scale is None or scale == 0.0) else scale
                scale_buf = torch.empty(1, dtype=torch.float32, device=q.device)
                scale_buf[0] = scale_scalar
                dot_q_hstate_kernel[(1,)](q_vec, h_state_vec, scale_buf, out_buf, V=V)
                output[b_idx, h_idx] = out_buf[0]

                # 6) write new_state[b,h] as [V,K] broadcasting h_state_vec across K
                write_new_state_kernel[(V,)](h_state_vec, new_state[b_idx, h_idx], V=V, K=K)

        # Assemble output [B,1,H] bfloat16
        output_final = output.view(B, 1, H).to(torch.bfloat16)

        return output_final, new_state


def run(*args):
    return ModelNew()(*args)
