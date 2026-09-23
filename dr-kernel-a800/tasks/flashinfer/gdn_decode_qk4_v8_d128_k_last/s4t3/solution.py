import math
import torch
import triton
import triton.language as tl


# Kernel: compute g = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
# softplus(x) = log(1 + exp(x))
@triton.jit
def softplus_and_exp_kernel(
    A_log_ptr,      # float32 [H]
    a_ptr,          # float32 [B*H]
    dt_bias_ptr,    # float32 [B*H] (we will index with b_idx)
    g_out_ptr,      # float32 [B*H]
    B: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    b = pid // H
    h = pid % H
    A_log = tl.load(A_log_ptr + h)
    a_val = tl.load(a_ptr + pid)
    db_val = tl.load(dt_bias_ptr + b * H + h)
    x = a_val + db_val
    s = tl.log(1.0 + tl.exp(x))  # softplus(x)
    e = tl.exp(A_log)
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Kernel: beta = sigmoid(b) per (b,h)
@triton.jit
def sigmoid_kernel(
    b_ptr,          # float32 [B*H]
    beta_out_ptr,   # float32 [B*H]
):
    pid = tl.program_id(axis=0)
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Kernel: compute old_v = dot(k_vec, state_mat) where
# k_vec is [K], state_mat is [V,K]
@triton.jit
def dot_k_state_kernel(
    k_ptr,          # float32 [K], k[b,h]
    state_ptr,      # float32 [V*K], flattened state[b,h]
    old_v_out_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j
    tl.store(old_v_out_ptr, acc)


# Kernel: compute state_remove = dot(k_vec, g * state_mat)
@triton.jit
def dot_k_gstate_kernel(
    g_scalar,       # float32
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    state_remove_out_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            g_state_ij = g_scalar * state_ij
            acc += g_state_ij * k_j
    tl.store(state_remove_out_ptr, acc)


# Kernel: compute state_update = dot(k_vec, new_v_vec), new_v_vec = beta * v_vec + (1 - beta) * old_v
@triton.jit
def dot_k_newv_kernel(
    beta_scalar,    # float32
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    old_v,          # float32
    state_update_out_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    c = 1.0 - beta_scalar
    acc = 0.0
    for i in range(V):
        v_i = tl.load(v_ptr + i)
        new_v_i = beta_scalar * v_i + c * old_v
        for j in range(K):
            k_j = tl.load(k_ptr + j)
            acc += new_v_i * k_j
    tl.store(state_update_out_ptr, acc)


# Kernel: compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    g_scalar,        # float32
    state_remove,    # float32
    state_update,    # float32
    state_ptr,       # float32 [V*K]
    h_state_ptr,     # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        acc = 0.0
        for j in range(K):
            s_ij = tl.load(state_ptr + i * K + j)
            acc += s_ij
        h_state_i = acc * g_scalar - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_i)


# Kernel: compute output_scalar = scale * dot(q_vec, h_state_vec)
@triton.jit
def dot_q_hstate_kernel(
    scale,           # float32
    q_ptr,           # float32 [K]
    h_state_ptr,     # float32 [V]
    output_ptr,      # float32 [B*H]
    V: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    acc_q = 0.0
    for j in range(K):
        q_j = tl.load(q_ptr + j)
        acc_q += q_j
    acc_hs = 0.0
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        acc_hs += h_i
    out = acc_q * acc_hs * scale
    tl.store(output_ptr + pid, out)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,     # float32 [V]
    new_state_ptr,   # float32 [B*H*V*K] flattened
    V: tl.constexpr,
    K: tl.constexpr,
    base_offset: tl.constexpr,  # equals b*H*V + h*V
):
    for i in range(V):
        val = tl.load(h_state_ptr + i)
        for j in range(K):
            offset = base_offset + i * K + j
            tl.store(new_state_ptr + offset, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original run function.
        Returns:
          - output: [B,1,H] in bfloat16
          - new_state: [B,H,V,K] in float32
        """
        # Shapes
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        device = q.device
        H = num_v_heads
        assert T == 1
        assert num_q_heads == 4
        assert num_k_heads == 4
        assert num_v_heads == 8
        assert K == 128 and V == 128

        # Flatten a and b to [B*H]
        a_flat = a.squeeze(1).reshape(B * H).contiguous()                 # [B*H], dtype bfloat16 or float32
        dt_bias = dt_bias.reshape(B * H).contiguous()                    # [B*H], dtype float32 (or bfloat16; cast to float32)
        b_flat = b.squeeze(1).reshape(B * H).contiguous()                # [B*H], dtype bfloat16 or float32

        # A_log is [H], ensure float32
        A_log = A_log.reshape(H).contiguous()                           # [H], dtype float32

        # Allocate tensors for g and beta in float32
        g = torch.empty((B * H,), dtype=torch.float32, device=device)    # [B*H]
        beta = torch.empty((B * H,), dtype=torch.float32, device=device) # [B*H]

        # Launch kernels to compute g and beta
        grid_g_beta = (B * H,)
        # Cast inputs to float32 for kernels
        a_in = a_flat.to(torch.float32)
        b_in = b_flat.to(torch.float32)
        dt_bias_in = dt_bias.to(torch.float32)
        softplus_and_exp_kernel[grid_g_beta](A_log, a_in, dt_bias_in, g, B=B, H=H)
        sigmoid_kernel[grid_g_beta](b_in, beta)

        # Allocate outputs and new_state
        output = torch.empty((B * H,), dtype=torch.float32, device=device)  # [B*H]
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)  # [B,H,V,K]

        # For each (b,h), compute old_v, state_remove, state_update, h_state_vec, output[b,h], and fill new_state[b,h]
        for b_idx in range(B):
            for h_idx in range(H):
                pid = b_idx * H + h_idx
                base = b_idx * H * V + h_idx * V

                # Prepare vectors/ matrices (float32)
                q_vec = q[b_idx, h_idx].to(torch.float32).contiguous()    # [K]
                k_vec = k[b_idx, h_idx].to(torch.float32).contiguous()    # [K]
                v_vec = v[b_idx, h_idx].to(torch.float32).contiguous()    # [V]
                state_mat = state[b_idx, h_idx].contiguous()              # [V,K], float32

                # old_v = dot(k_vec, state_mat)
                old_v_buf = torch.empty((1,), dtype=torch.float32, device=device)
                dot_k_state_kernel[(1,)](k_vec, state_mat, old_v_buf, V=V, K=K)

                # g_scalar and beta_scalar from precomputed g and beta
                g_val = g[pid]                                             # float32 scalar
                beta_val = beta[pid]                                      # float32 scalar

                # state_remove = dot(k_vec, g * state_mat)
                state_remove_buf = torch.empty((1,), dtype=torch.float32, device=device)
                dot_k_gstate_kernel[(1,)](g_val, k_vec, state_mat, state_remove_buf, V=V, K=K)

                # state_update = dot(k_vec, new_v_vec) where new_v_vec = beta*v + (1-beta)*old_v
                new_v_vec = beta_val * v_vec + (1.0 - beta_val) * old_v_buf[0]
                state_update_buf = torch.empty((1,), dtype=torch.float32, device=device)
                dot_k_newv_kernel[(1,)](beta_val, k_vec, v_vec, old_v_buf[0], state_update_buf, V=V, K=K)

                # h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
                h_state_vec = torch.empty((V,), dtype=torch.float32, device=device)
                h_state_vec_kernel[(1,)](g_val, state_remove_buf[0], state_update_buf[0], state_mat, h_state_vec, V=V, K=K)

                # output_scalar[b,h] = scale * dot(q_vec, h_state_vec)
                dot_q_hstate_kernel[(1,)](scale, q_vec, h_state_vec, output, V=V, K=K)

                # write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
                write_new_state_kernel[(1,)](h_state_vec, new_state, V=V, K=K, base_offset=base)

        # output: [B,1,H] bfloat16
        output_expanded = output.view(B, H).unsqueeze(1).to(torch.bfloat16)
        return output_expanded, new_state


def run(*args):
    return ModelNew()(*args)
