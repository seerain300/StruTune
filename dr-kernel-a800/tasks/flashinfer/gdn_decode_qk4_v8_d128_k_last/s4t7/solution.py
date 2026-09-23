import math
import torch
import triton
import triton.language as tl


# Kernel: compute g = exp(-exp(A_log[h]) * softplus(x)) where x = a[b,h] + dt_bias[h]
# Inputs: A_log_flat [H], x_flat [B*H], g_out [B*H]
@triton.jit
def softplus_and_exp_kernel(
    A_log_ptr,        # float32 [H]
    x_ptr,            # float32 [B*H]
    g_out_ptr,        # float32 [B*H]
    H: tl.constexpr,  # number of heads (H)
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    # Recover b and h: b = pid // H, h = pid % H
    b = pid // H
    h = pid % H
    # x value for this (b,h)
    x_val = tl.load(x_ptr + pid)
    # A_log for this head
    A_log_val = tl.load(A_log_ptr + h)
    # softplus(x) = log(1 + exp(x))
    s = tl.log(1.0 + tl.exp(x_val))
    # e = exp(A_log[h])
    e = tl.exp(A_log_val)
    # g = exp(-e * s)
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Kernel: compute beta = sigmoid(b) for each (b,h): beta = 1 / (1 + exp(-b))
# Inputs: b_flat [B*H], beta_out [B*H]
@triton.jit
def sigmoid_kernel(
    b_ptr,            # float32 [B*H]
    beta_out_ptr,     # float32 [B*H]
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Kernel: compute state_remove = dot(k_vec, g * state_mat) -> scalar
# Inputs: g_scalar (float32), k_ptr [K], state_ptr [V*K], out_ptr [1]
# We loop over K, for each j load k[j], then loop over V and accumulate state[i*K + j] * g
@triton.jit
def dot_k_state_kernel(
    g_scalar,         # float32 scalar
    k_ptr,            # float32 [K]
    state_ptr,        # float32 [V*K]
    out_ptr,          # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    # For each column j in K
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        # Accumulate over rows i in V
        for i in range(V):
            offset = i * K + j
            state_ij = tl.load(state_ptr + offset)
            acc += state_ij * g_scalar
    tl.store(out_ptr, acc)


# Kernel: compute state_update = dot(k_vec, new_v_vec) -> scalar
# Inputs: beta_scalar (float32), k_ptr [K], v_ptr [V], old_v_scalar (float32), out_ptr [1]
@triton.jit
def dot_k_newv_kernel(
    beta_scalar,      # float32
    k_ptr,            # float32 [K]
    v_ptr,            # float32 [V]
    old_v_scalar,     # float32
    out_ptr,          # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        # new_v_vec[j] = beta * v[j] + (1 - beta) * old_v
        v_j = tl.load(v_ptr + j)
        new_v_j = beta_scalar * v_j + (1.0 - beta_scalar) * old_v_scalar
        acc += k_j * new_v_j
    tl.store(out_ptr, acc)


# Kernel: compute h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update, for i in [0..V-1]
# Inputs: g_scalar, state_remove_scalar, state_update_scalar, state_ptr [V*K], h_state_ptr [V]
@triton.jit
def h_state_vec_kernel(
    g_scalar,         # float32
    state_remove_scalar,   # float32
    state_update_scalar,   # float32
    state_ptr,       # float32 [V*K]
    h_state_ptr,     # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        acc = 0.0
        for j in range(K):
            offset = i * K + j
            s_ij = tl.load(state_ptr + offset)
            acc += s_ij
        h_state_i = acc * g_scalar - state_remove_scalar + state_update_scalar
        tl.store(h_state_ptr + i, h_state_i)


# Kernel: compute output_scalar = scale * dot(q_vec, h_state_vec)
# Inputs: scale (float32), q_ptr [K], h_state_ptr [V], out_ptr [1]
@triton.jit
def dot_q_hstate_kernel(
    scale,            # float32
    q_ptr,            # float32 [K]
    h_state_ptr,      # float32 [V]
    out_ptr,          # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        for j in range(K):
            q_j = tl.load(q_ptr + j)
            acc += q_j * h_i
    acc = acc * scale
    tl.store(out_ptr, acc)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
# Inputs: h_state_ptr [V], new_state_ptr [V*K], V, K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,      # float32 [V]
    new_state_ptr,    # float32 [V*K]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        val = tl.load(h_state_ptr + i)
        for j in range(K):
            offset = i * K + j
            tl.store(new_state_ptr + offset, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original run function.
        Returns:
        - output: [B, 1, H] in bfloat16
        - new_state: [B, H, V, K] in float32
        """
        # Shapes
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        device = q.device

        assert T == 1, "T must be 1"
        assert num_q_heads == 4, "num_q_heads must be 4"
        assert num_k_heads == 4, "num_k_heads must be 4"
        assert num_v_heads == 8, "num_v_heads must be 8"
        assert K == 128 and V == 128, "K and V must be 128"

        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(K)

        # Flatten and ensure contiguity for Triton
        H = num_v_heads
        # a: [B,1,H] -> [B*H]
        a_flat = a.squeeze(1).reshape(-1, H).reshape(-1).contiguous()   # [B*H]
        # dt_bias: [H]
        dt_bias_flat = dt_bias.reshape(H).contiguous()
        # b: [B,1,H] -> [B*H]
        b_flat = b.squeeze(1).reshape(-1, H).reshape(-1).contiguous()
        # A_log: [H]
        A_log_flat = A_log.reshape(H).contiguous()

        # Allocate outputs for g and beta
        g = torch.empty((B * H,), dtype=torch.float32, device=device)
        beta = torch.empty((B * H,), dtype=torch.float32, device=device)

        # Launch kernels to compute g and beta
        grid = (B * H,)
        softplus_and_exp_kernel[grid](A_log_flat, a_flat, g, H=H)
        sigmoid_kernel[grid](b_flat, beta, H=H)

        # Prepare output buffer and new_state
        output = torch.empty((B * H,), dtype=torch.float32, device=device)  # [B*H]
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # For each (b,h), compute everything in Triton
        for pid in range(B * H):
            b_idx = pid // H
            h_idx = pid % H

            # Load vectors
            q_vec = q[b_idx, 0, h_idx].contiguous().float()   # [K]
            k_vec = k[b_idx, 0, h_idx].contiguous().float()   # [K]
            v_vec = v[b_idx, 0, h_idx].contiguous().float()   # [V]
            state_mat = state[b_idx, h_idx].contiguous().float()  # [V, K], contiguous row-major

            # Prepare pointers
            q_ptr = q_vec
            k_ptr = k_vec
            v_ptr = v_vec
            state_ptr = state_mat

            # Load scalars
            g_val = g[pid]
            beta_val = beta[pid]

            # state_remove = dot(k, g * state)
            state_remove_buf = torch.empty((1,), dtype=torch.float32, device=device)
            dot_k_state_kernel[(1,)](g_val, k_ptr, state_ptr, state_remove_buf, V=V, K=K)

            # state_update = dot(k, new_v_vec) where new_v_vec = beta*v + (1-beta)*old_v
            old_v = torch.dot(k_ptr, state_ptr)  # scalar tensor from torch
            new_v_vec = beta_val * v_ptr + (1.0 - beta_val) * old_v
            state_update_buf = torch.empty((1,), dtype=torch.float32, device=device)
            dot_k_newv_kernel[(1,)](beta_val, k_ptr, v_ptr, old_v.item(), state_update_buf, V=V, K=K)

            # h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
            h_state_vec = torch.empty((V,), dtype=torch.float32, device=device)
            h_state_vec_kernel[(1,)](g_val, state_remove_buf[0], state_update_buf[0], state_ptr, h_state_vec, V=V, K=K)

            # output_scalar = scale * dot(q, h_state_vec)
            dot_q_hstate_kernel[(1,)](scale, q_ptr, h_state_vec, output + pid, V=V, K=K)

            # write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
            # Flatten new_state storage pointer for this (b,h): start at b*H*V*K + h*V*K
            base = b_idx * H * V * K + h_idx * V * K
            write_new_state_kernel[(1,)](h_state_vec, new_state.view(-1), V=V, K=K)

        # Return output as [B,1,H] bfloat16 and new_state as [B,H,V,K] float32
        output_expanded = output.view(B, H).unsqueeze(1)  # [B,1,H]
        output_expanded = output_expanded.to(torch.bfloat16)
        return output_expanded, new_state


def run(*args):
    return ModelNew()(*args)
