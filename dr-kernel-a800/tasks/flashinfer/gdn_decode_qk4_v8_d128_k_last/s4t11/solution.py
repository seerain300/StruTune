import math
import torch
import triton
import triton.language as tl


# Kernel: compute g = exp(-exp(A_log[h]) * softplus(x)), where x = a[b,h] + dt_bias[h]
# We pass x_ptr (per (b,h)) and A_log_ptr (per head). Grid is (B*H,), and we index A_log by (pid % H).
@triton.jit
def softplus_and_exp_kernel(
    x_ptr,          # float32 [B*H], x = a + dt_bias
    A_log_ptr,      # float32 [H]
    g_out_ptr,      # float32 [B*H]
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    x_val = tl.load(x_ptr + pid)
    h = pid % H
    A_log = tl.load(A_log_ptr + h)
    # softplus(x) = log(1 + exp(x))
    s = tl.log(1.0 + tl.exp(x_val))
    e = tl.exp(A_log)
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Kernel: compute beta = sigmoid(b[b,h]) where idx = pid
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


# Kernel: for each (b,h), compute old_v = dot(k_vec, state_mat) where state_mat is [V*K] flattened
# Inputs:
#   k_ptr:       float32 [K]
#   state_ptr:   float32 [V*K]
#   out_ptr:     float32 [1] (we write a single scalar)
@triton.jit
def dot_k_state_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    out_ptr,        # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j
    tl.store(out_ptr, acc)


# Kernel: compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
# Inputs:
#   state_ptr:    float32 [V*K]
#   g_scalar:     float32
#   state_remove: float32
#   state_update: float32
#   h_state_ptr:  float32 [V]
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_scalar,       # float32
    state_remove,   # float32
    state_update,   # float32
    h_state_ptr,    # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        acc = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij
        h_state_i = acc * g_scalar - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_i)


# Kernel: compute output_scalar = scale * dot(q_vec, h_state_vec)
# Inputs:
#   q_ptr:        float32 [K]
#   h_state_ptr:  float32 [V]
#   scale:        float32
#   out_ptr:      float32 [1]
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    scale,          # float32
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
    tl.store(out_ptr, acc * scale)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
# We assume new_state is flattened [B*H*V*K] and we write a single (b,h) slice starting at base.
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [B*H*V*K]
    base,           # int32 base offset for this (b,h)
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        val = tl.load(h_state_ptr + i)
        for j in range(K):
            offset = base + i * K + j
            tl.store(new_state_ptr + offset, val)


def _run_triton_only(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Triton-only implementation of the original run function.
    Returns (output, new_state). output is [B,1,H], new_state is [B,H,V,K].
    """
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

    # Ensure inputs are contiguous and on device
    q = q.to(device=device, dtype=torch.float32).squeeze(1)      # [B,4,K]
    k = k.to(device=device, dtype=torch.float32).squeeze(1)      # [B,4,K]
    v = v.to(device=device, dtype=torch.float32).squeeze(1)      # [B,8,V]
    state = state.to(device=device, dtype=torch.float32)         # [B,H,V,K]

    # Flatten inputs for Triton
    a_flat = a.squeeze(1).reshape(B * H).contiguous().to(device=device, dtype=torch.float32)        # [B*H]
    dt_bias_flat = dt_bias.reshape(H).contiguous().to(device=device, dtype=torch.float32)           # [H]
    b_flat = b.squeeze(1).reshape(B * H).contiguous().to(device=device, dtype=torch.float32)        # [B*H]
    A_log_flat = A_log.reshape(H).contiguous().to(device=device, dtype=torch.float32)               # [H]

    # Precompute x = a + dt_bias on host (PyTorch), then pass to Triton kernel
    x_ptr = a_flat + dt_bias_flat[(torch.arange(B * H) % H).to(device=device)]  # [B*H]
    # Create a proper tensor x_ptr on device
    x_ptr = (a_flat + dt_bias_flat[(torch.arange(B * H)) % H]).to(device=device, dtype=torch.float32)

    # Allocate outputs for g and beta
    g = torch.empty((B * H,), dtype=torch.float32, device=device)
    beta = torch.empty((B * H,), dtype=torch.float32, device=device)

    # Launch kernels to compute g and beta
    grid = (B * H,)
    softplus_and_exp_kernel[grid](x_ptr, A_log_flat, g, H=H)
    sigmoid_kernel[grid](b_flat, beta, H=H)

    # Prepare output vector [B*H] and new_state flattened [B*H*V*K]
    output = torch.empty((B * H,), dtype=torch.float32, device=device)
    new_state = torch.empty((B * H * V * K,), dtype=torch.float32, device=device)

    # For each (b,h), compute dot products and updates using Triton kernels
    for pid in range(B * H):
        # Indices for batch and head
        b_idx = pid // H
        h_idx = pid % H

        # Gather vectors
        q_vec = q[b_idx, h_idx].reshape(K).contiguous()         # [K]
        k_vec = k[b_idx, h_idx].reshape(K).contiguous()         # [K]
        v_vec = v[b_idx, h_idx].reshape(V).contiguous()         # [V]
        state_mat = state[b_idx, h_idx].reshape(V, K).contiguous()  # [V,K]

        # Compute old_v = k @ state
        old_v_buf = torch.empty((1,), dtype=torch.float32, device=device)
        dot_k_state_kernel[(1,)](k_vec, state_mat.reshape(-1), old_v_buf, V=V, K=K)

        # Compute output_scalar = scale * q @ h_state
        # First compute g_val and beta_val
        g_val = g[pid]                # float32 scalar
        beta_val = beta[pid]          # float32 scalar

        # state_remove = dot(k_vec, g * state_mat)
        state_remove_buf = torch.empty((1,), dtype=torch.float32, device=device)
        dot_k_state_kernel[(1,)]((g_val * k_vec).reshape(K), (g_val * state_mat).reshape(-1), state_remove_buf, V=V, K=K)
        # Note: The above line is incorrect. We cannot pass k_vec scaled in this way. Fix by computing on host or use a generic kernel with scale. We will fix by using a proper dot with g*state_mat.
        # Correct approach: compute (g*state_mat) as a tensor and feed to kernel.
        g_state_mat = (g_val * state_mat).reshape(-1)           # [V*K]
        dot_k_state_kernel[(1,)](k_vec, g_state_mat, state_remove_buf, V=V, K=K)

        # state_update = dot(k_vec, new_v_vec) where new_v_vec = beta*v + (1-beta)*old_v
        new_v_vec = (beta_val * v_vec) + ((1.0 - beta_val) * old_v_buf[0])  # [V]
        state_update_buf = torch.empty((1,), dtype=torch.float32, device=device)
        dot_k_state_kernel[(1,)](k_vec, new_v_vec.reshape(-1), state_update_buf, V=V, K=K)

        # h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
        h_state_vec = torch.empty((V,), dtype=torch.float32, device=device)
        h_state_vec_kernel[(1,)](state_mat.reshape(-1), g_val, state_remove_buf[0], state_update_buf[0], h_state_vec, V=V, K=K)

        # output_scalar = scale * dot(q_vec, h_state_vec)
        dot_q_hstate_kernel[(1,)](q_vec, h_state_vec, scale, output + pid, V=V, K=K)

        # Write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
        base = (b_idx * H + h_idx) * (V * K)
        write_new_state_kernel[(1,)](h_state_vec, new_state, base, V=V, K=K)

    # Assemble output as [B,1,H] in bfloat16
    output_expanded = output.view(B, H)[:, None, :]  # [B,1,H]
    output_expanded = output_expanded.to(torch.bfloat16)

    # Return output and new_state
    return output_expanded, new_state.view(B, H, V, K)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure tensors are on the same device and dtype float32 for compute
        device = q.device
        if state.device != device:
            state = state.to(device)
        if A_log.device != device:
            A_log = A_log.to(device)
        if a.device != device:
            a = a.to(device)
        if dt_bias.device != device:
            dt_bias = dt_bias.to(device)
        if b.device != device:
            b = b.to(device)

        output, new_state = _run_triton_only(q, k, v, state, A_log, a, dt_bias, b, scale)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
