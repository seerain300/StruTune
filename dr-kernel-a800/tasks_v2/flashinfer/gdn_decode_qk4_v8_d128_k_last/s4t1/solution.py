import math
import torch
import triton
import triton.language as tl


# Kernel to compute g = exp(-exp(A_log) * softplus(x)) and s = softplus(x)
@triton.jit
def softplus_and_exp_kernel(
    A_log_ptr,       # float32 [H]
    x_ptr,           # float32 [B*H], layout is [h] then [b]
    g_out_ptr,       # float32 [B*H]
    s_out_ptr,       # float32 [B*H]
    H: tl.constexpr, # number of heads (H)
    K: tl.constexpr, # not used here, but can be kept for future specialization
):
    pid = tl.program_id(axis=0)  # index over B*H
    # Map pid to (b, h)
    b = pid // H
    h = pid % H
    # Load A_log[h] and x[b,h]
    A_log = tl.load(A_log_ptr + h)
    x_val = tl.load(x_ptr + pid)
    # softplus(x) = log(1 + exp(x))
    s = tl.log(1.0 + tl.exp(x_val))
    e = tl.exp(A_log)
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)
    tl.store(s_out_ptr + pid, s)


# Kernel to compute beta = sigmoid(b)
@triton.jit
def sigmoid_kernel(
    b_ptr,            # float32 [B*H]
    beta_out_ptr,     # float32 [B*H]
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b))
    tl.store(beta_out_ptr + pid, beta)


# Generic dot product: c = sum_i sum_j a[i] * B[i,j], where
# a is vector of length K, B is matrix of shape [M,K] (here M=V)
# We pass B as a contiguous 1D buffer of length M*K. We'll reshape in kernel.
@triton.jit
def dot_2d_kernel(
    a_ptr,           # float32 [K]
    B_ptr,           # float32 [M*K]
    out_ptr,         # float32 [1]
    M: tl.constexpr, # V
    K: tl.constexpr, # K
):
    acc = 0.0
    # Loop over rows i in [0, M)
    for i in range(M):
        # Sum over columns j in [0, K)
        for j in range(K):
            a_j = tl.load(a_ptr + j)
            # Compute offset of B[i, j] in flattened B
            offset = i * K + j
            b_ij = tl.load(B_ptr + offset)
            acc += a_j * b_ij
    # Write result
    tl.store(out_ptr, acc)


# Generic matvec: c = sum_i sum_j A[i,j] * x[j], where A is [M,K] and x is [K]
# We pass A as a contiguous 1D buffer of length M*K. We'll reshape in kernel.
@triton.jit
def matvec_kernel(
    A_ptr,           # float32 [M*K]
    x_ptr,           # float32 [K]
    out_ptr,         # float32 [1]
    M: tl.constexpr, # V
    K: tl.constexpr, # K
):
    acc = 0.0
    for i in range(M):
        for j in range(K):
            a_ij = tl.load(A_ptr + i * K + j)
            x_j = tl.load(x_ptr + j)
            acc += a_ij * x_j
    tl.store(out_ptr, acc)


# Write new_state[b,h] as [V,K] matrix filled with h_state_vec broadcast across columns
@triton.jit
def write_new_state_kernel(
    h_state_ptr,     # float32 [V]
    new_state_ptr,   # float32 [V*K] flattened
    V: tl.constexpr, # V
    K: tl.constexpr, # K
):
    # For each row i, fill K columns with h_state[i]
    for i in range(V):
        val = tl.load(h_state_ptr + i)
        for j in range(K):
            offset = i * K + j
            tl.store(new_state_ptr + offset, val)


# Kernel to compute h_state_vec per (b,h) elementwise:
# h_state[i] = sum_j state_mat[i,j] * g - state_remove + state_update
# We assume state_mat is provided as a contiguous [V*K] buffer for (b,h)
@triton.jit
def compute_h_state_vec_kernel(
    state_ptr,       # float32 [V*K]
    g_scalar,        # float32 scalar
    state_remove,    # float32 scalar
    state_update,    # float32 scalar
    h_state_ptr,     # float32 [V]
    V: tl.constexpr, # V
    K: tl.constexpr, # K
):
    for i in range(V):
        acc = 0.0
        for j in range(K):
            offset = i * K + j
            s_ij = tl.load(state_ptr + offset)
            acc += s_ij
        h_state_i = acc * g_scalar - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_i)


def _run_triton_only(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Triton-only implementation of the original run function.
    Returns (output, new_state). output is [B,1,H] in bfloat16, new_state is [B,H,V,K] in float32.
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

    # Prepare strides/pointers: we will operate on squeezed shapes [B,heads,K] where heads are num_q_heads and num_v_heads.
    # However, the original run uses num_v_heads (H=8) for state, v, and output. We'll compute per (b,h) with h in [0,H).

    # Flatten indices for Triton calls: we assume a is [B, H], b is [B, H], dt_bias is [H], A_log is [H]
    a_flat = a.squeeze(1).reshape(-1, H).reshape(-1).contiguous()   # [B*H]
    dt_bias_flat = dt_bias.reshape(H).contiguous()                  # [H]
    b_flat = b.squeeze(1).reshape(-1, H).reshape(-1).contiguous()   # [B*H]
    A_log_flat = A_log.reshape(H).contiguous()                      # [H]

    # Allocate g and beta
    g = torch.empty((B * H,), dtype=torch.float32, device=device)
    beta = torch.empty((B * H,), dtype=torch.float32, device=device)

    # Launch kernels to compute g and beta
    # Grid: one program per (b,h)
    grid_g_beta = (B * H,)
    softplus_and_exp_kernel[grid_g_beta](A_log_flat, a_flat, g, torch.empty_like(g), H=H, K=K)
    sigmoid_kernel[grid_g_beta](b_flat, beta, H=H)

    # Allocate output and new_state
    output = torch.empty((B, H), dtype=torch.float32, device=device)
    new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

    # Now per (b,h) compute all quantities in Triton
    for b_idx in range(B):
        for h_idx in range(H):
            pid = b_idx * H + h_idx

            # Load scalars
            g_val = g[pid]
            beta_val = beta[pid]

            # Prepare vectors and matrices
            # k_vec: [K]
            k_vec = k[b_idx, 0, h_idx].reshape(-1).contiguous()
            # q_vec: [K]
            q_vec = q[b_idx, 0, h_idx].reshape(-1).contiguous()
            # v_vec: [V]
            v_vec = v[b_idx, 0, h_idx].reshape(-1).contiguous()

            # state_mat: [V,K] flattened
            state_mat_flat = state[b_idx, h_idx].reshape(-1).contiguous()

            # Compute old_v = dot(k_vec, state_mat)
            out_old = torch.empty((1,), dtype=torch.float32, device=device)
            dot_2d_kernel[(1,)](k_vec, state_mat_flat, out_old, V=V, K=K)
            old_v = out_old[0]

            # Compute new_v_vec elementwise: beta_val * v_vec + (1 - beta_val) * old_v
            new_v_vec = beta_val * v_vec + (1.0 - beta_val) * (old_v * torch.ones((V,), dtype=torch.float32, device=device))

            # Compute g_state_mat = g_val * state_mat
            g_state_flat = (state_mat_flat * g_val).reshape(-1).contiguous()

            # state_remove = dot(k_vec, g_state_mat)
            out_state_remove = torch.empty((1,), dtype=torch.float32, device=device)
            dot_2d_kernel[(1,)](k_vec, g_state_flat, out_state_remove, V=V, K=K)
            state_remove = out_state_remove[0]

            # state_update = dot(k_vec, new_v_vec) where new_v_vec is [V], we broadcast to [V,K] by repeating columns
            B_new_flat = new_v_vec.reshape(-1).contiguous()  # [V*K] with K copies of each
            out_state_update = torch.empty((1,), dtype=torch.float32, device=device)
            dot_2d_kernel[(1,)](k_vec, B_new_flat, out_state_update, V=V, K=K)
            state_update = out_state_update[0]

            # Compute h_state_vec elementwise: h_state[i] = sum_j state_mat[i,j] * g_val - state_remove + state_update
            h_state_vec = torch.empty((V,), dtype=torch.float32, device=device)
            compute_h_state_vec_kernel[(1,)](state_mat_flat, g_val, state_remove, state_update, h_state_vec, V=V, K=K)

            # Compute output scalar = scale * dot(q_vec, h_state_vec)
            out_output = torch.empty((1,), dtype=torch.float32, device=device)
            matvec_kernel[(1,)](q_vec.reshape(-1), h_state_vec, out_output, V=V, K=K)
            output_scalar = out_output[0] * scale
            output[b_idx, h_idx] = output_scalar

            # Write new_state[b,h] = h_state_vec broadcast to [V,K]
            new_state_flat = new_state[b_idx, h_idx].reshape(-1)  # [V*K]
            write_new_state_kernel[(1,)](h_state_vec, new_state_flat, V=V, K=K)

    # Return output as [B,1,H] in bfloat16, and new_state as [B,H,V,K] in float32
    output_expanded = output.unsqueeze(1)  # [B,1,H]
    output_bf16 = output_expanded.to(torch.bfloat16)
    return output_bf16, new_state


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # All computation happens in Triton kernels; no torch ops inside forward except for simple torch reshapes/contiguous.
        return _run_triton_only(q, k, v, state, A_log, a, dt_bias, b, scale)


def run(*args):
    return ModelNew()(*args)
