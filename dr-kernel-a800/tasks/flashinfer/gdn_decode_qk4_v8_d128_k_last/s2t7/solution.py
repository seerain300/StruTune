import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, h] and dt_bias[h]
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)  # A_log is [H]
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + h * b_ptr.stride(1)), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H] layout
    tl.store(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1), g_val)
    tl.store(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1), beta_val)


@triton.jit
def triton_update_and_output_kernel(
    q_ptr,             # *bfloat16, shape [B, H, K]
    k_ptr,             # *bfloat16, shape [B, H, K]
    v_ptr,             # *bfloat16, shape [B, H, V]
    state_ptr,         # *float32,  shape [B, H, V, K]
    g_ptr,             # *float32,  shape [B, 1, H]  (we index as [B,H])
    beta_ptr,          # *float32,  shape [B, 1, H]
    output_ptr,        # *float32,  shape [B, H]
    new_state_ptr,     # *float32,  shape [B, H, V, K]
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load parameters
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1))  # scalar
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1))  # scalar

    # Load q[b, h, :], k[b, h, :], v[b, h, :]
    q_base = q_ptr + b * q_ptr.stride(0) + h * q_ptr.stride(1)
    k_base = k_ptr + b * k_ptr.stride(0) + h * k_ptr.stride(1)
    v_base = v_ptr + b * v_ptr.stride(0) + h * v_ptr.stride(1)

    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        qj = tl.cast(tl.load(q_base + j), tl.float32)
        kj = tl.cast(tl.load(k_base + j), tl.float32)
        q_vec[j] = qj
        k_vec[j] = kj
    for j in range(0, V):
        v_vec[j] = tl.cast(tl.load(v_base + j), tl.float32)

    # Load state_old: [V, K]
    state_base = state_ptr + b * state_ptr.stride(0) + h * state_ptr.stride(1)
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_base + v_idx * state_ptr.stride(2)
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(row_base + k_idx), tl.float32)

    # Compute old_v = k @ (g * state_old) -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = 0.0
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val
        old_v_vec[j] = sum_val

    # Compute new_v = beta * v + (1 - beta) * old_v
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update as scalars: k @ old_v and k @ new_v
    state_remove = 0.0
    state_update = 0.0
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = h_state * g_val - state_remove + state_update

    # Compute output scalar: output = (1/sqrt(K)) * q_h @ h_state_new
    inv_sqrtK = 1.0 / tl.sqrt(K)
    output_scalar = 0.0
    for j in range(0, K):
        qj = q_vec[j]
        # Sum over V: q_h @ h_state_new equals sum_j q[j] * sum over V of h_state_new[:, j]
        col_j_sum = 0.0
        for v_idx in range(0, V):
            col_j_sum += h_state_new[v_idx, j]
        output_scalar += qj * col_j_sum

    tl.store(output_ptr + b * output_ptr.stride(0) + h * output_ptr.stride(1), output_scalar)

    # Write new_state[b, h, :, :] = h_state_new
    new_state_base = new_state_ptr + b * new_state_ptr.stride(0) + h * new_state_ptr.stride(1)
    for v_idx in range(0, V):
        row_base = new_state_base + v_idx * new_state_ptr.stride(2)
        for k_idx in range(0, K):
            tl.store(row_base + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation:
        - gate_beta_kernel computes g and beta per (b, h)
        - update_and_output_kernel computes per-(b, h) state update and output scalar, including 1/sqrt(K) inside kernel
        Returns:
        - output: bfloat16, shape (B, 1, H), unsqueezed (1) added by caller
        - new_state: float32, shape (B, H, V, K)
        """
        # Ensure inputs have expected dims (the original run asserts fixed dims, we mimic that)
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q,k,v must be [B,1,H,K]"
        assert state.dim() == 4, "state must be [B,H,V,K]"
        B, T_q, num_q_heads, K = q.shape
        Bk, Tk, num_k_heads, _ = k.shape
        Bv, Tv, num_v_heads, V = v.shape
        Bst, H, Vst, Kst = state.shape
        assert B == Bk == Bv == Bst, "Batch mismatch"
        assert T_q == 1 and Tk == 1 and Tv == 1, "T must be 1"
        # Original run asserts num_q_heads == 4, num_k_heads == 4, num_v_heads == 8, K == 128, V == 128
        # We keep these assertions to match the reference behavior; the benchmark axes vary but these must hold for correctness.
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8, "Head counts must match"
        assert K == 128 and V == 128 and Kst == 128 and Vst == 128, "Dimensions must be 128"

        device = q.device

        # Ensure contiguous tensors
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        state_c = state.contiguous()
        A_log_c = A_log.contiguous()
        a_c = a.contiguous()
        dt_bias_c = dt_bias.contiguous()
        b_c = b.contiguous()

        # Allocate g and beta as float32 (B, 1, H)
        g = torch.empty((B, 1, num_v_heads), dtype=torch.float32, device=device)
        beta = torch.empty((B, 1, num_v_heads), dtype=torch.float32, device=device)

        # Launch gate_beta_kernel
        grid1 = (B * num_v_heads,)
        triton_gate_beta_kernel[grid1](
            A_log_c, a_c, dt_bias_c, b_c, g, beta, B, num_v_heads
        )

        # Allocate outputs
        output = torch.empty((B, num_v_heads), dtype=torch.float32, device=device)
        new_state = torch.empty_like(state_c, dtype=torch.float32, device=device)

        # Launch update_and_output_kernel
        grid2 = (B * num_v_heads,)
        triton_update_and_output_kernel[grid2](
            q_c, k_c, v_c, state_c, g, beta, output, new_state, B, num_v_heads, V, K
        )

        # Return as expected: output as bfloat16 unsqueezed, new_state as float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
