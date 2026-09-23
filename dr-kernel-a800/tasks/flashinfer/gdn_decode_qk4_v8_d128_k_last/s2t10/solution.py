import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_full_update_kernel(
    A_log_ptr,           # *float32, shape [H]
    a_ptr,               # *bfloat16, shape [B, 1, H]
    dt_bias_ptr,         # *float32, shape [H]
    b_ptr,               # *bfloat16, shape [B, 1, H]
    q_ptr,               # *bfloat16, shape [B, H, K]
    k_ptr,               # *bfloat16, shape [B, H, K]
    v_ptr,               # *bfloat16, shape [B, H, V]
    state_ptr,           # *float32, shape [B, H, V, K]
    out_ptr,             # *float32, shape [B, H]
    new_state_ptr,       # *float32, shape [B, H, V, K]
    B: tl.constexpr,     # batch size
    H: tl.constexpr,     # number of heads
    V: tl.constexpr,     # K dimension in state/v/v output
    K: tl.constexpr,     # K dimension for q/k
    inv_sqrtK: tl.constexpr,  # 1.0 / sqrt(K), e.g., 1.0 / 11.313708 for K=128
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Compute gate g and beta (float32)
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
    x = a_val + dt_val
    abs_x = tl.abs(x)
    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    A_log_val = tl.load(A_log_ptr + h)  # A_log is [H]
    g_val = tl.exp(-tl.exp(A_log_val) * softplus)
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + h * b_ptr.stride(1)), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Load q, k, v for this (b, h): shape vectors
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_ptr + b * q_ptr.stride(0) + h * q_ptr.stride(1) + j * q_ptr.stride(2)), tl.float32)
        k_elem = tl.cast(tl.load(k_ptr + b * k_ptr.stride(0) + h * k_ptr.stride(1) + j * k_ptr.stride(2)), tl.float32)
        q_vec[j] = q_elem
        k_vec[j] = k_elem

    for v_idx in range(0, V):
        v_elem = tl.cast(tl.load(v_ptr + b * v_ptr.stride(0) + h * v_ptr.stride(1) + v_idx * v_ptr.stride(2)), tl.float32)
        v_vec[v_idx] = v_elem

    # Load state_old: [V, K]
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + b * state_ptr.stride(0) + h * state_ptr.stride(1) + v_idx * state_ptr.stride(2)
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(row_base + k_idx * state_ptr.stride(3)), tl.float32)

    # Compute old_v = k @ (g * state_old)  -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = 0.0
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val
        old_v_vec[j] = tl.sum(sum_val * k_vec[j])  # scalar multiply

    # Compute new_v = beta * v + (1-beta) * old_v
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update as scalars: k @ old_v and k @ new_v
    state_remove = 0.0
    state_update = 0.0
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = h_state * g_val - state_remove + state_update

    # Compute output scalar: output = inv_sqrtK * q_h @ h_state_new
    output_scalar = 0.0
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state_new[:, j]  # [V]
        sum_j = 0.0
        for v_idx in range(0, V):
            sum_j += row_j[v_idx]
        output_scalar += qj * sum_j

    # Store results
    tl.store(out_ptr + b * out_ptr.stride(0) + h * out_ptr.stride(1), output_scalar)

    # Store new_state: [B, H, V, K]
    for v_idx in range(0, V):
        row_base = new_state_ptr + b * new_state_ptr.stride(0) + h * new_state_ptr.stride(1) + v_idx * new_state_ptr.stride(2)
        for k_idx in range(0, K):
            tl.store(row_base + k_idx * new_state_ptr.stride(3), h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward:
        - All computation in Triton kernels.
        - Returns: output as bfloat16 unsqueezed (B, 1, H), new_state as float32 (B, H, V, K).
        """
        # Ensure dtypes: inputs are bfloat16; A_log, a, dt_bias, b are float32.
        # State is float32 as in original.
        assert q.shape[1] == 1 and k.shape[1] == 1 and v.shape[1] == 1, "Only T=1 supported"
        B, T_q, num_q_heads, K = q.shape
        B2, T_k, num_k_heads, K2 = k.shape
        B3, T_v, num_v_heads, V = v.shape
        assert B == B2 == B3, "Batch sizes must match"
        assert T_q == T_k == T_v == 1, "T must be 1"
        assert K == 128 and V == 128, "Fixed K=128, V=128 per original"
        assert state is not None and state.shape == (B, num_v_heads, V, K), "State must be provided"

        # Prepare outputs
        out = torch.empty((B, num_v_heads), dtype=torch.float32, device=q.device)
        new_state = torch.empty_like(state, dtype=torch.float32, device=q.device)

        # Compute inv_sqrtK on host and pass to kernel (no PyTorch ops in host)
        if scale is None or scale == 0.0:
            inv_sqrtK = 1.0 / math.sqrt(K)
        else:
            inv_sqrtK = float(scale) / math.sqrt(K)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * num_v_heads,)
        triton_full_update_kernel[grid](
            A_log, a, dt_bias, b, q, k, v, state, out, new_state,
            B, num_v_heads, V, K, inv_sqrtK,
        )

        # Return outputs as expected: output bfloat16 unsqueezed, new_state float32
        output_bf16 = out.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
