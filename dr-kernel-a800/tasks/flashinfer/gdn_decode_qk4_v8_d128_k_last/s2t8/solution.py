import math
import torch
import triton
import triton.language as tl


# Kernel 1: compute gating parameters g and beta per (b, h)
@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16, shape [B, 1, H]
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H]
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size (compile-time for grid, runtime indexing fine)
    H: tl.constexpr,   # number of heads
):
    pid = tl.program_id(axis=0)  # one program per (b, h)
    b = pid // H
    h = pid % H

    # Load scalars
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H], contiguous
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + h * b_ptr.stride(1)), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H] layout
    tl.store(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1), g_val)
    tl.store(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1), beta_val)


# Kernel 2: per-(b, h) update and output computation
@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16, shape [B, H, K]
    k_ptr,             # *bfloat16, shape [B, H, K]
    v_ptr,             # *bfloat16, shape [B, H, V]
    state_ptr,         # *float32, shape [B, H, V, K]
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    output_ptr,        # *float32, shape [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    scale_over_sqrtK: tl.constexpr,  # scalar float32
):
    pid = tl.program_id(axis=0)  # one program per (b, h)
    b = pid // H
    h = pid % H

    # Load q[b, h, :], k[b, h, :], v[b, h, :]
    q_base = q_ptr + b * q_ptr.stride(0) + h * q_ptr.stride(1)
    k_base = k_ptr + b * k_ptr.stride(0) + h * k_ptr.stride(1)
    v_base = v_ptr + b * v_ptr.stride(0) + h * v_ptr.stride(1)

    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_base + j), tl.float32)
        k_elem = tl.cast(tl.load(k_base + j), tl.float32)
        q_vec[j] = q_elem
        k_vec[j] = k_elem

    for v_idx in range(0, V):
        v_elem = tl.cast(tl.load(v_base + v_idx), tl.float32)
        v_vec[v_idx] = v_elem

    # Load old_state = state[b, h, :, :] as [V, K] float32
    state_base = state_ptr + b * state_ptr.stride(0) + h * state_ptr.stride(1)  # points to [V, K] slice
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_base + v_idx * K  # each row is contiguous of length K
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(row_base + k_idx), tl.float32)

    # Gate scalar
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1))

    # Compute old_v = k @ (g * old_state) -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = 0.0
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val
        old_v_vec[j] = sum_val

    # new_v = beta * v + (1 - beta) * old_v -> (V,)
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1))
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update as scalars
    state_remove = 0.0
    state_update = 0.0
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = h_state * g_val - state_remove + state_update  # broadcast scalar

    # Compute output scalar: output = scale_over_sqrtK * (q_h @ h_state_new)
    output_val = 0.0
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state_new[:, j]  # [V]
        sum_j = 0.0
        for v_idx in range(0, V):
            sum_j += row_j[v_idx]
        output_val += qj * sum_j

    output_val = scale_over_sqrtK * output_val

    # Store output
    tl.store(output_ptr + b * output_ptr.stride(0) + h * output_ptr.stride(1), output_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version of the original run function.
        - All computation is done in Triton kernels.
        - Returns output as bfloat16 (unsqueezed to [B, 1, H]) and new_state as float32 [B, H, V, K].
        """
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q, k, v must be 4D tensors"
        assert state.dim() == 4, "state must be 4D [B, H, V, K]"
        B, Tq, H_q, K = q.shape
        Bk, Tk, H_k, K_k = k.shape
        Bv, Tv, H_v, V = v.shape
        Bst, Hst, Vst, Kst = state.shape
        assert Tq == 1 and Tk == 1 and Tv == 1, "T must be 1"
        assert H_q == 4 and H_k == 4 and H_v == 8, "Head sizes must match the original expectations"
        assert K == 128 and V == 128 and Kst == 128 and Vst == 128, "K and V must be 128"
        assert B == Bst and Hst == (H_v // 2), "Batch and head compatibility expected"

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        a = a.contiguous() if a is not None else a
        b = b.contiguous() if b is not None else b
        dt_bias = dt_bias.contiguous()
        A_log = A_log.contiguous()

        # Prepare outputs and gating parameters
        num_heads = H_v  # num_v_heads = 8
        g = torch.empty((B, 1, num_heads), dtype=torch.float32, device=q.device)
        beta = torch.empty((B, 1, num_heads), dtype=torch.float32, device=q.device)

        # Launch gate and beta kernel
        grid_g = (B * num_heads,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, num_heads
        )

        # Compute scale_over_sqrtK: host-side scalar (not tensor op on GPU)
        if scale is None or scale == 0.0:
            scale_over_sqrtK = float(1.0 / math.sqrt(K))
        else:
            scale_over_sqrtK = float(scale) / math.sqrt(K)

        # Allocate output and new_state
        output = torch.empty((B, num_heads), dtype=torch.float32, device=q.device)
        new_state = torch.empty_like(state, dtype=torch.float32, device=q.device)

        # Launch update kernel: one program per (b, h)
        grid_u = (B * num_heads,)
        triton_update_kernel[grid_u](
            q, k, v, state, g, beta, output, B, num_heads, V, K, scale_over_sqrtK
        )

        # Return outputs as expected by the original: output as bfloat16 unsqueezed, new_state float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
