import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_update_kernel_bh(
    q_ptr,             # *bfloat16, [B, H, K]
    k_ptr,             # *bfloat16, [B, H, K]
    v_ptr,             # *bfloat16, [B, H, V]
    state_ptr,         # *float32,  [B, H, V, K] (we write updated state here)
    A_log_ptr,         # *float32,  [H]
    dt_bias_ptr,       # *float32,  [H]
    b_ptr,             # *bfloat16, [B, 1, H]
    output_ptr,        # *float32,  [B, H]
    scale: tl.float32, # scalar float32
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # num_heads
    V: tl.constexpr,   # V dimension of state
    K: tl.constexpr,   # K dimension of state
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Compute g = exp(-exp(A_log[h]) * softplus(a[b,0,h] + dt_bias[h]))
    a_elem = tl.cast(tl.load(b_ptr + b * (1 * H) + h), tl.float32)  # b_ptr has shape [B,1,H], stride(1)=H, so index b*H + h
    dt_elem = tl.load(dt_bias_ptr + h)
    x = a_elem + dt_elem

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    A_log_elem = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_elem) * sp)  # scalar float32

    # beta = sigmoid(b[b,0,h]) = 1 / (1 + exp(-b))
    b_elem = tl.cast(tl.load(b_ptr + b * (1 * H) + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_elem))  # scalar float32

    # Load vectors q_h, k_h, v_h
    q_base = b * (H * K) + h * K
    k_base = b * (H * K) + h * K
    v_base = b * (H * V) + h * V

    # Initialize vectors
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    # Fill q_vec and k_vec
    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_ptr + q_base + j), tl.float32)
        k_elem = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
        q_vec[j] = q_elem
        k_vec[j] = k_elem

    # Fill v_vec
    for v_idx in range(0, V):
        v_elem = tl.cast(tl.load(v_ptr + v_base + v_idx), tl.float32)
        v_vec[v_idx] = v_elem

    # Load state_old for this (b,h): [V, K]
    state_old = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + b * (H * V * K) + h * V * K + v_idx * K
        for k_idx in range(0, K):
            state_old[v_idx, k_idx] = tl.cast(tl.load(state_ptr + b * (H * V * K) + h * V * K + v_idx * K + k_idx), tl.float32)

    # Compute old_v = k @ (g * state_old) -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_val += state_old[v_idx, j] * g_val
        old_v_vec[j] = sum_val * k_vec[j]  # multiply by k_j, which is scalar

    # new_v = beta * v + (1 - beta) * old_v -> (V,)
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update: scalars k @ old_v and k @ new_v
    state_remove = tl.zeros((), dtype=tl.float32)
    state_update = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        k_j = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
        state_remove += old_v_vec[j] * k_j
        state_update += new_v_vec[j] * k_j

    # Update h_state_new elementwise: h_state_new = (g * state_old) - state_remove + state_update
    h_state_new = state_old * g_val - state_remove + state_update

    # Compute output scalar: output = scale * q_h @ h_state_new
    # q_h is already in q_vec
    output_scalar = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        row_j = h_state_new[:, j]  # [V]
        output_scalar += q_vec[j] * tl.sum(row_j)

    output_scalar = scale * output_scalar

    # Store output
    tl.store(output_ptr + b * H + h, output_scalar)

    # Store updated state for this (b,h): [V, K], contiguous
    for v_idx in range(0, V):
        row_base_out = state_ptr + b * (H * V * K) + h * V * K + v_idx * K
        for k_idx in range(0, K):
            tl.store(state_ptr + b * (H * V * K) + h * V * K + v_idx * K + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original run function.
        - q: [B, 1, H_q, K], H_q=4
        - k: [B, 1, H_k, K], H_k=4
        - v: [B, 1, H_v, V], H_v=8, V=128
        - state: [B, H_v, V, K], float32
        - A_log: [H_v], float32
        - a: [B, 1, H_v], bfloat16 or float32
        - dt_bias: [H_v], float32
        - b: [B, 1, H_v], bfloat16
        - scale: float or None; if None or 0.0, use 1/sqrt(K)
        Returns:
        - output: bfloat16, shape [B, 1, H_v] (caller will unsqueeze(1) as in original)
        - new_state: float32, shape [B, H_v, V, K]
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All tensors must be CUDA tensors for Triton."
        B, Tq, Hq, K = q.shape
        B2, Tk, Hk, K2 = k.shape
        B3, Tv, Hv, V = v.shape
        B4, H4, V4, K4 = state.shape
        assert B == B2 == B3 == B4, "Batch sizes must match"
        assert Tq == 1 and Tk == 1, "T must be 1"
        assert Hq == 4 and Hk == 4 and Hv == 8 and V == 128 and K == 128 and K2 == 128 and V4 == 128 and K4 == 128, "Expected fixed dims (4,4,8,128)"

        # Compute scale as a Python float to avoid float() on callable
        if scale is None or scale == 0.0:
            scale_val = float(1.0 / math.sqrt(K))
        else:
            scale_val = float(scale)

        # Ensure inputs are contiguous and device is CUDA
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        state_c = state.contiguous()

        # Allocate outputs
        output = torch.empty((B, Hv), dtype=torch.float32, device=q.device)
        new_state = torch.empty_like(state_c, dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b,h)
        grid = (B * Hv,)
        triton_update_kernel_bh[grid](
            q_c, k_c, v_c, state_c, A_log, dt_bias, b, output, scale_val,
            B, Hv, V, K,
        )

        # Return output as bfloat16 (unsqueezed by caller to [B,1,Hv]) and new_state as float32
        return output.unsqueeze(1).to(torch.bfloat16), new_state


def run(*args):
    return ModelNew()(*args)
