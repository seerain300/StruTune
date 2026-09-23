import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16 or float32, shape [B, 1, H], we index by (b,h)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H], we index by (b,h)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, 0, h], dt_bias[h], and b[b, 0, h]
    # Assume a_ptr, b_ptr have stride H along dim-1; indexing by b,h only
    a_val = tl.cast(tl.load(a_ptr + b * H + h), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log[h]) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b[h]) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * H + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store to [B, 1, H] layout (stride-1 along dim-1)
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_invsqrt_scale_kernel(
    K: tl.constexpr,   # int, known at launch
    out_ptr            # *float32, shape [1] to store 1/sqrt(K)
):
    inv = 1.0 / tl.sqrt(K)
    tl.store(out_ptr, inv)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16 or float32, shape [B, H, K]
    k_ptr,             # *bfloat16 or float32, shape [B, H, K]
    v_ptr,             # *bfloat16 or float32, shape [B, H, V]
    state_ptr,         # *float32, shape [B, H, V, K]
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    output_ptr,        # *float32, shape [B * H] (we will index by pid)
    new_state_ptr,     # *float32, shape [B * H * V * K] (we will index by pid)
    scale_ptr,         # *float32, shape [1] to store scale as float
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads
    V: tl.constexpr,   # V
    K: tl.constexpr,   # K
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load g_val and beta_val (scalars for this (b, h))
    g_val = tl.load(g_ptr + b * H + h)
    beta_val = tl.load(beta_ptr + b * H + h)
    scale_val = tl.load(scale_ptr)

    # Prepare output scalar
    out = tl.zeros((), dtype=tl.float32)

    # Load q_h, k_h, v_h vectors
    q_h = tl.zeros([K], dtype=tl.float32)
    k_h = tl.zeros([K], dtype=tl.float32)
    v_h = tl.zeros([V], dtype=tl.float32)

    for j in tl.static_range(K):
        q_j = tl.cast(tl.load(q_ptr + b * H * K + h * K + j), tl.float32)
        k_j = tl.cast(tl.load(k_ptr + b * H * K + h * K + j), tl.float32)
        q_h[j] = q_j
        k_h[j] = k_j

    for v_idx in tl.static_range(V):
        v_h[v_idx] = tl.cast(tl.load(v_ptr + b * H * V + h * V + v_idx), tl.float32)

    # Load state_old: [V, K]
    state_old = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in tl.static_range(V):
        row_base = b * H * V * K + h * V * K + v_idx * K
        for k_idx in tl.static_range(K):
            state_old[v_idx, k_idx] = tl.cast(tl.load(state_ptr + row_base + k_idx), tl.float32)

    # Compute old_v = k_h @ (g * state_old)  -> (K,)
    old_v = tl.zeros([K], dtype=tl.float32)
    for j in tl.static_range(K):
        sum_val = tl.zeros((), dtype=tl.float32)
        for v_idx in tl.static_range(V):
            # g applied elementwise
            sum_val += state_old[v_idx, j] * g_val
        old_v[j] = sum_val * k_h[j]

    # Compute new_v = beta * v_h + (1 - beta) * old_v  -> (V,)
    new_v = tl.zeros([V], dtype=tl.float32)
    for v_idx in tl.static_range(V):
        new_v[v_idx] = beta_val * v_h[v_idx] + (1.0 - beta_val) * old_v[v_idx]

    # Compute state_remove and state_update as scalars: k_h @ old_v and k_h @ new_v
    state_remove = tl.zeros((), dtype=tl.float32)
    state_update = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(K):
        state_remove += old_v[j] * k_h[j]
        state_update += new_v[j] * k_h[j]

    # Update h_state_new = (g * state_old) - state_remove + state_update
    h_state_new = state_old * g_val - state_remove + state_update

    # Compute output scalar: output = scale * q_h @ h_state_new
    # q_h @ h_state_new = sum_j q_h[j] * (sum_v h_state_new[v, j])
    for j in tl.static_range(K):
        col_sum = tl.zeros((), dtype=tl.float32)
        for v_idx in tl.static_range(V):
            col_sum += h_state_new[v_idx, j]
        out += q_h[j] * col_sum

    out = out * scale_val
    tl.store(output_ptr + pid, out)

    # Write new_state[b,h,:,:]
    for v_idx in tl.static_range(V):
        row_base = b * H * V * K + h * V * K + v_idx * K
        for k_idx in tl.static_range(K):
            tl.store(new_state_ptr + b * H * V * K + h * V * K + v_idx * K + k_idx,
                     h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original run() behavior.
        Returns:
          - output: bfloat16 tensor with shape (B, 1, H) (unsqueezed from (B, H))
          - new_state: float32 tensor with shape (B, H, V, K)
        """
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        H = num_v_heads  # number of heads

        # We assume q, k, v are [B, 1, H, K] (T=1), and state is [B, H, V, K]
        # Make sure inputs are on CUDA device
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All tensors must be on CUDA device for Triton."
        assert B == 1 or B > 1, "Batch size must be > 0"
        assert T == 1, "This implementation assumes T == 1"

        device = q.device

        # Allocate g and beta as float32 tensors [B, H]
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch gate/beta kernel
        grid_g = (B * H,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, H
        )

        # Compute scale_inv = 1 / sqrt(K) via Triton
        scale_inv = torch.empty((1,), dtype=torch.float32, device=device)
        grid_s = (1,)
        triton_invsqrt_scale_kernel[grid_s](
            K, scale_inv
        )
        # If scale is provided and non-zero, use it; otherwise use scale_inv
        if scale is None or scale == 0.0:
            scale_val = float(scale_inv.item())
        else:
            scale_val = float(scale)

        # Allocate output and new_state
        output = torch.empty((B * H,), dtype=torch.float32, device=device)  # we will return as (B, H)
        new_state = torch.empty_like(state, dtype=torch.float32, device=device)

        # Launch update kernel
        grid_u = (B * H,)
        triton_update_kernel[grid_u](
            q, k, v, state, g, beta, output, new_state, scale_inv, B, H, V, K
        )

        # Reshape output to (B, H), then unsqueeze to (B, 1, H) and cast to bfloat16
        output = output.view(B, H).unsqueeze(1).to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
