import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16/float32, shape [B, 1, H] (we index by b,h)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # num heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, 0, h]
    a_val = tl.load(a_ptr + b * H + h)
    a_val = tl.cast(a_val, tl.float32)

    # Load dt_bias[h]
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]

    # x = a + dt
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # Load A_log[h]
    A_log_val = tl.load(A_log_ptr + h)  # A_log is [H]

    # g = exp(-exp(A_log[h]) * softplus(x))
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b[b, 0, h])
    b_val = tl.load(b_ptr + b * H + h)
    b_val = tl.cast(b_val, tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16, shape [B, 1, Q], here Q=4, we index by (b)
    k_ptr,             # *bfloat16, shape [B, 1, K], K=128
    v_ptr,             # *bfloat16, shape [B, 1, V], V=8
    state_ptr,         # *float32, shape [B, H, V, K], contiguous
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    new_state_ptr,     # *float32, shape [B, H, V, K], contiguous
    output_ptr,        # *float32, shape [B, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # num heads
    Q: tl.constexpr,   # num q heads per b, actually 4 in provided inputs
    V: tl.constexpr,   # num v heads, actually 8
    K: tl.constexpr,   # k dimension, actually 128
    scale: tl.float32, # scalar 1/sqrt(K) computed on host
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Slices for q_h, k_h, v_h (make contiguous)
    q_base = q_ptr + b  # shape [Q]
    k_base = k_ptr + b  # shape [K]
    v_base = v_ptr + b  # shape [V]

    # Prepare vectors
    g_val = tl.load(g_ptr + b * H + h)
    beta_val = tl.load(beta_ptr + b * H + h)

    # Load q_h (Q=4), k_h, v_h (V=8)
    q_h = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        q_j = tl.load(q_ptr + b * 4 + j)  # q shape [B,1,4] => index b*4 + j
        q_h[j] = tl.cast(q_j, tl.float32)

    k_h = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        k_j = tl.load(k_ptr + b * 128 + j)  # k shape [B,1,128], but we pass [B,128] base and iterate
        k_h[j] = tl.cast(k_j, tl.float32)

    v_h = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        v_elem = tl.load(v_ptr + b * 8 + v_idx)  # v shape [B,1,8]
        v_h[v_idx] = tl.cast(v_elem, tl.float32)

    # Load state_old: contiguous slice [V, K] starting at state_ptr[b, h]
    state_old = tl.zeros([V, K], dtype=tl.float32)
    # For contiguous [B,H,V,K], element (b,h,v,k) offset: b*(H*V*K) + h*(V*K) + v*K + k
    base_bh = b * (H * V * K) + h * (V * K)
    for v_idx in range(0, V):
        row_base = base_bh + v_idx * K
        for k_idx in range(0, K):
            state_old[v_idx, k_idx] = tl.load(state_ptr + base_bh + v_idx * K + k_idx)

    # Compute old_v = k_h @ (g * state_old)
    old_v = tl.zeros([K], dtype=tl.float32)
    for k_idx in range(0, K):
        sum_val = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            # state_old[v_idx, k_idx] is float32
            sum_val += state_old[v_idx, k_idx] * g_val
        old_v[k_idx] = sum_val * k_h[k_idx]

    # Compute new_v = beta * v + (1 - beta) * old_v
    new_v = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v[v_idx] = beta_val * v_h[v_idx] + (1.0 - beta_val) * old_v[v_idx]

    # Compute state_remove and state_update: scalars k_h @ old_v and k_h @ new_v
    state_remove = tl.zeros((), dtype=tl.float32)
    state_update = tl.zeros((), dtype=tl.float32)
    for k_idx in range(0, K):
        state_remove += old_v[k_idx] * k_h[k_idx]
        state_update += new_v[k_idx] * k_h[k_idx]

    # Update h_state_new = (g * state_old) - state_remove + state_update (elementwise)
    h_state_new = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = base_bh + v_idx * K
        for k_idx in range(0, K):
            state_val = tl.load(state_ptr + base_bh + v_idx * K + k_idx)  # original state[b,h,v,k]
            # h_state_new[v_idx, k_idx] = state_val * g_val - state_remove + state_update
            h_state_new[v_idx, k_idx] = state_val * g_val - state_remove + state_update

    # Store new_state back (contiguous)
    for v_idx in range(0, V):
        row_base = base_bh + v_idx * K
        for k_idx in range(0, K):
            tl.store(new_state_ptr + base_bh + v_idx * K + k_idx, h_state_new[v_idx, k_idx])

    # Compute output scalar: output = scale * q_h @ h_state_new
    out_scalar = tl.zeros((), dtype=tl.float32)
    for k_idx in range(0, K):
        row_k = h_state_new[:, k_idx]  # [V] vector
        # Sum over V
        for v_idx in range(0, V):
            out_scalar += row_k[v_idx] * q_h[k_idx]

    # Store output[b, h]
    tl.store(output_ptr + b * H + h, out_scalar * scale)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation. All computation happens inside Triton kernels.

        Args:
          q: [B, 1, 4, 128] (bfloat16 or float32)
          k: [B, 1, 4, 128] (bfloat16 or float32)
          v: [B, 1, 8, 128] (bfloat16 or float32)
          state: [B, 8, 128, 128] (float32)
          A_log: [8] (float32)
          a: [B, 1, 8] (bfloat16 or float32)
          dt_bias: [8] (float32)
          b: [B, 1, 8] (bfloat16)
          scale: float or None; if None or 0.0, use 1/sqrt(K)

        Returns:
          output: [B, 8] (float32) — later converted to bfloat16 by caller
          new_state: [B, 8, 128, 128] (float32)
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All tensors must be CUDA."
        assert A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "Parameter tensors must be CUDA."

        B, _, Q, K = q.shape
        _, _, num_k_heads, _ = k.shape  # 4
        _, _, num_v_heads, V = v.shape  # 8
        H = num_v_heads  # 8
        device = q.device

        # Compute g and beta using Triton gate kernel
        g = torch.empty(B, H, device=device, dtype=torch.float32)
        beta = torch.empty(B, H, device=device, dtype=torch.float32)

        # Launch gate kernel: grid over (B*H)
        grid = (B * H,)
        triton_gate_beta_kernel[grid](A_log, a.reshape(-1, H), dt_bias, b.reshape(-1, H), g, beta, B=B, H=H)

        # Compute 1/sqrt(K) on host (scale may be None/0)
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(K)

        # Allocate outputs and new_state
        new_state = torch.empty_like(state, dtype=torch.float32, device=device)  # Triton writes float32
        output = torch.empty(B, H, device=device, dtype=torch.float32)

        # Launch update kernel: grid over (B*H)
        triton_update_kernel[grid](
            q.view(B, Q * K),         # [B, 512] but we use Q=4, iterate j in 0..127; we pass q[0], q[1], q[2], q[3]
            k.view(B, K),             # [B, 128]
            v.view(B, V),             # [B, 8]
            state,                    # [B, H, V, K], float32 contiguous
            g.view(B, H),             # [B, H]
            beta.view(B, H),          # [B, H]
            new_state,                # [B, H, V, K], float32
            output,                   # [B, H]
            B=B, H=H, Q=Q, V=V, K=K,
            scale=float(scale),
        )

        # Return: output is [B, H] float32, new_state is [B, H, V, K] float32
        # The original Model returns output as bfloat16 (unsqueezed). Here we keep float32 to match compute.
        # Convert to bfloat16 as requested by the original get_inputs and caller:
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
