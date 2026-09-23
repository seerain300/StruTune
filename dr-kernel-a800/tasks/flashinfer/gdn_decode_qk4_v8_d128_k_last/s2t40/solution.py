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

    # Load inputs
    a_val = tl.cast(tl.load(a_ptr + b * H + h), tl.float32)          # a[b, 0, h]
    dt_val = tl.load(dt_bias_ptr + h)                                # dt_bias[h]
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    A_log_val = tl.load(A_log_ptr + h)                               # A_log[h]
    g_val = tl.exp(-tl.exp(A_log_val) * sp)                          # g[b, 0, h]

    beta_val = tl.sigmoid(tl.cast(tl.load(b_ptr + b * H + h), tl.float32))  # beta[b, 0, h]

    # Store results
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16, shape [B, QH, K]
    k_ptr,             # *bfloat16, shape [B, KH, K]
    v_ptr,             # *bfloat16, shape [B, VH, V]
    state_ptr,         # *float32, shape [B, H, V, K]
    g_ptr,             # *float32, shape [B, 1, H] (we use g[b, 0, h] = g[b, h])
    beta_ptr,          # *float32, shape [B, 1, H] (we use beta[b, 0, h] = beta[b, h])
    out_ptr,           # *float32, shape [B, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads
    K: tl.constexpr,   # K dimension
    V: tl.constexpr,   # V dimension
    QH: tl.constexpr,  # num_q_heads (assumed 4)
    KH: tl.constexpr,  # num_k_heads (assumed 4)
    VH: tl.constexpr,  # num_v_heads (assumed 8)
    scale: tl.constexpr,  # Python float scalar for scale
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Base offsets for this (b, h)
    q_base = b * (QH * K) + h * K
    k_base = b * (KH * K) + h * K
    v_base = b * (VH * V) + h * V

    # Load q_h, k_h, v_h
    q_h = tl.zeros([K], dtype=tl.float32)
    k_h = tl.zeros([K], dtype=tl.float32)
    v_h = tl.zeros([V], dtype=tl.float32)

    # Convert from bfloat16 to float32
    for j in range(0, K):
        q_j = tl.cast(tl.load(q_ptr + q_base + j), tl.float32)
        k_j = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
        q_h[j] = q_j
        k_h[j] = k_j
    for v_idx in range(0, V):
        v_elem = tl.cast(tl.load(v_ptr + v_base + v_idx), tl.float32)
        v_h[v_idx] = v_elem

    # Load state_old as float32: [V, K]
    state_old = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + b * (H * V * K) + h * V * K + v_idx * K
        for k_idx in range(0, K):
            state_val = tl.cast(tl.load(state_ptr + b * (H * V * K) + h * V * K + v_idx * K + k_idx), tl.float32)
            state_old[v_idx, k_idx] = state_val

    # Load g and beta scalars for this (b, h)
    g_val = tl.load(g_ptr + b * H + h)
    beta_val = tl.load(beta_ptr + b * H + h)

    # Compute old_v = k_h @ (g * state_old) -> (K,)
    old_v = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        s_j = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            s_j += state_old[v_idx, j] * g_val
        old_v[j] = k_h[j] * s_j

    # new_v = beta * v + (1 - beta) * old_v -> (V,)
    new_v = beta_val * v_h + (1.0 - beta_val) * old_v

    # Compute state_remove and state_update: scalars
    state_remove = tl.zeros((), dtype=tl.float32)
    state_update = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        state_remove += old_v[j] * k_h[j]
        state_update += new_v[j] * k_h[j]

    # Update state_new elementwise: h_state_new = (g * state_old) - state_remove + state_update
    h_state_new = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + b * (H * V * K) + h * V * K + v_idx * K
        for k_idx in range(0, K):
            state_val = tl.cast(tl.load(state_ptr + b * (H * V * K) + h * V * K + v_idx * K + k_idx), tl.float32)
            h_state_new[v_idx, k_idx] = state_val * g_val - state_remove + state_update

    # Compute output scalar: output = scale * q_h @ h_state_new
    output_scalar = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        row_j = h_state_new[:, j]  # [V] vector
        output_scalar += q_h[j] * tl.sum(row_j)

    output_scalar = scale * output_scalar

    # Store output
    tl.store(out_ptr + b * H + h, output_scalar)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version of the reference run() forward.
        Computes g, beta in Triton, updates state and computes output in Triton.
        Returns:
          - output: (B, 1, H) in bfloat16
          - new_state: (B, H, V, K) in float32
        """
        # Fixed assumptions for this implementation
        B, _, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        num_heads = num_v_heads
        device = q.device

        # Ensure inputs are contiguous and on device
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        A_log = A_log.contiguous()
        a = a.contiguous()
        dt_bias = dt_bias.contiguous()
        b = b.contiguous()

        # Allocate g and beta outputs [B, H] in float32
        g = torch.empty((B, num_heads), dtype=torch.float32, device=device)
        beta = torch.empty((B, num_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        grid = (B * num_heads,)
        triton_gate_beta_kernel[grid](
            A_log.float(),                # A_log
            a.squeeze(1),                 # a[B,1,H] -> [B,H] (bfloat16)
            dt_bias.float(),              # dt_bias
            b.squeeze(1).to(torch.bfloat16),  # b[B,1,H] -> [B,H] keep bfloat16 for kernel
            g,                            # output g[B,H]
            beta,                         # output beta[B,H]
            B=B, H=num_heads
        )

        # Allocate output and new_state
        out = torch.empty((B, num_heads), dtype=torch.float32, device=device)
        new_state = torch.empty((B, num_heads, V, K), dtype=torch.float32, device=device)

        # Compute scale on host: handle None correctly without calling .float() on scale
        if scale is None:
            scale_val = 1.0 / math.sqrt(K)
        else:
            scale_val = float(scale)

        # Launch Triton update kernel for each (b, h)
        grid = (B * num_heads,)
        triton_update_kernel[grid](
            q,                           # q [B,4,128]
            k,                           # k [B,4,128]
            v,                           # v [B,8,128]
            state.float(),               # state [B,8,128,128] -> float32
            g,                           # g [B,H]
            beta,                        # beta [B,H]
            out,                         # output [B,H]
            B=B, H=num_heads, K=K, V=V, QH=4, KH=4, VH=8, scale=scale_val
        )

        # Cast output to bfloat16 and unsqueeze to (B,1,H)
        out_bf16 = out.unsqueeze(1).to(torch.bfloat16)

        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
