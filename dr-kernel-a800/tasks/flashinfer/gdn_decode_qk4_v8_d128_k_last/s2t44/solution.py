import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16 or float32, shape [B, 1, H] (index by b,h)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H] (index by b,h)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size (not used here, per (b,h) loop handled by host)
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, 0, h] and dt_bias[h]
    a_val = tl.cast(tl.load(a_ptr + b * H + h), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log[h]) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b[b, 0, h])
    b_val = tl.cast(tl.load(b_ptr + b * H + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16, shape [B, 8, K] (since QH=4 -> 8 after repeat_interleave)
    k_ptr,             # *bfloat16, shape [B, 8, K]
    v_ptr,             # *bfloat16, shape [B, 8, V] (VH=8)
    state_ptr,         # *float32, shape [B, H, V, K]
    g_ptr,             # *float32, shape [B, 1, H] (we index g[b, h])
    beta_ptr,          # *float32, shape [B, 1, H] (we index beta[b, h])
    out_ptr,           # *float32, shape [B, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
    K: tl.constexpr,   # K dimension (128)
    V: tl.constexpr,   # V dimension (128)
    QH: tl.constexpr,  # num_q_heads (4) -> 8 in inputs
    KH: tl.constexpr,  # num_k_heads (4) -> 8 in inputs
    VH: tl.constexpr,  # num_v_heads (8)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Map head h -> q_idx=k_idx=v_idx = h due to repeat_interleave(2, dim=1)
    q_base = b * (QH * K) + h * K
    k_base = b * (KH * K) + h * K
    v_base = b * (VH * V) + h * V

    # Load q_h, k_h, v_h (K and V vectors)
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        qj = tl.cast(tl.load(q_ptr + q_base + j), tl.float32)
        kj = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
        q_vec[j] = qj
        k_vec[j] = kj
    for j in range(0, V):
        v_vec[j] = tl.cast(tl.load(v_ptr + v_base + j), tl.float32)

    # Load state_old: [V, K]
    state_old = tl.zeros([V, K], dtype=tl.float32)
    for v in range(0, V):
        row_base = state_ptr + b * (H * V * K) + h * V * K + v * K
        for k in range(0, K):
            state_old[v, k] = tl.cast(tl.load(state_ptr + b * (H * V * K) + h * V * K + v * K + k), tl.float32)

    # Compute old_v = k_h @ (g * state_old) -> (K,)
    g_val = tl.load(g_ptr + b * H + h)  # g[b, h]
    old_v = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        dot = tl.zeros((), dtype=tl.float32)
        for k in range(0, K):
            dot += k_vec[k] * g_val * state_old[:, k]
        old_v[j] = dot

    # new_v = beta * v_h + (1 - beta) * old_v
    beta_val = tl.load(beta_ptr + b * H + h)
    new_v = tl.zeros([V], dtype=tl.float32)
    for v in range(0, V):
        new_v[v] = beta_val * v_vec[v] + (1.0 - beta_val) * old_v[v]

    # Compute state_remove and state_update: scalars k_h @ old_v and k_h @ new_v
    state_remove = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        state_remove += k_vec[j] * old_v[j]
    state_update = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        state_update += k_vec[j] * new_v[j]

    # Update new_state: new_state = (g * state_old) - state_remove + state_update
    new_state = g_val * state_old - state_remove + state_update

    # Write new_state back
    for v in range(0, V):
        row_base = state_ptr + b * (H * V * K) + h * V * K + v * K
        for k in range(0, K):
            tl.store(state_ptr + b * (H * V * K) + h * V * K + v * K + k, new_state[v, k])

    # Compute output scalar: output = q_h @ new_state
    output_scalar = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        row_j = new_state[:, j]  # [V]
        output_scalar += q_vec[j] * tl.sum(row_j)

    # Store output
    tl.store(out_ptr + b * H + h, output_scalar)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure contiguous and on CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "Tensors must be on CUDA device."
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        A_log = A_log.contiguous()
        a = a.contiguous()
        dt_bias = dt_bias.contiguous()
        b = b.contiguous()

        B = q.size(0)
        H = b.size(1)  # num_v_heads
        # Prepare output buffers
        g = torch.empty((B, H), device=q.device, dtype=torch.float32)
        beta = torch.empty((B, H), device=q.device, dtype=torch.float32)
        out = torch.empty((B, H), device=q.device, dtype=torch.float32)

        # Launch kernel to compute g and beta per (b,h)
        grid = (B * H,)
        triton_gate_beta_kernel[grid](
            A_log, a, dt_bias, b, g, beta, B, H,
            num_warps=2
        )

        # Launch kernel to update state and compute output per (b,h)
        # K and V are fixed to 128 in provided inputs, but make them runtime constants in kernel via meta
        K = 128
        V = 128
        QH = 4  # num_q_heads in original, expanded to 8 in inputs
        KH = 4  # num_k_heads in original, expanded to 8 in inputs
        VH = 8  # num_v_heads

        triton_update_kernel[grid](
            q, k, v, state, g, beta, out,
            B, H, K, V, QH, KH, VH,
            num_warps=2
        )

        # Return output as bfloat16 and unsqueeze to (B,1,H), new_state remains float32
        out_bf16 = out.unsqueeze(1).to(torch.bfloat16)
        new_state = state  # updated in kernel
        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
