import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16 or float32, shape [B, 1, H] (we index by (b,h))
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H] (we index by (b,h))
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load scalars for this (b, h)
    a_val = tl.cast(tl.load(a_ptr + b * H + h), tl.float32)        # [B,1,H] -> index by (b,h)
    dt_val = tl.load(dt_bias_ptr + h)                               # [H]
    A_log_val = tl.load(A_log_ptr + h)                              # [H]

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    x = a_val + dt_val
    abs_x = tl.abs(x)
    softplus_x = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log[h]) * softplus(x))
    g_val = tl.exp(-tl.exp(A_log_val) * softplus_x)

    # beta = sigmoid(b[h]) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * H + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_invsqrt_kernel(
    K_ptr,             # *int32, shape [1] containing K
    scale_ptr,         # *float32, shape [1]
    B: tl.constexpr,
):
    # Single program computes 1/sqrt(K) and writes to scale_ptr[0]
    K_val = tl.load(K_ptr)  # int32
    inv = 1.0 / tl.sqrt(tl.cast(K_val, tl.float32))
    tl.store(scale_ptr, inv)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16, shape [B,1,4,K]
    k_ptr,             # *bfloat16, shape [B,1,4,K]
    v_ptr,             # *bfloat16, shape [B,1,8,V]
    state_ptr,         # *float32, shape [B,8,V,K]
    g_ptr,             # *float32, shape [B,1,H] (we use h from pid)
    beta_ptr,          # *float32, shape [B,1,H] (we use h from pid)
    new_state_ptr,     # *float32, shape [B,8,V,K]
    output_ptr,        # *float32, shape [B,8]
    K_ptr,             # *int32, shape [1] (for scale read)
    scale_ptr,         # *float32, shape [1] (pass scale from host)
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
    V: tl.constexpr,   # V dimension (128)
    K: tl.constexpr,   # K dimension (128)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load g and beta scalars for this (b, h)
    g_val = tl.load(g_ptr + b * H + h)  # [B,1,H] -> index by (b,h)
    beta_val = tl.load(beta_ptr + b * H + h)

    # Map pid (b,h) to heads: q/k have 4 heads, v has 8 heads
    q_head = h % 4
    k_head = h % 4
    v_head = h % 8

    # Base offsets
    q_base = b * (4 * K) + q_head * K
    k_base = b * (4 * K) + k_head * K
    v_base = b * (8 * V) + v_head * V

    # Load q_h and k_h vectors
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        q_j = tl.cast(tl.load(q_ptr + b * (4 * K) + q_head * K + j), tl.float32)
        k_j = tl.cast(tl.load(k_ptr + b * (4 * K) + k_head * K + j), tl.float32)
        q_vec[j] = q_j
        k_vec[j] = k_j

    # Load v_h vector
    v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        v_elem = tl.cast(tl.load(v_ptr + b * (8 * V) + v_head * V + v_idx), tl.float32)
        v_vec[v_idx] = v_elem

    # Load state_old [V, K] for (b, h)
    state_old = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = b * (H * V * K) + h * V * K + v_idx * K
        for k_idx in range(0, K):
            val = tl.cast(tl.load(state_ptr + row_base + k_idx), tl.float32)
            state_old[v_idx, k_idx] = val

    # g_scaled = g * state_old
    g_scaled = state_old * g_val

    # old_v = k_h @ g_scaled -> [K]
    old_v = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        for v_idx in range(0, V):
            old_v[j] += g_scaled[v_idx, j] * k_vec[j]

    # new_v = beta * v_h + (1 - beta) * old_v -> [V]
    s_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        for j in range(0, K):
            s_vec[v_idx] += g_scaled[v_idx, j] * k_vec[j]
    new_v = beta_val * v_vec + (1.0 - beta_val) * s_vec  # [V]

    # Compute state_remove = sum_j old_v[j] * k[j], state_update = sum_j new_v[j] * k[j] (scalars)
    state_remove = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        state_remove += old_v[j] * k_vec[j]
    state_update = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        state_update += new_v[j] * k_vec[j]

    # h_state_new = g * state_old - state_remove + state_update
    h_state_new = g_scaled - state_remove + state_update  # broadcast scalar

    # output = scale * (q_h @ h_state_new) -> scalar
    q_dot = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        q_dot += q_vec[j] * tl.sum(h_state_new[:, j])  # sum over V

    # Load scale and compute output
    scale_val = tl.load(scale_ptr)  # 1/sqrt(K)
    out_scalar = scale_val * q_dot
    tl.store(output_ptr + b * H + h, out_scalar)

    # Store new_state (b, h) slice
    for v_idx in range(0, V):
        row_base_new = new_state_ptr + b * (H * V * K) + h * V * K + v_idx * K
        row_base_old = state_ptr + b * (H * V * K) + h * V * K + v_idx * K
        for k_idx in range(0, K):
            new_val = h_state_new[v_idx, k_idx]
            tl.store(row_base_new + k_idx, new_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Shapes
        B, T_q, num_q_heads, K = q.shape
        _, T_k, num_k_heads, _ = k.shape
        _, T_v, num_v_heads, V = v.shape
        assert T_q == 1 and T_k == 1 and T_v == 1

        # Allocate outputs
        output = torch.empty(B * num_v_heads, dtype=torch.float32, device=q.device)
        new_state = torch.empty_like(state)

        # g and beta: [B, H]
        g = torch.empty(B * num_v_heads, dtype=torch.float32, device=q.device)
        beta = torch.empty(B * num_v_heads, dtype=torch.float32, device=q.device)

        # K scalar for scale
        K_tensor = torch.tensor([K], dtype=torch.int32, device=q.device)
        scale_tensor = torch.empty(1, dtype=torch.float32, device=q.device)  # we will fill this once

        # Launch Triton kernels: compute g and beta
        triton_gate_beta_kernel[(B * num_v_heads,)](
            A_log, a.squeeze(1).contiguous(), dt_bias, b.squeeze(1).contiguous(),
            g, beta,
            B=B, H=num_v_heads
        )

        # Compute scale = 1/sqrt(K)
        triton_invsqrt_kernel[(1,)](K_tensor, scale_tensor)

        # Update and compute output per (b, h)
        triton_update_kernel[(B * num_v_heads,)](
            q.contiguous(), k.contiguous(), v.contiguous(), state.contiguous(),
            g, beta, new_state, output, K_tensor, scale_tensor,
            B=B, H=num_v_heads, V=V, K=K
        )

        # Reshape output to (B, H) and return as bfloat16 unsqueezed
        output = output.view(B, num_v_heads).unsqueeze(1).to(torch.bfloat16)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
