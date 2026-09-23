import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h using 1D contiguous storage)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h using 1D contiguous storage)
    g_ptr,             # *float32, shape [B, 1, H] (1D contiguous)
    beta_ptr,          # *float32, shape [B, 1, H] (1D contiguous)
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, 0, h] from a_ptr (1D contiguous)
    a_val = tl.cast(tl.load(a_ptr + b * H + h), tl.float32)
    # Load dt_bias[h]
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is 1D contiguous
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    x = a_val + dt_val
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # Load A_log[h]
    A_log_val = tl.load(A_log_ptr + h)  # A_log is 1D contiguous
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # Load b[b, 0, h], compute beta = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * H + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H] layout (flat 1D tensors, stride = 1)
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16, shape [B, H, K]
    k_ptr,             # *bfloat16, shape [B, H, K]
    v_ptr,             # *bfloat16, shape [B, H, V]
    state_ptr,         # *float32, shape [B, H, V, K]
    g_ptr,             # *float32, shape [B, H] (we index by b,h)
    beta_ptr,          # *float32, shape [B, H] (we index by b,h)
    output_ptr,        # *float32, shape [B, H]
    new_state_ptr,     # *float32, shape [B, H, V, K]
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Prepare vectors
    K_vec = tl.arange(0, K)
    V_vec = tl.arange(0, V)

    # Load q[b, h, :], k[b, h, :], v[b, h, :]
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    # Base offsets for (b,h)
    q_base = b * H * K + h * K
    k_base = b * H * K + h * K
    v_base = b * H * V + h * V

    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_ptr + q_base + j), tl.float32)
        k_elem = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
        q_vec[j] = q_elem
        k_vec[j] = k_elem

    for v_idx in range(0, V):
        v_elem = tl.cast(tl.load(v_ptr + v_base + v_idx), tl.float32)
        v_vec[v_idx] = v_elem

    # Load state_old: [V, K]
    h_state = tl.zeros([V, K], dtype=tl.float32)
    state_base = b * H * V * K + h * V * K
    for v_idx in range(0, V):
        row_base = state_base + v_idx * K
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(state_ptr + row_base + k_idx), tl.float32)

    # Load gate and beta scalars
    g_val = tl.load(g_ptr + b * H + h)
    beta_val = tl.load(beta_ptr + b * H + h)

    # Compute old_v = k @ (g * state_old)  -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = tl.zeros([], dtype=tl.float32)
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val
        old_v_vec[j] = tl.sum(sum_val * k_vec[j], axis=0)  # k_vec[j] is scalar, so sum_val * k[j]

    # Compute new_v = beta * v + (1-beta) * old_v
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update as scalars: k @ old_v and k @ new_v
    state_remove = tl.zeros([], dtype=tl.float32)
    state_update = tl.zeros([], dtype=tl.float32)
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = h_state * g_val - state_remove + state_update

    # Compute output scalar: output = scale * q_h @ h_state_new
    # scale = 1.0 / sqrt(K)
    inv_sqrtK = 1.0 / tl.sqrt(K)
    output_scalar = tl.zeros([], dtype=tl.float32)
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state_new[:, j]  # [V]
        sum_j = tl.zeros([], dtype=tl.float32)
        for v_idx in range(0, V):
            sum_j += row_j[v_idx]
        output_scalar += qj * sum_j
    output_scalar = output_scalar * inv_sqrtK

    # Store outputs
    tl.store(output_ptr + b * H + h, output_scalar)

    # Store new_state[b, h, :, :] = h_state_new
    for v_idx in range(0, V):
        row_base = state_base + v_idx * K
        for k_idx in range(0, K):
            tl.store(new_state_ptr + b * H * V * K + h * V * K + v_idx * K + k_idx,
                     h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device and dtype: all inputs should be CUDA for Triton
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All inputs must be CUDA tensors for Triton."
        assert A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "Parameters must be CUDA tensors."

        # Dimensions
        B = q.shape[0]
        Hq = q.shape[1]
        K = q.shape[2]
        Bk = k.shape[0]
        Hk = k.shape[1]
        Kk = k.shape[2]
        Bv = v.shape[0]
        Hv = v.shape[1]
        V = v.shape[2]

        # Squeeze T=1 (original code uses squeeze(1))
        q = q.squeeze(1).contiguous()  # [B, Hq, K]
        k = k.squeeze(1).contiguous()  # [B, Hk, K]
        v = v.squeeze(1).contiguous()  # [B, Hv, V]

        # num_v_heads is the head count for output/state. From original asserts, num_v_heads=8.
        num_v_heads = Hv

        # Compute g and beta in float32 (1D contiguous buffers)
        g = torch.empty(B * num_v_heads, dtype=torch.float32, device=q.device)
        beta = torch.empty(B * num_v_heads, dtype=torch.float32, device=q.device)

        grid_g = (B * num_v_heads,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, num_v_heads
        )

        # Compute output scalar and new state
        output = torch.empty(B * num_v_heads, dtype=torch.float32, device=q.device)
        new_state = torch.empty(B, num_v_heads, V, K, dtype=torch.float32, device=q.device)

        grid_u = (B * num_v_heads,)
        triton_update_kernel[grid_u](
            q, k, v, state, g, beta, output, new_state,
            B, num_v_heads, V, K,
        )

        # Return output (bfloat16, unsqueezed to (B, 1, H)), and new state (float32, shape (B, H, V, K))
        output_bf16 = output.view(B, num_v_heads).unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
