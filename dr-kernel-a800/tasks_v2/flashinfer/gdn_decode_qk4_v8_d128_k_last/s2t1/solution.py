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
    # Compute x = a + dt_bias for this (b, h)
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
def triton_update_kernel(
    q_ptr,             # *bfloat16, shape [B, Hq, K] but we only use one head per (b,h) -> we pass q per b,h
    k_ptr,             # *bfloat16, shape [B, Hk, K]
    v_ptr,             # *bfloat16, shape [B, Hv, V]
    state_ptr,         # *float32, shape [B, H, V, K]
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    output_ptr,        # *float32, shape [B, H]
    new_state_ptr,     # *float32, shape [B, H, V, K]
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    q_stride_b, q_stride_h, q_stride_k,
    k_stride_b, k_stride_h, k_stride_k,
    v_stride_b, v_stride_h, v_stride_v,
    state_stride_b, state_stride_h, state_stride_v, state_stride_k,
    g_stride_b, g_stride_h,
    beta_stride_b, beta_stride_h,
    out_stride_b, out_stride_h,
    ns_stride_b, ns_stride_h, ns_stride_v, ns_stride_k,
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load vectors and matrices
    # q_vec: [K]
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    # Load q[b, h, :], k[b, h, :], v[b, h, :]
    # q
    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_ptr + b * q_stride_b + h * q_stride_h + j * q_stride_k), tl.float32)
        q_vec[j] = q_elem
    # k
    for j in range(0, K):
        k_elem = tl.cast(tl.load(k_ptr + b * k_stride_b + h * k_stride_h + j * k_stride_k), tl.float32)
        k_vec[j] = k_elem
    # v
    for v_idx in range(0, V):
        v_elem = tl.cast(tl.load(v_ptr + b * v_stride_b + h * v_stride_h + v_idx * v_stride_v), tl.float32)
        v_vec[v_idx] = v_elem

    # Load state_old: [V, K]
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            val = tl.cast(tl.load(state_ptr + b * state_stride_b + h * state_stride_h + v_idx * state_stride_v + k_idx * state_stride_k), tl.float32)
            h_state[v_idx, k_idx] = val

    # Load gate and beta scalars
    g_val = tl.load(g_ptr + b * g_stride_b + h * g_stride_h)
    beta_val = tl.load(beta_ptr + b * beta_stride_b + h * beta_stride_h)

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
    # q_vec is [K], h_state_new is [V, K] -> we perform a reduction: sum_j q[j] * sum_v h_state_new[v, j]
    output_scalar = tl.zeros([], dtype=tl.float32)
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state_new[:, j]  # [V]
        sum_j = tl.zeros([], dtype=tl.float32)
        for v_idx in range(0, V):
            sum_j += row_j[v_idx]
        output_scalar += qj * sum_j

    # Store outputs
    tl.store(output_ptr + b * out_stride_b + h * out_stride_h, output_scalar)
    # Store new_state[b, h, :, :] = h_state_new
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            tl.store(new_state_ptr + b * ns_stride_b + h * ns_stride_h + v_idx * ns_stride_v + k_idx * ns_stride_k,
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

        # Squeeze T=1
        q = q.squeeze(1).contiguous()  # [B, Hq, K]
        k = k.squeeze(1).contiguous()  # [B, Hk, K]
        v = v.squeeze(1).contiguous()  # [B, Hv, V]

        # num_v_heads is the head count for state output. In provided get_inputs, num_v_heads=8, consistent with v.shape[1].
        num_v_heads = Hv

        # Compute g and beta in float32
        g = torch.empty(B, 1, num_v_heads, dtype=torch.float32, device=q.device)
        beta = torch.empty(B, 1, num_v_heads, dtype=torch.float32, device=q.device)

        grid_g = (B * num_v_heads,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, num_v_heads
        )

        # Compute output scalar and new state
        output = torch.empty(B, num_v_heads, dtype=torch.float32, device=q.device)
        new_state = torch.empty(B, num_v_heads, V, K, dtype=torch.float32, device=q.device)

        # Get strides (note: Triton expects integer strides in elements)
        q_stride_b = q.stride(0); q_stride_h = q.stride(1); q_stride_k = q.stride(2)
        k_stride_b = k.stride(0); k_stride_h = k.stride(1); k_stride_k = k.stride(2)
        v_stride_b = v.stride(0); v_stride_h = v.stride(1); v_stride_v = v.stride(2)
        state_stride_b = state.stride(0); state_stride_h = state.stride(1); state_stride_v = state.stride(3); state_stride_k = state.stride(4)
        g_stride_b = g.stride(0); g_stride_h = g.stride(1)
        beta_stride_b = beta.stride(0); beta_stride_h = beta.stride(1)
        out_stride_b = output.stride(0); out_stride_h = output.stride(1)
        ns_stride_b = new_state.stride(0); ns_stride_h = new_state.stride(1); ns_stride_v = new_state.stride(2); ns_stride_k = new_state.stride(3)

        grid_u = (B * num_v_heads,)
        triton_update_kernel[grid_u](
            q, k, v, state, g, beta, output, new_state,
            B, num_v_heads, V, K,
            q_stride_b, q_stride_h, q_stride_k,
            k_stride_b, k_stride_h, k_stride_k,
            v_stride_b, v_stride_h, v_stride_v,
            state_stride_b, state_stride_h, state_stride_v, state_stride_k,
            g_stride_b, g_stride_h,
            beta_stride_b, beta_stride_h,
            out_stride_b, out_stride_h,
            ns_stride_b, ns_stride_h, ns_stride_v, ns_stride_k,
        )

        # Return output (bfloat16, unsqueezed to (B, 1, H)), and new state (float32, shape (B, H, V, K))
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
