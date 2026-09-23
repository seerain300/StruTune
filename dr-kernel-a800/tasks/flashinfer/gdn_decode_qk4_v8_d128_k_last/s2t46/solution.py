import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *float32, shape [B, 1, H] (we index by b,h)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *float32, shape [B, 1, H] (we index by b,h)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load inputs
    a_val = tl.load(a_ptr + b * H + h)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log[h]) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b[b, 0, h])
    b_val = tl.load(b_ptr + b * H + h)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_update_output_kernel(
    q_ptr,             # *float32, [B, 1, 4, K], contiguous
    k_ptr,             # *float32, [B, 1, 4, K], contiguous
    v_ptr,             # *float32, [B, 1, 8, V], contiguous
    state_ptr,         # *float32, [B, H, V, K], contiguous
    g_ptr,             # *float32, [B, 1, H] (we index by h)
    beta_ptr,          # *float32, [B, 1, H] (we index by h)
    output_ptr,        # *float32, [B, H]
    new_state_ptr,     # *float32, [B, H, V, K], contiguous (we will write updated slice)
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads
    V: tl.constexpr,   # 128
    K: tl.constexpr,   # 128
    scale: tl.constexpr,  # float32 scalar 1/sqrt(K)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Base offsets (dim-1 is heads, so each "head" is a contiguous block)
    # q has shape [B, 1, 4, K] so 1*4*K elements; but in inputs heads are 0 so base is b*(1*4*K)
    q_base = b * (1 * 4 * K)  # heads dimension is size 1, so total K elements for head 0
    k_base = b * (1 * 4 * K)
    v_base = b * (1 * 8 * V)  # heads dimension is size 1, so total V elements for head 0

    # Load q_h, k_h, v_h (heads are 0 in dim-1 due to input shapes)
    q_h = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        q_j = tl.load(q_ptr + q_base + h * K + j)
        q_h[j] = q_j

    k_h = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        k_j = tl.load(k_ptr + k_base + h * K + j)
        k_h[j] = k_j

    v_h = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        v_j = tl.load(v_ptr + v_base + h * V + v_idx)
        v_h[v_idx] = v_j

    # Load g_val and beta_val for this (b, h)
    g_val = tl.load(g_ptr + b * H + h)
    beta_val = tl.load(beta_ptr + b * H + h)

    # Load state_old slice: [V, K] starting at state_ptr[b, h, :, :]
    # state layout is contiguous [B, H, V, K] => linear index = b*(H*V*K) + h*(V*K) + v_idx*K + k_idx
    state_old = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = b * (H * V * K) + h * V * K + v_idx * K
        for k_idx in range(0, K):
            state_val = tl.load(state_ptr + row_base + k_idx)
            state_old[v_idx, k_idx] = state_val

    # Compute old_v = k_h @ (g * state_old) -> (K,)
    old_v = tl.zeros([K], dtype=tl.float32)
    for k_idx in range(0, K):
        sum_val = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_val += state_old[v_idx, k_idx] * g_val
        old_v[k_idx] = sum_val * k_h[k_idx]

    # Compute new_v = beta * v_h + (1 - beta) * old_v -> (V,)
    new_v = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v[v_idx] = beta_val * v_h[v_idx] + (1.0 - beta_val) * old_v[v_idx]

    # Compute state_remove and state_update: scalars k_h @ old_v and k_h @ new_v
    state_remove = tl.zeros((), dtype=tl.float32)
    state_update = tl.zeros((), dtype=tl.float32)
    for k_idx in range(0, K):
        state_remove += old_v[k_idx] * k_h[k_idx]
        state_update += new_v[k_idx] * k_h[k_idx]

    # Update h_state_new elementwise: h_state_new = (g * state_old) - state_remove + state_update
    h_state_new = state_old * g_val
    h_state_new = h_state_new - state_remove + state_update

    # Compute output scalar: output = scale * q_h @ h_state_new
    out_scalar = tl.zeros((), dtype=tl.float32)
    for k_idx in range(0, K):
        row_k = h_state_new[:, k_idx]  # [V] vector
        out_scalar += q_h[k_idx] * tl.sum(row_k)

    # Store output at [b, h] (scalar per (b,h))
    tl.store(output_ptr + b * H + h, out_scalar * scale)

    # Store new_state slice: write h_state_new back to new_state_ptr at (b,h)
    for v_idx in range(0, V):
        row_base = b * (H * V * K) + h * V * K + v_idx * K
        for k_idx in range(0, K):
            tl.store(new_state_ptr + row_base + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Inputs:
        # q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128] (float32)
        # A_log: [8], a: [B,1,8], dt_bias: [8], b: [B,1,8], scale: float or None
        B = q.size(0)
        H = b.size(-1)  # number of heads (num_v_heads), i.e., 8
        device = q.device

        # Compute g and beta with Triton kernel: outputs are [B, 1, H] float32
        g = torch.empty((B, 1, H), device=device, dtype=torch.float32)
        beta = torch.empty((B, 1, H), device=device, dtype=torch.float32)

        # Launch kernel to compute g and beta
        triton_gate_beta_kernel[(B * H,)](
            A_log.to(torch.float32),
            a.to(torch.float32),
            dt_bias.to(torch.float32),
            b.to(torch.float32),
            g,
            beta,
            B=B, H=H,
        )

        # Compute scale = 1 / sqrt(K) on host (Python), pass as float
        K = q.size(-1)  # 128
        scale_val = 1.0 / math.sqrt(K)

        # Allocate output and new state
        output = torch.empty((B, H), device=device, dtype=torch.float32)
        new_state = state.clone()  # state is float32, keep float32

        # Launch Triton kernel to compute outputs and updated state for each (b, h)
        # Cast inputs to float32 for kernel math (q, k, v)
        triton_update_output_kernel[(B * H,)](
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            state,                     # read-only
            g, beta,
            output,
            new_state,                 # write updated state
            B=B, H=H, V=128, K=128,
            scale=scale_val,
        )

        # Return output (bfloat16, unsqueezed to [B,1,H]) and new_state (float32)
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
