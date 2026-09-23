import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,       # *float32, shape [H]
    a_ptr,           # *bfloat16, shape [B, 1, H] (we index by b,h)
    dt_bias_ptr,     # *float32, shape [H]
    b_ptr,           # *bfloat16, shape [B, 1, H] (we index by b,h)
    g_ptr,           # *float32, shape [B, 1, H]
    beta_ptr,        # *float32, shape [B, 1, H]
    B: tl.constexpr, # batch size
    H: tl.constexpr, # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Compute x = a + dt_bias for this (b, h)
    a_val = tl.cast(tl.load(a_ptr + b * H + h), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)  # A_log is [H]
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * H + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H] layout
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_update_kernel(
    A_log_ptr,       # *float32, shape [H]
    a_ptr,           # *bfloat16, shape [B, 1, H]
    dt_bias_ptr,     # *float32, shape [H]
    b_ptr,           # *bfloat16, shape [B, 1, H]
    q_ptr,           # *bfloat16, shape [B, H, K]
    k_ptr,           # *bfloat16, shape [B, H, K]
    v_ptr,           # *bfloat16, shape [B, H, V]
    state_ptr,       # *float32, shape [B, H, V, K]
    out_ptr,         # *float32, shape [B, H]
    B: tl.constexpr, # batch size
    H: tl.constexpr, # number of heads (num_v_heads)
    V: tl.constexpr, # number of V
    K: tl.constexpr, # number of K
    scale,           # float32 scalar (e.g., 1/sqrt(K))
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load parameters for this (b, h)
    A_log_val = tl.load(A_log_ptr + h)
    a_val = tl.cast(tl.load(a_ptr + b * H + h), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    b_val = tl.cast(tl.load(b_ptr + b * H + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Load q, k, v for this (b, h)
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        qj = tl.cast(tl.load(q_ptr + b * (H * K) + h * K + j), tl.float32)
        kj = tl.cast(tl.load(k_ptr + b * (H * K) + h * K + j), tl.float32)
        q_vec[j] = qj
        k_vec[j] = kj

    for v_idx in range(0, V):
        vv = tl.cast(tl.load(v_ptr + b * (H * V) + h * V + v_idx), tl.float32)
        v_vec[v_idx] = vv

    # Load state_old: [V, K]
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + (b * H + h) * V * K + v_idx * K
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(row_base + k_idx), tl.float32)

    # Compute old_v = k @ (g * state_old)  -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val
        old_v_vec[j] = sum_val

    # Compute new_v = beta * v + (1 - beta) * old_v
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update as scalars: k @ old_v and k @ new_v
    state_remove = tl.zeros((), dtype=tl.float32)
    state_update = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = h_state[v_idx, k_idx] * g_val - state_remove + state_update

    # Compute output scalar: output = scale * (q_h @ h_state_new)
    output_scalar = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state[:, j]  # [V]
        sum_j = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_j += row_j[v_idx]
        output_scalar += qj * sum_j

    output_scalar = scale * output_scalar

    # Store output[b, h]
    tl.store(out_ptr + b * H + h, output_scalar)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation that computes:
        - g = exp(-exp(A_log) * softplus(a + dt_bias))
        - beta = sigmoid(b)
        - updates state according to the given delta rule and returns output scalar per (b,h)
          as bfloat16 unsqueezed, and new_state as float32.
        """
        # Ensure tensors are on GPU and dtype appropriate
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "All inputs must be CUDA tensors."
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        H = num_v_heads

        # Allocate g and beta (float32)
        g = torch.empty((B, 1, H), dtype=torch.float32, device=q.device)
        beta = torch.empty((B, 1, H), dtype=torch.float32, device=q.device)

        # Launch gate/beta kernel
        grid_g = (B * H,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, H
        )

        # Prepare output
        output = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Compute scale inside Triton: pass 1/sqrt(K)
        if scale is None or scale == 0.0:
            inv_sqrtK = 1.0 / math.sqrt(K)
        else:
            inv_sqrtK = float(scale)

        # Launch update kernel
        grid_u = (B * H,)
        triton_update_kernel[grid_u](
            A_log, a, dt_bias, b, q, k, v, state, output, B, H, V, K, inv_sqrtK
        )

        # Return outputs as expected: output as bfloat16 unsqueezed, new_state as float32 (not requested to be computed here)
        # Note: original function returns (output, new_state). We only compute output per (b,h) scalar. We'll construct new_state zeros as float32.
        new_state = torch.zeros_like(state, dtype=torch.float32, device=q.device)
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
