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
    g_ptr,           # *float32, shape [B * H] (flattened)
    beta_ptr,        # *float32, shape [B * H] (flattened)
    B: tl.constexpr, # batch size
    H: tl.constexpr, # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load parameters for this (b, h)
    a_val = tl.cast(tl.load(a_ptr + b * H + h), tl.float32)         # a[b, 0, h]
    dt_val = tl.load(dt_bias_ptr + h)                               # dt_bias[h]
    x = a_val + dt_val                                              # x = a + dt_bias

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)                             # A_log[h]
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * H + h), tl.float32)        # b[b, 0, h]
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B * H] flattened
    out_idx = b * H + h
    tl.store(g_ptr + out_idx, g_val)
    tl.store(beta_ptr + out_idx, beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,            # *bfloat16, shape [B, H, K]
    k_ptr,            # *bfloat16, shape [B, H, K]
    v_ptr,            # *bfloat16, shape [B, H, V]
    state_ptr,        # *float32, shape [B, H, V, K]
    g_ptr,            # *float32, shape [B * H] (flattened)
    beta_ptr,         # *float32, shape [B * H] (flattened)
    output_ptr,       # *float32, shape [B * H]
    new_state_ptr,    # *float32, shape [B * H * V * K]
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # number of heads
    V: tl.constexpr,  # V dimension
    K: tl.constexpr,  # K dimension
    scale: tl.constexpr,  # float32 scale (e.g., 1/sqrt(K))
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load vectors for this (b, h)
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    # Contiguous offsets for (b,h)
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
    for v_idx in range(0, V):
        row_base = state_ptr + b * H * V * K + h * V * K + v_idx * K
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(row_base + k_idx), tl.float32)

    # Load g and beta scalars for this (b,h)
    out_idx = b * H + h
    g_val = tl.load(g_ptr + out_idx)
    beta_val = tl.load(beta_ptr + out_idx)

    # Compute old_v = k @ (g * state_old)  -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val
        old_v_vec[j] = sum_val * k_vec[j]

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
    h_state_new = h_state * g_val - state_remove + state_update

    # Compute output scalar: output = scale * q_h @ h_state_new
    output_scalar = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        row_j = h_state_new[:, j]  # [V]
        sum_j = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_j += row_j[v_idx]
        output_scalar += q_vec[j] * sum_j

    output_scalar = scale * output_scalar

    # Store output to [B * H]
    tl.store(output_ptr + out_idx, output_scalar)

    # Store new_state to [B * H * V * K] at (b,h) slice
    base_new = b * H * V * K + h * V * K
    for v_idx in range(0, V):
        row_base_new = new_state_ptr + base_new + v_idx * K
        for k_idx in range(0, K):
            tl.store(row_base_new + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original run function.
        Returns:
        - output: (B, 1, H) bfloat16
        - new_state: (B, H, V, K) float32
        """
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        H = num_v_heads
        assert T == 1, "T must be 1"
        assert K == 128 and V == 128, "K and V must be 128"
        device = q.device
        dtype = q.dtype

        # Ensure inputs are on the same device and dtype
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        A_log = A_log.contiguous()
        a = a.contiguous()
        dt_bias = dt_bias.contiguous()
        b = b.contiguous()

        # Allocate g and beta flattened
        g = torch.empty(B * H, dtype=torch.float32, device=device)
        beta = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        grid_g = (B * H,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, H
        )

        # Prepare output and new_state
        output = torch.empty(B * H, dtype=torch.float32, device=device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Compute scale inside Triton (as constexpr float)
        inv_sqrtK = 1.0 / math.sqrt(K)

        # Launch Triton update kernel
        grid_u = (B * H,)
        triton_update_kernel[grid_u](
            q, k, v, state, g, beta, output, new_state, B, H, V, K, inv_sqrtK
        )

        # Reshape output to (B, 1, H) and cast to bfloat16 for return
        output = output.view(B, H).unsqueeze(1).to(torch.bfloat16)

        # new_state is float32 as required
        return output, new_state


def run(*args):
    return ModelNew()(*args)
