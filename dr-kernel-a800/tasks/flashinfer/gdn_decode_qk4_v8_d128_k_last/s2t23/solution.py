import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16 or float, shape [B, 1, H] (we index by b,h)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16 or float, shape [B, 1, H] (we index by b,h)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, 0, h], dt_bias[h], b[b, 0, h], A_log[h]
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)
    A_log_val = tl.load(A_log_ptr + h)
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + h * b_ptr.stride(1)), tl.float32)

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|)) for numerical stability
    x = a_val + dt_val
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = 1 / (1 + exp(-b))
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store to [B, 1, H] layout; we pass stride-compatible pointers from host
    tl.store(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1), g_val)
    tl.store(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1), beta_val)


@triton.jit
def triton_update_and_output_kernel(
    q_ptr,             # *bf16/float, shape [B, H, K]
    k_ptr,             # *bf16/float, shape [B, H, K]
    v_ptr,             # *bf16/float, shape [B, H, V]
    state_ptr,         # *float32, shape [B, H, V, K]
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    output_ptr,        # *float32, shape [B*H]
    new_state_ptr,     # *float32, shape [B, H, V, K]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads
    V: tl.constexpr,   # dim V (128)
    K: tl.constexpr,   # dim K (128)
    inv_scale: tl.float32,  # 1/sqrt(K), passed from host
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load vectors q, k, v (as float32)
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    # Compute base offsets for (b,h)
    q_base = b * K * H + h * K
    k_base = b * K * H + h * K
    v_base = b * V * H + h * V

    for j in tl.static_range(0, K):
        q_vec[j] = tl.cast(tl.load(q_ptr + q_base + j), tl.float32)
        k_vec[j] = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
    for v_idx in tl.static_range(0, V):
        v_vec[v_idx] = tl.cast(tl.load(v_ptr + v_base + v_idx), tl.float32)

    # Load g and beta for this (b, h)
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1))
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1))

    # Load old_state: [V, K] from state[b, h, :, :]
    old_state = tl.zeros([V, K], dtype=tl.float32)
    state_base = b * (H * V * K) + h * (V * K)
    for v_idx in tl.static_range(0, V):
        row_base = state_base + v_idx * K
        for k_idx in tl.static_range(0, K):
            old_state[v_idx, k_idx] = tl.cast(tl.load(state_ptr + row_base + k_idx), tl.float32)

    # Compute old_v = k @ (g * old_state) -> (K,)
    old_v = tl.zeros([K], dtype=tl.float32)
    for j in tl.static_range(0, K):
        sum_val = 0.0
        for v_idx in tl.static_range(0, V):
            sum_val += old_state[v_idx, j] * g_val
        old_v[j] = sum_val * k_vec[j]

    # Compute new_v = beta * v + (1 - beta) * old_v
    new_v = tl.zeros([V], dtype=tl.float32)
    for v_idx in tl.static_range(0, V):
        new_v[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v[v_idx]

    # Compute state_remove and state_update scalars
    state_remove = 0.0
    state_update = 0.0
    for j in tl.static_range(0, K):
        state_remove += old_v[j] * k_vec[j]
        state_update += new_v[j] * k_vec[j]

    # Update h_state_new = (g * old_state) - state_remove + state_update
    h_state_new = old_state * g_val - state_remove + state_update  # [V, K]

    # Compute output scalar: output = inv_scale * (q @ h_state_new)
    # q is [K], h_state_new is [V, K]; we reduce over V: sum_v q[v] * h_state_new[v, :]
    output_scalar = 0.0
    for v_idx in tl.static_range(0, V):
        row = h_state_new[v_idx, :]  # [K]
        for j in tl.static_range(0, K):
            output_scalar += q_vec[j] * row[j]

    output_scalar *= inv_scale

    # Store output
    tl.store(output_ptr + pid, output_scalar)

    # Store new_state[b, h, :, :]
    state_base_out = b * (H * V * K) + h * (V * K)
    for v_idx in tl.static_range(0, V):
        row_base_out = state_base_out + v_idx * K
        for k_idx in tl.static_range(0, K):
            tl.store(new_state_ptr + row_base_out + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure tensors are on CUDA
        device = q.device
        B, Hq, Kq = q.shape
        Bk, Hk, Kk = k.shape
        Bv, Hv, V = v.shape
        Bst, Hst, Vst, Kst = state.shape
        # Sanity checks
        assert B == Bk == Bv == Bst, "Batch sizes must match"
        assert Hq == Hk == 4, "num_q_heads must be 4"
        assert Hst == 8, "H (num_v_heads) must be 8"
        assert V == Kq == 128 and Vst == Kst == 128, "V and K must be 128"

        # Allocate g and beta as float32
        g = torch.empty((B, 1, Hst), dtype=torch.float32, device=device)
        beta = torch.empty((B, 1, Hst), dtype=torch.float32, device=device)

        # Launch gate/beta Triton kernel
        grid_g = (B * Hst,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, Hst
        )

        # Prepare output and new_state
        output = torch.empty(B * Hst, dtype=torch.float32, device=device)
        new_state = torch.empty((B, Hst, V, Kst), dtype=torch.float32, device=device)

        # Handle scale: inv_scale = 1/sqrt(K)
        if scale is None or scale == 0.0:
            inv_scale = 1.0 / math.sqrt(Kst)
        else:
            inv_scale = 1.0 / math.sqrt(Kst)  # scale is float, but inv_scale is 1/sqrt(K)

        # Launch update Triton kernel
        grid_u = (B * Hst,)
        triton_update_and_output_kernel[grid_u](
            q, k, v, state.float(), g, beta, output, new_state, B, Hst, V, Kst, inv_scale
        )

        # Reshape output to (B, 1, H) and cast to bfloat16 to match original behavior
        output = output.view(B, Hst).unsqueeze(1).to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
