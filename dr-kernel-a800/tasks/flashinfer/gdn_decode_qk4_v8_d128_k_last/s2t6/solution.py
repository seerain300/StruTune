import math
import torch
import triton
import triton.language as tl


@triton.jit
def gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16, shape [B, 1, H]
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H]
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)  # one program per (b, h)
    b = pid // H
    h = pid % H

    # Load a[b, h] and dt_bias[h]
    a_val = tl.cast(tl.load(a_ptr + b * 1 * H + h), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)  # A_log is [H]
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * 1 * H + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H]
    tl.store(g_ptr + b * 1 * H + h, g_val)
    tl.store(beta_ptr + b * 1 * H + h, beta_val)


@triton.jit
def update_and_output_kernel(
    q_ptr,             # *bfloat16, shape [B, H, K]
    k_ptr,             # *bfloat16, shape [B, H, K]
    v_ptr,             # *bfloat16, shape [B, H, V]
    state_ptr,         # *float32,  shape [B, H, V, K]
    g_ptr,             # *float32,  shape [B, 1, H]
    beta_ptr,          # *float32,  shape [B, 1, H]
    output_ptr,        # *float32,  shape [B, H]
    new_state_ptr,     # *float32,  shape [B, H, V, K]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads
    V: tl.constexpr,   # num_v
    K: tl.constexpr,   # K
    inv_sqrtK: tl.constexpr,  # float32 scalar
):
    pid = tl.program_id(axis=0)  # one program per (b, h)
    b = pid // H
    h = pid % H

    # Load vectors q[h], k[h], v[h]
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    # Base offsets for (b,h)
    q_base = b * H * K + h * K
    k_base = b * H * K + h * K
    v_base = b * H * V + h * V

    for j in range(0, K):
        qj = tl.cast(tl.load(q_ptr + q_base + j), tl.float32)
        kj = tl.cast(tl.load(k_ptr + k_base + j), tl.float32)
        q_vec[j] = qj
        k_vec[j] = kj

    for j in range(0, V):
        v_vec[j] = tl.cast(tl.load(v_ptr + v_base + j), tl.float32)

    # Load state_old: [V, K]
    state_old = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = b * H * V * K + h * V * K + v_idx * K
        for k_idx in range(0, K):
            state_old[v_idx, k_idx] = tl.cast(tl.load(state_ptr + row_base + k_idx), tl.float32)

    # Gate scalar for this head
    g_val = tl.load(g_ptr + b * 1 * H + h)  # g_ptr shape [B, 1, H]
    beta_val = tl.load(beta_ptr + b * 1 * H + h)

    # Compute old_v = k @ (g * state_old) -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_val += state_old[v_idx, j] * g_val
        old_v_vec[j] = tl.sum(sum_val * k_vec[j])  # k_vec[j] is scalar

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

    # Update h_state_new elementwise: h_state_new = (g * state_old) - state_remove + state_update
    h_state_new = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            h_state_new[v_idx, k_idx] = state_old[v_idx, k_idx] * g_val - state_remove + state_update

    # Compute output scalar: output = inv_sqrtK * q @ h_state_new
    output_scalar = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        qj = q_vec[j]
        col_j = h_state_new[:, j]  # [V]
        sum_j = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_j += col_j[v_idx]
        output_scalar += qj * sum_j

    # Store output
    tl.store(output_ptr + b * H + h, output_scalar)

    # Store new_state[b, h, :, :]
    for v_idx in range(0, V):
        row_base = b * H * V * K + h * V * K + v_idx * K
        for k_idx in range(0, K):
            tl.store(new_state_ptr + row_base + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward that computes:
          - g = exp(-exp(A_log) * softplus(a + dt_bias))
          - beta = sigmoid(b)
          - updates new_state[b,h,:,:] and computes output[b,h] using Triton kernels
        Returns:
          - output: torch.Tensor, shape (B, 1, H), dtype bfloat16
          - new_state: torch.Tensor, shape (B, H, V, K), dtype float32
        """
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4 and state.ndim == 4
        assert A_log.ndim == 1 and a.ndim == 3 and dt_bias.ndim == 1 and b.ndim == 3
        B, Tq, Hq, K = q.shape
        Bk, Tk, Hk, K2 = k.shape
        Bv, Tv, Hv, V = v.shape
        Bst, Hst, Vst, Kst = state.shape
        assert Tq == 1 and Tk == 1 and Tv == 1
        assert Hq == 4 and Hk == 4 and Hv == 8
        assert K == 128 and V == 128 and Kst == 128 and Vst == 128
        assert B == Bst == Bk == Bq == Bv
        assert Hst == Hv == 8  # num_v_heads

        # Make tensors contiguous and float32 for computation where needed
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        state_c = state.contiguous().float()  # compute in float32

        # Allocate g and beta in float32 [B, 1, H]
        g = torch.empty((B, 1, Hst), dtype=torch.float32, device=q.device)
        beta = torch.empty((B, 1, Hst), dtype=torch.float32, device=q.device)

        # Compute grid for Triton kernels
        grid = (B * Hst,)

        # Triton: compute g and beta
        gate_beta_kernel[grid](
            A_log, a, dt_bias, b, g, beta, B, Hst
        )

        # Prepare output and new_state
        output = torch.empty((B, Hst), dtype=torch.float32, device=q.device)
        new_state = torch.empty((B, Hst, Vst, Kst), dtype=torch.float32, device=q.device)

        # Handle scale: scale_over_sqrtK = scale * (1/sqrt(K)) if scale given, else 1/sqrt(K)
        if scale is None or scale == 0.0:
            scale_over_sqrtK = 1.0 / math.sqrt(K)
        else:
            scale_over_sqrtK = float(scale) * (1.0 / math.sqrt(K))
        inv_sqrtK = 1.0 / math.sqrt(K)

        # Triton: perform updates and compute outputs
        update_and_output_kernel[grid](
            q_c, k_c, v_c, state_c, g, beta, output, new_state,
            B, Hst, Vst, Kst, inv_sqrtK
        )

        # Return output as bfloat16 unsqueezed to (B, 1, H), and new_state as float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
