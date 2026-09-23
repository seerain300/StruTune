import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16 or float32, shape [B, 1, H], indexed by (b,h)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H], indexed by (b,h)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a[b, 0, h] and dt_bias[h], compute x = a + dt_bias
    a_val = tl.cast(tl.load(a_ptr + b * H + h), tl.float32)  # a_ptr is contiguous over B*H
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log[h]) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)  # A_log is [H]
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b[h]) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * H + h), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H] contiguous layout
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def triton_invsqrt_scale(K: tl.constexpr):
    inv = 1.0 / tl.sqrt(K)
    return inv  # scalar return


@triton.jit
def triton_update_kernel(
    q_ptr,             # *bfloat16 or float32, shape [B, H, K]
    k_ptr,             # *bfloat16 or float32, shape [B, H, K]
    v_ptr,             # *bfloat16 or float32, shape [B, H, V]
    state_ptr,         # *float32, shape [B, H, V, K] (k-last)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    output_ptr,        # *float32, shape [B, H]
    new_state_ptr,     # *float32, shape [B, H, V, K]
    scale,             # float32 scalar (1/sqrt(K))
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
    V: tl.constexpr,   # V dimension
    K: tl.constexpr,   # K dimension
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load scalar parameters
    g_val = tl.load(g_ptr + b * H + h)
    beta_val = tl.load(beta_ptr + b * H + h)

    # Base offsets for this (b, h)
    q_base = b * H * K + h * K
    k_base = b * H * K + h * K
    v_base = b * H * V + h * V
    state_base = b * (H * V * K) + h * V * K

    # Indices vectors
    k_idx = tl.arange(0, K)  # [K]
    v_idx = tl.arange(0, V)  # [V]

    # Load q[b,h,:] and k[b,h,:]
    q_vec = tl.load(q_ptr + q_base + k_idx)  # [K]
    k_vec = tl.load(k_ptr + k_base + k_idx)  # [K]
    q_vec = tl.cast(q_vec, tl.float32)
    k_vec = tl.cast(k_vec, tl.float32)

    # Load v[b,h,:]
    v_vec = tl.load(v_ptr + v_base + v_idx)  # [V]
    v_vec = tl.cast(v_vec, tl.float32)

    # Load state_old: [V, K] contiguous
    h_state = tl.load(state_ptr + state_base + v_idx[:, None] * K + k_idx[None, :])  # [V, K]
    h_state = tl.cast(h_state, tl.float32)

    # Compute old_v = k @ (g * state_old) -> (K,)
    # g is scalar, so multiply elementwise
    g_state = h_state * g_val
    # Multiply by k_vec, then reduce across rows (axis=0) -> [K]
    prod = g_state * k_vec[None, :]  # [V, K]
    old_v_vec = tl.sum(prod, axis=0)  # [K]

    # Compute new_v = beta * v + (1 - beta) * old_v -> (V,)
    new_v_vec = beta_val * v_vec + (1.0 - beta_val) * old_v_vec[v_idx]  # broadcast old_v over V

    # Compute state_remove and state_update: scalars k @ old_v and k @ new_v
    state_remove = tl.sum(old_v_vec * k_vec, axis=0)
    state_update = tl.sum(new_v_vec * k_vec, axis=0)

    # Update h_state_new elementwise: h_state_new = (g * state_old) - state_remove + state_update
    # Broadcast scalars: state_remove and state_update are [1], g_val is scalar
    h_state_new = g_state - state_remove + state_update  # [V, K]

    # Compute output scalar: output = scale * q @ h_state_new
    # q @ h_state_new = sum_j q_j * (sum_i h_state_new[i, j])
    col_sums = tl.sum(h_state_new, axis=0)  # [K]
    output_scalar = tl.sum(q_vec * col_sums, axis=0)
    output_scalar = scale * output_scalar

    # Store output[b, h]
    tl.store(output_ptr + b * H + h, output_scalar)

    # Store updated state for [b, h, :, :] in new_state_ptr (float32, contiguous)
    new_state_base = b * (H * V * K) + h * V * K
    tl.store(new_state_ptr + new_state_base + v_idx[:, None] * K + k_idx[None, :], h_state_new)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
        A_log: [8], a: [B, 1, 8] (bfloat16), dt_bias: [8] (float32), b: [B, 1, 8] (bfloat16), scale: float or None
        """
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and state.dim() == 4
        assert q.shape[1] == 1 and k.shape[1] == 1 and v.shape[1] == 1
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        Bst, H, Vst, Kst = state.shape
        assert T == 1 and Bst == B and H == num_v_heads and Vst == V and Kst == K

        device = q.device

        # Compute g and beta with Triton kernel
        g = torch.empty((B, 1, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, 1, H), dtype=torch.float32, device=device)

        grid_g = (B * H,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, H
        )

        # Compute 1/sqrt(K) in Triton and pass as scalar
        invK = triton_invsqrt_scale[K]  # K is a constexpr here; we specialize at launch

        # Allocate output and new_state
        output = torch.empty((B, H), dtype=torch.float32, device=device)
        new_state = torch.empty_like(state, dtype=torch.float32, device=device)

        # Launch update kernel: one program per (b, h)
        grid_u = (B * H,)
        triton_update_kernel[grid_u](
            q, k, v, state, g, beta, output, new_state, float(invK), B, H, V, K
        )

        # Return output as bfloat16 unsqueezed and new_state in float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
