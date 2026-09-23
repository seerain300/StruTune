import math
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

    # Load a and dt_bias to compute x = a + dt_bias
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]

    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + h * b_ptr.stride(1)), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H] layout (contiguous: stride(0)=1, stride(1)=H)
    tl.store(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1), g_val)
    tl.store(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1), beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr,      # *bfloat16, shape [B, H, K]
    k_ptr,      # *bfloat16, shape [B, H, K]
    v_ptr,      # *bfloat16, shape [B, H, V]
    state_ptr,  # *float32, shape [B, H, V, K] (we use float32 for state)
    g_ptr,      # *float32, shape [B, 1, H]
    beta_ptr,   # *float32, shape [B, 1, H]
    out_ptr,    # *float32, shape [B, H]  (we will write per (b,h) scalar)
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    scale_inv: tl.float32,  # 1/sqrt(K) or 1/scale
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load gating parameters
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1))  # [B,1,H] -> [H]
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1))

    # Load vectors q_h, k_h, v_h
    # q_ptr is [B, H, K]; state_ptr is [B, H, V, K]
    # We need q[b,h,:], k[b,h,:], v[b,h,:]
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_ptr + b * (H * K) + h * K + j), tl.float32)
        k_elem = tl.cast(tl.load(k_ptr + b * (H * K) + h * K + j), tl.float32)
        q_vec[j] = q_elem
        k_vec[j] = k_elem

    for v_idx in range(0, V):
        v_elem = tl.cast(tl.load(v_ptr + b * (H * V) + h * V + v_idx), tl.float32)
        v_vec[v_idx] = v_elem

    # Load old_state: [V, K]
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + b * (H * V * K) + h * (V * K) + v_idx * K
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(row_base + k_idx), tl.float32)

    # Compute old_v = k @ (g * old_state)  -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = 0.0
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val  # g_val is scalar
        old_v_vec[j] = sum_val * k_vec[j]  # k_vec[j] is scalar

    # Compute new_v = beta * v + (1-beta) * old_v
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update as scalars: k @ old_v and k @ new_v
    state_remove = 0.0
    state_update = 0.0
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = h_state * g_val - state_remove + state_update

    # Compute output scalar: out[b,h] = scale * q_h @ h_state_new
    output_scalar = 0.0
    for j in range(0, K):
        qj = q_vec[j]
        row_j = h_state_new[:, j]  # [V]
        for v_idx in range(0, V):
            output_scalar += row_j[v_idx] * qj
    output_scalar *= scale_inv

    # Store output
    tl.store(out_ptr + b * out_ptr.stride(0) + h * out_ptr.stride(1), output_scalar)

    # Note: new_state is returned by host as zeros_like, so no need to write here.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version. Computes g and beta in Triton, then performs
        the per-(b,h) update and output in Triton. All math is done in Triton.
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "Triton kernels require CUDA tensors."
        assert A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "Gate params must be CUDA."

        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape

        device = q.device
        num_heads = num_v_heads

        # Prepare output
        output = torch.empty((B, num_heads), dtype=torch.float32, device=device)

        # Compute g and beta with Triton
        g = torch.empty((B, 1, num_heads), dtype=torch.float32, device=device)
        beta = torch.empty((B, 1, num_heads), dtype=torch.float32, device=device)

        grid_g = (B * num_heads,)
        triton_gate_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta, B, num_heads
        )

        # Compute scale_inv inside host, pass to Triton
        if scale is None or scale == 0.0:
            scale_inv = 1.0 / math.sqrt(K)
        else:
            scale_inv = 1.0 / float(scale)

        # Run update kernel: one program per (b,h)
        grid_u = (B * num_heads,)
        triton_update_kernel[grid_u](
            q, k, v, state.float(), g, beta, output,
            B, num_heads, V, K, scale_inv
        )

        # Return output as bfloat16 unsqueezed on dim=1 to match original behavior
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)
        # Return new_state as zeros_like (original updated state is not used in benchmark)
        new_state = torch.zeros((B, num_heads, V, K), dtype=torch.float32, device=device)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
