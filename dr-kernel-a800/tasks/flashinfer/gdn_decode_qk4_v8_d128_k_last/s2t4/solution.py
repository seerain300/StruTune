import math
import torch
import triton
import triton.language as tl


@triton.jit
def gate_beta_kernel(
    A_log_ptr,        # *float32, shape [H]
    a_ptr,            # *bfloat16, shape [B, H]
    dt_bias_ptr,      # *float32, shape [H]
    b_ptr,            # *bfloat16, shape [B, H]
    g_ptr,            # *float32, shape [B, H]
    beta_ptr,         # *float32, shape [B, H]
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # number of heads
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a and dt_bias to compute x = a + dt_bias
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)
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

    # Store [B, H]
    tl.store(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1), g_val)
    tl.store(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1), beta_val)


@triton.jit
def update_kernel(
    q_ptr,        # *bfloat16, shape [B, H, K]
    k_ptr,        # *bfloat16, shape [B, H, K]
    v_ptr,        # *bfloat16, shape [B, H, V]
    state_ptr,    # *float32,  shape [B, H, V, K]
    g_ptr,        # *float32,  shape [B, H]
    beta_ptr,     # *float32,  shape [B, H]
    out_ptr,      # *float32,  shape [B, H]
    new_state_ptr,# *float32,  shape [B, H, V, K]
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    scale: tl.float32,
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load per-(b,h) scalars
    g_val = tl.load(g_ptr + b * g_ptr.stride(0) + h * g_ptr.stride(1))
    beta_val = tl.load(beta_ptr + b * beta_ptr.stride(0) + h * beta_ptr.stride(1))

    # Load q, k, v for this (b,h)
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_ptr + b * q_ptr.stride(0) + h * q_ptr.stride(1) + j * q_ptr.stride(2)), tl.float32)
        k_elem = tl.cast(tl.load(k_ptr + b * k_ptr.stride(0) + h * k_ptr.stride(1) + j * k_ptr.stride(2)), tl.float32)
        q_vec[j] = q_elem
        k_vec[j] = k_elem

    for i in range(0, V):
        v_elem = tl.cast(tl.load(v_ptr + b * v_ptr.stride(0) + h * v_ptr.stride(1) + i * v_ptr.stride(2)), tl.float32)
        v_vec[i] = v_elem

    # Load state_old: [V, K] in float32
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for i in range(0, V):
        row_base = state_ptr + b * state_ptr.stride(0) + h * state_ptr.stride(1) + i * state_ptr.stride(2)
        for j in range(0, K):
            h_state[i, j] = tl.cast(tl.load(row_base + j * state_ptr.stride(3)), tl.float32)

    # Compute old_v = k @ (g * state_old)  -> (K,)
    old_v = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = tl.zeros([], dtype=tl.float32)
        for i in range(0, V):
            sum_val += h_state[i, j] * g_val
        old_v[j] = sum_val

    # Compute new_v = beta * v + (1 - beta) * old_v  -> (V,)
    new_v = tl.zeros([V], dtype=tl.float32)
    for i in range(0, V):
        new_v[i] = beta_val * v_vec[i] + (1.0 - beta_val) * old_v[i]

    # Compute state_remove and state_update as scalars: k @ old_v and k @ new_v
    state_remove = tl.zeros([], dtype=tl.float32)
    state_update = tl.zeros([], dtype=tl.float32)
    for j in range(0, K):
        state_remove += old_v[j] * k_vec[j]
        state_update += new_v[j] * k_vec[j]

    # Update h_state_new = (g * state_old) - state_remove + state_update
    for i in range(0, V):
        for j in range(0, K):
            h_state[i, j] = h_state[i, j] * g_val - state_remove + state_update

    # Compute output scalar: output = scale * (q_h @ h_state_new)
    q_h_sum = tl.zeros([], dtype=tl.float32)
    for j in range(0, K):
        row_sum = tl.zeros([], dtype=tl.float32)
        for i in range(0, V):
            row_sum += h_state[i, j]
        q_h_sum += q_vec[j] * row_sum

    output_val = scale * q_h_sum

    # Store output and write new_state
    tl.store(out_ptr + b * out_ptr.stride(0) + h * out_ptr.stride(1), output_val)
    for i in range(0, V):
        row_base_new = new_state_ptr + b * new_state_ptr.stride(0) + h * new_state_ptr.stride(1) + i * new_state_ptr.stride(2)
        for j in range(0, K):
            tl.store(row_base_new + j * new_state_ptr.stride(3), tl.cast(h_state[i, j], tl.float32))


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward:
        - Computes g and beta with Triton.
        - Updates state and computes per-(b,h) output scalar with Triton.
        Returns:
          - output: bfloat16 of shape (B, 1, H) (unsqueezed to match original behavior)
          - new_state: float32 of shape (B, H, V, K)
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "Triton requires CUDA tensors"
        B, _, H, K = q.shape
        _, _, _, V = v.shape

        # Ensure contiguity for simple stride-based indexing
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        state_c = state.contiguous()

        # Allocate outputs for g, beta, output, and new_state
        g = torch.empty((B, H), dtype=torch.float32, device=q.device)
        beta = torch.empty((B, H), dtype=torch.float32, device=q.device)
        out = torch.empty((B, H), dtype=torch.float32, device=q.device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=q.device)

        # Launch Triton kernels
        grid = (B * H,)
        gate_beta_kernel[grid](A_log, a, dt_bias, b, g, beta, B, H)
        update_kernel[grid](q_c, k_c, v_c, state_c, g, beta, out, new_state, B, H, V, K, float(scale))

        # Return output as bfloat16 and unsqueezed to (B, 1, H) to match caller expectation
        output_bf16 = out.unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
