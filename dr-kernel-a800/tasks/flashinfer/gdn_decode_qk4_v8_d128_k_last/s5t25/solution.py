import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(dt_bias_ptr, A_log_ptr, a_ptr, g_ptr, H):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a_ptr[h] + dt_bias_ptr[h])) for h in [0..H-1]
    a_ptr: [H], float32
    dt_bias_ptr: [H], float32
    A_log_ptr: [H], float32
    g_ptr: [H], float32
    """
    h = tl.program_id(0)
    x = dt_bias_ptr[h] + a_ptr[h]
    sp = tl.log(1.0 + tl.exp(x))
    g = tl.exp(-tl.exp(A_log_ptr[h]) * sp)
    tl.store(g_ptr + h, g)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, H_total):
    """
    Compute beta[h] = sigmoid(b_ptr[h]) for h in [0..H_total-1]
    b_ptr: [H_total], float32 (we pass b[0, 1, :] flattened)
    beta_ptr: [H_total], float32
    """
    h = tl.program_id(0)
    b_val = tl.load(b_ptr + h)  # float32
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + h, beta_val)


@triton.jit
def _update_output_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    beta_ptr, g_ptr,
    out_ptr, new_state_ptr,
    B, H, V, K, scale,
):
    """
    One program per (b,h). Update new_state[b,h] and compute output[b,h].
    Tensors are pointers; no torch math here.
    q_ptr: [B,4,K]
    k_ptr: [B,4,K]
    v_ptr: [B,8,V]
    state_ptr: [B,H,V,K]
    beta_ptr: [H], float32
    g_ptr: [H], float32
    out_ptr: [B,H], float32
    new_state_ptr: [B*H*V*K], float32 (we store row-wise for each (b,h))
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load g[h] and beta[h]
    g_val = tl.load(g_ptr + h)  # scalar
    beta_val = tl.load(beta_ptr + h)  # scalar

    # Initialize new_state[b,h] as zeros [V,K]
    new_state_base = new_state_ptr + (b * H + h) * V * K

    # Prepare row accumulators for new_state
    for m in range(V):
        # old_state[m,:] = g * state[b,h, m, :]
        state_base = state_ptr + b * (H * V * K) + h * (V * K)
        row_ptr = state_base + m * K
        row = tl.load(row_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
        old_state_row = g_val * row
        # k @ old_state[m,:]
        old_v = 0.0
        for k_idx in range(4):
            k_k_ptr = k_ptr + b * (4 * K) + k_idx * K
            k_h = tl.load(k_k_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
            old_v += tl.sum(k_h * row)
        # k @ v_h
        v_base = v_ptr + b * (8 * V)
        v_h = tl.load(v_base + tl.arange(0, V), mask=tl.arange(0, V) < V, other=0.0)
        new_v = beta_val * v_h + (1.0 - beta_val) * old_v
        # Update new_state[m,:] = old_state_row - k @ old_state[m,:] + k @ new_v
        # Compute k @ new_v: sum over K of k_h * new_v[m]
        k_sum = 0.0
        for k_idx in range(4):
            k_k_ptr = k_ptr + b * (4 * K) + k_idx * K
            k_h = tl.load(k_k_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
            # new_v[m] is scalar; k @ new_v contributes new_v * sum_k k_h[k]
            k_sum += tl.sum(k_h)
        delta = -old_v + (new_v * k_sum)
        new_state_row = old_state_row + delta  # scalar delta applied to entire row
        # Store new_state[b,h, m, :]
        tl.store(new_state_base + m * K + tl.arange(0, K), new_state_row, mask=tl.arange(0, K) < K)

    # Compute output[b,h] = scale * (q_h @ new_state[b,h])
    # q_h is row vector of length K
    q_base = q_ptr + b * (4 * K)
    q_h = tl.zeros([K], dtype=tl.float32)
    for k_idx in range(4):
        q_k_ptr = q_base + k_idx * K
        q_row = tl.load(q_k_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
        q_h += q_row
    out_val = scale * tl.sum(q_h * tl.load(new_state_base + tl.arange(0, V * K), mask=tl.arange(0, V * K) < V * K, other=0.0))
    # Note: The above load of new_state to compute q_h @ new_state is problematic. Instead, we should iterate m to compute the dot. Let's correct this.

    # Correct approach: Iterate m in V, and for each m, q_h @ new_state[b,h, m,:] and accumulate.
    out_val = 0.0
    for m in range(V):
        row_ptr = new_state_base + m * K
        row = tl.load(row_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
        out_val += tl.sum(q_h * row)
    tl.store(out_ptr + b * H + h, out_val)


def _to_fp32(x):
    # Ensure float32 for Triton
    return x.float()


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, K], k: [B, 1, 4, K], v: [B, 1, 8, V], state: [B, 8, V, K], 
        A_log: [8], a: [B, 1, 8], b: [B, 1, 8], dt_bias: [8], scale: float
        Returns:
        - output: [B, H, V] in bfloat16
        - new_state: [B, H, V, K] in float32
        """
        # Ensure device and dtype for Triton
        B, _, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        H = num_v_heads
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "Inputs must be CUDA tensors for Triton."

        # Prepare 1D parameter vectors for Triton
        a32 = _to_fp32(a)  # [B, 1, H]
        dt_bias32 = _to_fp32(dt_bias)  # [H]
        A_log32 = _to_fp32(A_log)  # [H]
        b32 = _to_fp32(b)  # [B, 1, H]

        # Allocate outputs and new_state
        g = torch.empty(H, dtype=torch.float32, device=device)
        beta = torch.empty(H, dtype=torch.float32, device=device)
        out = torch.empty(B * H, dtype=torch.float32, device=device)
        new_state = torch.empty(B * H * V * K, dtype=torch.float32, device=device)

        # Launch Triton kernels: compute g and beta
        grid_g = (H,)
        _compute_g_kernel[grid_g](dt_bias32, A_log32, a32[0, 0, :], g, H)  # pass a[0,0,:] as [H]
        grid_beta = (H,)
        # b32 has shape [B,1,H]; pass b[0,1,:] to compute beta
        _compute_beta_kernel[grid_beta](b32[0, 0, :], beta, H)

        # Launch Triton kernel to compute output and new_state per (b,h)
        grid_out = (B, H)
        _update_output_kernel[grid_out](
            q, k, v, state,
            beta, g,
            out, new_state,
            B, H, V, K, scale,
        )

        # Reshape outputs
        output = out.view(B, H).to(torch.bfloat16)  # [B, H] in bfloat16
        # Reshape new_state to [B,H,V,K] float32
        new_state = new_state.view(B, H, V, K).contiguous()

        return output, new_state


def run(*args):
    return ModelNew()(*args)
