import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a[0,0,h] + dt_bias[h])) for h in [0..H-1]
    a_ptr: [H] (float32)
    dt_bias_ptr: [H] (float32)
    A_log_ptr: [H] (float32)
    g_ptr: [H] (float32)
    """
    h = tl.program_id(0)
    a_val = tl.load(a_ptr + h)
    dt_val = tl.load(dt_bias_ptr + h)
    A_val = tl.load(A_log_ptr + h)
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * sp)
    tl.store(g_ptr + h, g_val)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, H):
    """
    Compute beta[h] = sigmoid(b[0,0,h]) for h in [0..H-1]
    b_ptr: [H] (float32)
    beta_ptr: [H] (float32)
    """
    h = tl.program_id(0)
    b_val = tl.load(b_ptr + h)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + h, beta_val)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, out_ptr,
    B, H, V, K, scale
):
    """
    For each (b, h), update new_state[b,h] and compute output[b,h].
    q_ptr: [B, H, K] (float32), linearized as [B*H, K]
    k_ptr: [B, H, K] (float32), linearized as [B*H, K]
    v_ptr: [B, H, V] (float32), linearized as [B*H, V]
    state_ptr: [B, H, V, K] (float32), we treat as [B*H, V*K] for reading/writing
    g_ptr: [H] (float32)
    beta_ptr: [H] (float32)
    out_ptr: [B*H] (float32)
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Base offsets for q/k/v
    q_base = b * (H * K) + h * K
    k_base = b * (H * K) + h * K
    v_base = b * (H * V) + h * V

    # Load q and k vectors (length K)
    q_vec = [0.0] * K
    k_vec = [0.0] * K
    for i in range(0, K):
        q_vec[i] = tl.load(q_ptr + q_base + i)
        k_vec[i] = tl.load(k_ptr + k_base + i)

    # Load params
    g_val = tl.load(g_ptr + h)
    beta_val = tl.load(beta_ptr + h)

    # Load state[b,h] as [V, K] from state_ptr treated as [B*H, V*K]
    state_base = b * (H * V * K) + h * (V * K)
    old_state = [[0.0] * K for _ in range(0, V)]
    for i in range(0, V):
        for k in range(0, K):
            old_state[i][k] = tl.load(state_ptr + state_base + i * K + k)

    # old_v = k @ state_old (reduce over K)
    old_v = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_vec[k] * old_state[i][k]
        old_v[i] = sum_k

    # new_v = beta * v + (1 - beta) * old_v
    new_v = [0.0] * V
    for i in range(0, V):
        v_elem = tl.load(v_ptr + v_base + i)  # v[b,h,i]
        new_v[i] = beta_val * v_elem + (1.0 - beta_val) * old_v[i]

    # state_remove = k @ old_state (reduce over K)
    state_remove = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_vec[k] * old_state[i][k]
        state_remove[i] = sum_k

    # state_update = k @ new_v (reduce over K)
    state_update = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_vec[k] * new_v[i]
        state_update[i] = sum_k

    # Write new_state[b,h] into state_ptr at offset b*H*V*K + h*V*K -> base for new state
    new_state_base = b * (H * V * K) + h * (V * K)
    for i in range(0, V):
        for k in range(0, K):
            val = old_state[i][k] - state_remove[i] + state_update[i]
            tl.store(state_ptr + new_state_base + i * K + k, val)

    # Compute output[b,h] = scale * sum_i q[i] * (sum_k new_state[i,k])
    out_val = 0.0
    for i in range(0, V):
        row_sum = 0.0
        for k in range(0, K):
            row_sum += state_ptr[new_state_base + i * K + k]
        out_val += q_vec[i] * row_sum
    out_val *= scale
    tl.store(out_ptr + b * H + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only forward:
        - Compute g and beta via Triton kernels
        - Update state and compute output via Triton kernel
        Returns (output [B,H] cast to bfloat16), new_state [B,H,V,K] float32
        """
        # Identify device and extract tensors; default scalar scale if provided
        device = None
        for t in (q, k, v, state, A_log, a, dt_bias, b):
            if isinstance(t, torch.Tensor):
                if device is None:
                    device = t.device
                if t.shape == ():
                    # scalar tensor
                    scale = t.item()
            elif isinstance(t, (int, float)):
                if scale is None:
                    scale = t

        # Ensure we have the required tensors
        assert q is not None and k is not None and v is not None and state is not None and \
               A_log is not None and a is not None and dt_bias is not None and b is not None and \
               scale is not None, "Failed to identify required tensors or scalar in inputs"

        # Squeeze T=1
        q = q.squeeze(1)       # [B, 4, K]
        k = k.squeeze(1)       # [B, 4, K]
        v = v.squeeze(1)       # [B, 8, V]

        B = q.shape[0]
        H = 8                  # num_v_heads
        V = 128
        K = q.shape[2]

        # Cast all inputs to fp32 for Triton kernels
        q32 = q.to(torch.float32)        # [B, 4, K]
        k32 = k.to(torch.float32)        # [B, 4, K]
        v32 = v.to(torch.float32)        # [B, 8, V]
        state32 = state.to(torch.float32)  # [B, 8, V, K]

        # Expand q/k to H=8 heads (host-side repeat_interleave) to match original run behavior
        q32_exp = q32.repeat_interleave(H // 4, dim=1)  # [B, 8, K]
        k32_exp = k32.repeat_interleave(H // 4, dim=1)  # [B, 8, K]

        # Prepare parameter vectors [H] (A_log, a[:,0,:], dt_bias, b[:,0,:]) as fp32
        A_log32 = A_log.to(torch.float32)             # [8]
        a32 = a.to(torch.float32)[:, 0, :]            # [B, 8]
        dt_bias32 = dt_bias.to(torch.float32)         # [8]
        b32 = b.to(torch.float32)[:, 0, :]            # [B, 8]

        # Allocate g and beta as fp32 tensors of size H
        g = torch.empty(H, dtype=torch.float32, device=device)
        beta = torch.empty(H, dtype=torch.float32, device=device)

        # Launch Triton kernels to compute g and beta
        grid_g = (H,)
        grid_beta = (H,)
        _compute_g_kernel[grid_g](a32[0, :], dt_bias32, A_log32, g, H)
        _compute_beta_kernel[grid_beta](b32[0, :], beta, H)

        # Allocate output [B*H] and new_state [B, H, V, K] as fp32
        out = torch.empty(B * H, dtype=torch.float32, device=device)  # [B,H] flattened
        new_state = torch.empty_like(state32)  # fp32

        # Launch Triton kernel to update state and compute output
        grid_update = (B, H)
        _update_all_kernel[grid_update](
            q32_exp, k32_exp, v32, new_state, g, beta, out,
            B, H, V, K, scale
        )

        # Reshape output to [B, H]
        output = out.view(B, H)
        # Return output cast to bfloat16 and new_state (fp32)
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
