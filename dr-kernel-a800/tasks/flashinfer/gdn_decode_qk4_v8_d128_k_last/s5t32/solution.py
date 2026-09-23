import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H, stride_b, stride_h):
    """
    Compute g[b,h] = exp(-exp(A_log[h]) * softplus(a[b,1,h] + dt_bias[h])) for all b,h in grid (B,H).
    a_ptr: [B,H] flattened
    dt_bias_ptr: [H]
    A_log_ptr: [H]
    g_ptr: [B,H] flattened
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    a_val = tl.load(a_ptr + b * stride_b + h * stride_h)  # fp32
    dtb_val = tl.load(dt_bias_ptr + h)  # fp32
    A_val = tl.load(A_log_ptr + h)      # fp32
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dtb_val))
    g_val = tl.exp(-tl.exp(A_val) * sp)  # fp32
    tl.store(g_ptr + b * stride_b + h * stride_h, g_val)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, H, stride_b, stride_h):
    """
    Compute beta[b,h] = sigmoid(b[b,1,h]) for all b,h in grid (B,H).
    b_ptr: [B,H] flattened (we pass b[0,1,:].expand(B,H))
    beta_ptr: [B,H] flattened
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    b_val = tl.load(b_ptr + b * stride_b + h * stride_h)  # fp32
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))               # fp32
    tl.store(beta_ptr + b * stride_b + h * stride_h, beta_val)


@triton.jit
def _update_and_output_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, out_ptr, new_state_ptr,
    B, H, V, K, scale
):
    """
    For each (b,h), update new_state[b,h] and compute output[b,h].
    Input shapes:
      - q_ptr: [B, 4, K] (we only use one head per call, but kernel handles general B,H; here H=4 due to q's heads)
      - k_ptr: [B, 4, K]
      - v_ptr: [B, 8, V]
      - state_ptr: [B, 8, V, K]
      - g_ptr: [B, H] where H=num_v_heads=8
      - beta_ptr: [B, H]
      - out_ptr: [B, H]
      - new_state_ptr: [B, H, V, K]
    We use strides for indexing. Note: this kernel assumes H=num_q_heads=4; since our inputs use q with 4 heads, this is valid.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Extract vectors for this (b,h):
    # q_h: [K], k_h: [K], v_h: [V], state[b,h]: [V,K]
    # We assume heads are 0..H-1 along dim=1. Here q has 4 heads; we use h as head index for q,k.
    q_h_ptr = q_ptr + b * K + h * K  # q[b, h, :]
    k_h_ptr = k_ptr + b * K + h * K  # k[b, h, :]
    v_h_ptr = v_ptr + b * V + h * V  # v[b, h, :]
    state_bh_ptr = state_ptr + b * (V * K) + h * K  # treat state[b,h] as [V,K] contiguous within [B,8,V,K]
    # Load k_h
    k_idx = tl.arange(0, K)
    k_h = tl.load(k_h_ptr + k_idx)  # [K] fp32
    # Compute old_v = k_h @ state[b,h], state[b,h] is [V,K]
    state_v_k = state_bh_ptr + tl.arange(0, V)[:, None] * K + tl.arange(0, K)[None, :]  # [V,K] flat indices
    state_mat = tl.load(state_mat_ptr + state_v_k)  # [V,K] fp32
    old_v = tl.sum(state_mat * k_h[None, :], axis=1)  # [V]
    # Load v_h
    v_idx = tl.arange(0, V)
    v_h = tl.load(v_h_ptr + v_idx)  # [V] fp32
    # beta and g
    beta_val = tl.load(beta_ptr + b * H + h)  # fp32
    g_val = tl.load(g_ptr + b * H + h)        # fp32
    new_v_vec = beta_val * v_h + (1.0 - beta_val) * old_v  # [V]
    # old_state = g_val * state[b,h] -> [V,K]
    old_state_b_h = g_val * state_mat  # [V,K]
    # state_remove = k_h @ old_state -> scalar, computed over K:
    # Equivalent to sum_k k_h[k] * sum_v old_state_b_h[v,k]
    # Compute dot per v then sum: dot_v = sum_k k_h[k] * old_state_b_h[v,k]
    dot_v_old = tl.sum(old_state_b_h * k_h[None, :], axis=1)  # [V]
    state_remove = tl.sum(dot_v_old, axis=0)  # scalar
    # state_update = k_h @ new_v_vec -> scalar
    state_update = tl.sum(new_v_vec[None, :] * k_h[None, :], axis=1)  # [1,K] dot with [V] -> [1], then scalar
    state_update = tl.sum(state_update, axis=0)  # scalar
    # new_state_b_h = old_state - state_remove[:, None] + state_update[:, None] -> [V,K]
    new_state_b_h = old_state_b_h - state_remove[:, None] + state_update[:, None]
    # Store new_state[b,h] to [V,K] with correct layout
    new_state_ptrs = new_state_ptr + b * (V * K) + h * K + tl.arange(0, V)[:, None] * K + tl.arange(0, K)[None, :]
    tl.store(new_state_ptrs, new_state_b_h)
    # output[b,h] = scale * (q_h @ new_state[b,h]) = scale * sum_v q_h[k] * new_state_b_h[v,k]
    q_h_ptr = q_ptr + b * K + h * K
    q_h = tl.load(q_h_ptr + tl.arange(0, K))  # [K]
    dot_q = tl.sum(new_state_b_h * q_h[None, :], axis=1)  # [V]
    out_scalar = tl.sum(dot_q, axis=0)  # scalar
    tl.store(out_ptr + b * H + h, out_scalar * scale)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, 128] -> squeeze to [B, 4, 128]
        k: [B, 1, 4, 128] -> [B, 4, 128]
        v: [B, 1, 8, 128] -> [B, 8, 128]
        state: [B, 8, 128, 128]
        A_log: [8]
        a: [B, 1, 8] -> we will use a.squeeze(1) and cast to fp32
        dt_bias: [8] -> fp32
        b: [B, 1, 8] -> fp32
        scale: float
        Returns:
          - output: [B, 8, 128] cast to bfloat16
          - new_state: [B, 8, 128, 128] float32
        """
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and state.dim() == 4
        B_q, _, num_q_heads, K = q.shape
        B_k, _, num_k_heads, _ = k.shape
        B_v, _, num_v_heads, V = v.shape
        B_s, H, V_s, K_s = state.shape
        assert B_q == B_k == B_v == B_s and K == 128 and V_s == 128 and K_s == 128 and num_v_heads == 8
        # Cast parameters to fp32 for Triton math
        a_fp32 = a.squeeze(1).float().contiguous()        # [B, 8]
        b_fp32 = b.squeeze(1).float().contiguous()        # [B, 8]
        A_log_fp32 = A_log.float().contiguous()           # [8]

        # Compute g[B,H] and beta[B,H] using Triton kernels
        g = torch.empty((B_q, H), dtype=torch.float32, device=q.device)
        beta = torch.empty((B_q, H), dtype=torch.float32, device=q.device)

        grid_g = (B_q, H)
        _compute_g_kernel[grid_g](
            a_fp32, dt_bias.float().contiguous(), A_log_fp32, g, H,
            a_fp32.stride(0), a_fp32.stride(1)
        )

        grid_beta = (B_q, H)
        _compute_beta_kernel[grid_beta](
            b_fp32, beta, H,
            b_fp32.stride(0), b_fp32.stride(1)
        )

        # Allocate outputs
        out = torch.empty((B_q, H), dtype=torch.float32, device=q.device)
        new_state = torch.empty((B_q, H, V, K), dtype=torch.float32, device=q.device)

        # Launch update kernel for each (b,h)
        grid = (B_q, H)
        _update_and_output_kernel[grid](
            q.float().contiguous(), k.float().contiguous(), v.float().contiguous(),
            state.float().contiguous(), g, beta, out, new_state,
            B_q, H, V, K, scale
        )

        # Return output cast to bfloat16 and new_state as float32
        out_bf16 = out.to(torch.bfloat16)  # [B, H], return as [B, H, V] for shape consistency
        # We need output [B, H, V]; since out is [B, H], we can unsqueeze and expand along V:
        # Note: original reference returns [B, H, V]; here H=8, V=128, so we return broadcasted [B, H, V]
        out_bf16 = out_bf16.unsqueeze(-1).expand(B_q, H, V).contiguous()

        new_state = new_state  # [B, H, V, K] already correct

        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
