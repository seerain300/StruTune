import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H: tl.constexpr):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a_ptr[h] + dt_bias_ptr[h])) for h in [0..H-1]
    a_ptr, dt_bias_ptr, g_ptr: float32 1D vectors of length H
    A_log_ptr: float32 1D vector of length H
    """
    h = tl.program_id(0)
    x = dt_bias_ptr[h] + a_ptr[h]  # softplus: log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    g_val = tl.exp(-tl.exp(A_log_ptr[h]) * sp)
    tl.store(g_ptr + h, g_val)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, B_H: tl.constexpr):
    """
    Compute beta[h] = 1 / (1 + exp(-b_ptr[h])) for h in [0..B_H-1]
    b_ptr: float32 1D vector of length B_H
    beta_ptr: float32 1D vector of length B_H
    """
    h = tl.program_id(0)
    b_val = b_ptr[h]
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + h, beta_val)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_in_ptr,
    g_ptr, beta_ptr,
    new_state_ptr, out_ptr,
    B, H, V, K, scale
):
    """
    For each (b,h), update new_state and compute output:
    - new_state shape: [B,H,V,K] stored as linear buffer of length B*H*V*K
    - out shape: [B,H,V] stored as linear buffer of length B*H*V
    q_ptr: [B,4,K] flattened
    k_ptr: [B,4,K] flattened
    v_ptr: [B,8,V] flattened
    state_in_ptr: [B,8,V,K] flattened
    g_ptr: [H]
    beta_ptr: [H]
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Precompute bases
    q_base = q_ptr + b * (4 * K)
    state_base = state_in_ptr + b * (H * V * K)
    new_state_base = new_state_ptr + (b * H + h) * V * K
    out_base = out_ptr + b * (H * V)

    g_val = tl.load(g_ptr + h)
    beta_val = tl.load(beta_ptr + h)

    # Loop over m in V
    for m in tl.static_range(V):
        # Compute old_v = sum over k of k @ state[b,h,m,:] for all k in K
        old_v = 0.0
        for k_idx in tl.static_range(4):  # k per head index runs over 4 heads
            k_k_ptr = k_ptr + b * (4 * K) + k_idx * K
            k_h = tl.load(k_k_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)  # [K]
            row = tl.load(state_base + (h * V + m) * K + tl.arange(0, K),
                          mask=tl.arange(0, K) < K, other=0.0)  # [K]
            old_v += tl.sum(k_h * row)

        # Load v[b,h,m] as scalar
        v_m = tl.load(v_ptr + b * (8 * V) + (h * V + m), mask=True, other=0.0)

        # new_v = beta * v_m + (1 - beta) * old_v
        new_v = beta_val * v_m + (1.0 - beta_val) * old_v

        # Compute state_remove: sum_k k @ (g * state_row)
        state_remove = 0.0
        for k_idx in tl.static_range(4):
            k_k_ptr = k_ptr + b * (4 * K) + k_idx * K
            k_h = tl.load(k_k_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)  # [K]
            row = tl.load(state_base + (h * V + m) * K + tl.arange(0, K),
                          mask=tl.arange(0, K) < K, other=0.0)  # [K]
            state_remove += tl.sum(k_h * (g_val * row))

        # Compute state_update: sum_k k @ (beta*new_v + (1-beta)*old_v) which is scalar per row
        # sum_k k_h[k] per k_idx head
        k_sum = 0.0
        for k_idx in tl.static_range(4):
            k_k_ptr = k_ptr + b * (4 * K) + k_idx * K
            k_h = tl.load(k_k_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
            k_sum += tl.sum(k_h)
        # For row m: new_v is scalar, old_v is scalar; update = (beta*new_v + (1-beta)*old_v) * k_sum
        contrib = (beta_val * new_v + (1.0 - beta_val) * old_v) * k_sum

        # new_state row: old_state_row - state_remove + contrib
        row_old = tl.load(state_base + (h * V + m) * K + tl.arange(0, K),
                          mask=tl.arange(0, K) < K, other=0.0)  # [K]
        new_row = (g_val - 1.0) * row_old + contrib  # adjust formula if needed; here we mimic removal and delta
        tl.store(new_state_base + m * K + tl.arange(0, K),
                 new_row, mask=tl.arange(0, K) < K)

        # output[b,h,m] = scale * q[b,h,:] @ new_row
        q_h = tl.load(q_ptr + b * (4 * K) + h * K + tl.arange(0, K),
                      mask=tl.arange(0, K) < K, other=0.0)  # [K]
        out_val = tl.sum(q_h * new_row) * scale
        tl.store(out_base + m, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the Gated Delta Net decode.
        q: [B, 1, 4, K] -> squeeze
        k: [B, 1, 4, K] -> squeeze
        v: [B, 1, 8, V] -> squeeze
        state: [B, 8, V, K]
        A_log: [8]
        a: [B, 1, 8]
        dt_bias: [8]
        b: [B, 1, 8]
        scale: float
        Returns:
        - output: [B, H, V] in bfloat16
        - new_state: [B, H, V, K] in float32
        """
        B, _, Hq, K = q.shape
        _, _, Hv, V = v.shape
        assert Hq == 4 and Hv == 8
        assert k.shape == (B, 1, 4, K)
        assert state.shape == (B, 8, V, K)

        # Cast parameters to float32 and flatten correctly
        a_f32 = a.to(torch.float32).index_select(2, torch.arange(Hv)).flatten()  # [B*Hv]
        dt_bias_f32 = dt_bias.to(torch.float32)  # [Hv]
        A_log_f32 = A_log.to(torch.float32)      # [Hv]
        b_f32 = b.to(torch.float32).index_select(2, torch.arange(Hv)).flatten()  # [B*Hv]

        # Allocate outputs for g and beta
        g = torch.empty(Hv, dtype=torch.float32, device=q.device)
        beta = torch.empty(B * Hv, dtype=torch.float32, device=q.device)

        # Launch Triton kernels to compute g and beta
        grid_g = (Hv,)
        _compute_g_kernel[grid_g](a_f32, dt_bias_f32, A_log_f32, g, Hv, num_warps=1)
        grid_beta = (B * Hv,)
        _compute_beta_kernel[grid_beta](b_f32, beta, B * Hv, num_warps=1)

        # Prepare inputs for the update kernel
        q_flat = q.squeeze(1).to(torch.float32).reshape(B * 4 * K)  # [B*4*K]
        k_flat = k.squeeze(1).to(torch.float32).reshape(B * 4 * K)  # [B*4*K]
        v_flat = v.squeeze(1).to(torch.float32).reshape(B * 8 * V)  # [B*8*V]
        state_flat = state.to(torch.float32).reshape(B * 8 * V * K)  # [B*8*V*K]

        out_flat = torch.empty(B * Hv * V, dtype=torch.float32, device=q.device)
        new_state_flat = torch.empty(B * Hv * V * K, dtype=torch.float32, device=q.device)

        # Launch Triton kernel to compute new_state and out
        grid_update = (B, Hv)
        _update_all_kernel[grid_update](
            q_flat, k_flat, v_flat, state_flat,
            g, beta,
            new_state_flat, out_flat,
            B, Hv, V, K, float(scale),
            num_warps=1
        )

        # Reshape output to [B,Hv,V] and cast to bfloat16
        out = out_flat.view(B, Hv, V).to(torch.bfloat16)  # [B,Hv,V] in bfloat16

        # Reshape new_state to [B,Hv,V,K]
        new_state = new_state_flat.view(B, Hv, V, K)

        return out, new_state


def run(*args):
    return ModelNew()(*args)
