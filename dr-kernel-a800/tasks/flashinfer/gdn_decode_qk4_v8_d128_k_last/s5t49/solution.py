import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, B: tl.constexpr, H: tl.constexpr):
    """
    Compute g_vec of length B*H:
    g[b*H + h] = exp(-exp(A_log[h]) * softplus(a[b,1,h] + dt_bias[h]))
    a_ptr: shape [B*H] float32
    dt_bias_ptr: shape [H] float32
    A_log_ptr: shape [H] float32
    g_ptr: shape [B*H] float32
    Grid: 1D, size = B*H
    """
    pid = tl.program_id(0)  # index in [0, B*H)
    # softplus(x) = log(1 + exp(x))
    s = tl.softplus(tl.load(a_ptr + pid) + tl.load(dt_bias_ptr + pid % H))
    g_val = tl.exp(-tl.exp(tl.load(A_log_ptr + pid % H)) * s)
    g_ptr[pid] = g_val


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, B: tl.constexpr, H: tl.constexpr):
    """
    Compute beta_vec of length B*H:
    beta[b*H + h] = 1 / (1 + exp(-b[b,1,h]))
    b_ptr: shape [B*H] float32
    beta_ptr: shape [B*H] float32
    Grid: 1D, size = B*H
    """
    pid = tl.program_id(0)
    beta_ptr[pid] = 1.0 / (1.0 + tl.exp(-tl.load(b_ptr + pid)))


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr,
    out_ptr, new_state_ptr,
    B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr, scale
):
    """
    For each (b,h), compute:
    q_h: q[b, h, :] -> [K]
    k_h: k[b, h, :] -> [K]
    v_h: v[b, h, :] -> [V]
    state_old: state[b, h, :, :] -> [V, K] (we index as [K,V] via strides)
    g_val = g[b*H + h], beta_val = beta[b*H + h]
    old_v = k_h @ state_old (reduce over K -> [V])
    new_v = beta_val * v_h + (1 - beta_val) * old_v  -> [V]
    old_state = g_val * state_old  -> [V, K]
    state_remove = k_h @ old_state (reduce over K -> [V])
    state_update = k_h @ new_v (reduce over K -> [V])
    new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None] -> [V, K]
    out[b,h] = scale * (q_h @ new_state[b,h])  -> scalar
    """
    # Precompute linear indices for output and new_state flat
    total = B * H * V * K
    # Grid is 1D over (b,h)
    pid = tl.program_id(0)  # pid in [0, B*H)
    b = pid // H
    h = pid % H

    # Load q_h, k_h, v_h
    # q_ptr is [B*H*K], idx = b*H*K + h*K + k
    q_base = b * H * K + h * K
    k_base = b * H * K + h * K
    v_base = b * H * V + h * V

    # Prepare arrays for matmul (length K)
    q_vec = tl.zeros((K,), dtype=tl.float32)
    k_vec = tl.zeros((K,), dtype=tl.float32)
    # Load state_old as [K, V], then reduce over K to [V]
    # state_ptr layout: [B,H,V,K] contiguous => offset = b*(H*V*K) + h*(V*K) + v*K + k
    # We'll form a [K,V] tile by iterating k then v.
    state_old = tl.zeros((V, K), dtype=tl.float32)

    # Load q_h and k_h
    for k_idx in range(0, K):
        q_vec[k_idx] = tl.load(q_ptr + q_base + k_idx)
        k_vec[k_idx] = tl.load(k_ptr + k_base + k_idx)

    # Load v_h
    v_vec = tl.zeros((V,), dtype=tl.float32)
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            state_old[v_idx, k_idx] = tl.load(state_ptr + b * (H * V * K) + h * (V * K) + v_idx * K + k_idx)
        v_vec[v_idx] = tl.load(v_ptr + v_base + v_idx)

    # Load scalars g_val and beta_val
    g_val = tl.load(g_ptr + pid)
    beta_val = tl.load(beta_ptr + pid)

    # Compute old_v = k_h @ state_old (reduce over K -> [V])
    old_v = tl.zeros((V,), dtype=tl.float32)
    for v_idx in range(0, V):
        dot = 0.0
        for k_idx in range(0, K):
            dot += k_vec[k_idx] * state_old[v_idx, k_idx]
        old_v[v_idx] = dot

    # Compute new_v = beta * v_h + (1 - beta) * old_v
    new_v = tl.zeros((V,), dtype=tl.float32)
    for v_idx in range(0, V):
        new_v[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v[v_idx]

    # Compute old_state = g * state_old -> [V,K]
    old_state = tl.zeros((V, K), dtype=tl.float32)
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            old_state[v_idx, k_idx] = g_val * state_old[v_idx, k_idx]

    # Compute state_remove = k_h @ old_state (reduce over K -> [V])
    state_remove = tl.zeros((V,), dtype=tl.float32)
    for v_idx in range(0, V):
        dot = 0.0
        for k_idx in range(0, K):
            dot += k_vec[k_idx] * old_state[v_idx, k_idx]
        state_remove[v_idx] = dot

    # Compute state_update = k_h @ new_v (reduce over K -> [V])
    state_update = tl.zeros((V,), dtype=tl.float32)
    for v_idx in range(0, V):
        dot = 0.0
        for k_idx in range(0, K):
            dot += k_vec[k_idx] * new_v[v_idx]
        state_update[v_idx] = dot

    # Update new_state: old_state - state_remove[:, None] + state_update[:, None] -> [V,K]
    new_state_out = tl.zeros((V, K), dtype=tl.float32)
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            new_state_out[v_idx, k_idx] = old_state[v_idx, k_idx] - state_remove[v_idx] + state_update[v_idx]

    # Store new_state to flat pointer at offset b*H*V*K + h*V*K + v_idx*K + k_idx
    base_ns = b * H * V * K + h * V * K
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            tl.store(new_state_ptr + base_ns + v_idx * K + k_idx, new_state_out[v_idx, k_idx])

    # Compute output[b,h] = scale * (q_h @ new_state_out)  -> reduce over K
    out_val = 0.0
    for v_idx in range(0, V):
        dot = 0.0
        for k_idx in range(0, K):
            dot += new_state_out[v_idx, k_idx]
        out_val += q_vec[k_idx] * dot
    out_val = scale * out_val

    # Store out[b,h] at out_ptr[b*H + h]
    tl.store(out_ptr + b * H + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Squeeze to expected shapes: q,k [B,4,K], v [B,8,V], state [B,8,V,K]
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3 and state.dim() == 4
        B_q, H_q, K = q.shape
        B_k, H_k, _ = k.shape
        B_v, H_v, V = v.shape
        B_st, H_st, V_st, K_st = state.shape
        assert B_q == B_k == B_v == B_st, "Batch sizes must match"
        assert H_q == 4 and H_k == 4 and H_v == 8 and H_st == 8, "Fixed head counts expected"
        assert K == 128 and V == 128, "Expected K=128, V=128"

        # Cast parameters to float32 and prepare 1D vectors
        a_f32 = a.to(torch.float32)            # [B,1,H]
        dt_bias_f32 = dt_bias.to(torch.float32)  # [H]
        A_log_f32 = A_log.to(torch.float32)      # [H]
        b_f32 = b.to(torch.float32)              # [B,1,H]

        a_vec = a_f32.view(B_q, -1)[:, 0, :].contiguous().view(-1)   # [B*H]
        b_vec = b_f32.view(B_q, -1)[:, 0, :].contiguous().view(-1)   # [B*H]

        # Allocate outputs
        g_vec = torch.empty(B_q * H_v, dtype=torch.float32, device=q.device)
        beta_vec = torch.empty(B_q * H_v, dtype=torch.float32, device=q.device)
        out = torch.empty(B_q * H_v, dtype=torch.float32, device=q.device)
        new_state = torch.empty(B_q * H_v * V * K, dtype=torch.float32, device=q.device)

        # Launch Triton kernels
        # Grid sizes
        grid_g = (B_q * H_v,)
        grid_beta = (B_q * H_v,)
        grid_main = (B_q * H_v,)

        _compute_g_kernel[grid_g](a_vec, dt_bias_f32, A_log_f32, g_vec, B_q, H_v)
        _compute_beta_kernel[grid_beta](b_vec, beta_vec, B_q, H_v)

        # Prepare pointers for q,k,v,state (contiguous)
        q_flat = q.contiguous().view(-1)          # [B*H_q*K]
        k_flat = k.contiguous().view(-1)          # [B*H_k*K]
        v_flat = v.contiguous().view(-1)          # [B*H_v*V]
        state_flat = state.contiguous().view(-1)  # [B*H_st*V*]

# ... (middle omitted) ...


def run(*args):
    return ModelNew()(*args)
