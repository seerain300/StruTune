import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, db_ptr, A_log_ptr, g_ptr, B, H):
    """
    Compute g[b, h] = exp(-exp(A_log[h]) * softplus(a[b, 1, h] + dt_bias[h]))
    a_ptr: [B*H] float32
    db_ptr: [H] float32
    A_log_ptr: [H] float32
    g_ptr: [B*H] float32
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    # Base offset
    # a_ptr[pid] = a[b, 1, h]
    a_val = tl.load(a_ptr + pid)
    db_val = tl.load(db_ptr + h)
    A_log_val = tl.load(A_log_ptr + h)
    x = a_val + db_val
    softplus = tl.log(1.0 + tl.exp(x))
    g = tl.exp(-tl.exp(A_log_val) * softplus)
    tl.store(g_ptr + pid, g)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, B, H):
    """
    Compute beta[b, h] = 1 / (1 + exp(-b[b, 1, h]))
    b_ptr: [B*H] float32
    beta_ptr: [B*H] float32
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + pid, beta)


@triton.jit
def _update_all_kernel(q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr,
                        out_ptr, new_state_ptr,
                        B, H, V, K, scale):
    """
    For each (b,h):
      q_h: [K], k_h: [K], v_h: [V], state_old: [V,K] -> state_new: [V,K], output: scalar
    q_ptr: [B*H*K] float32
    k_ptr: [B*H*K] float32
    v_ptr: [B*H*V] float32
    state_ptr: [B*H*V*K] float32
    g_ptr: [B*H] float32
    beta_ptr: [B*H] float32
    out_ptr: [B*H] float32
    new_state_ptr: [B*H*V*K] float32
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Base offsets
    base_q = pid * K
    base_k = base_q  # same b,h
    base_v = pid * V
    base_state = (b * H + h) * V * K

    # Load q_h, k_h
    q_h = tl.zeros([K], dtype=tl.float32)
    k_h = tl.zeros([K], dtype=tl.float32)
    for k_idx in range(K):
        q_h[k_idx] = tl.load(q_ptr + base_q + k_idx)
        k_h[k_idx] = tl.load(k_ptr + base_k + k_idx)

    # Load g and beta
    g_val = tl.load(g_ptr + pid)
    beta_val = tl.load(beta_ptr + pid)

    # Load v_h
    v_h = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(V):
        v_h[v_idx] = tl.load(v_ptr + base_v + v_idx)

    # Load state_old: [V, K]
    state_old = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(V):
        for k_idx in range(K):
            state_old[v_idx, k_idx] = tl.load(state_ptr + base_state + v_idx * K + k_idx)

    # Compute old_v = k_h @ state_old (reduce over K)
    old_v = 0.0
    for k_idx in range(K):
        # sum over V of state_old[:, k_idx] * k_h[k_idx]
        row_sum = 0.0
        for v_idx in range(V):
            row_sum += state_old[v_idx, k_idx]
        old_v += row_sum * k_h[k_idx]

    # new_v = beta * v_h + (1 - beta) * old_v
    new_v = beta_val * v_h + (1.0 - beta_val) * old_v

    # old_state = g * state_old
    old_state_mat = g_val * state_old

    # state_remove = k_h @ old_state (reduce over K)
    state_remove = 0.0
    for k_idx in range(K):
        # sum over V of old_state[:, k_idx] * k_h[k_idx]
        row_sum = 0.0
        for v_idx in range(V):
            row_sum += old_state_mat[v_idx, k_idx]
        state_remove += row_sum * k_h[k_idx]

    # state_update = k_h @ new_v (reduce over K)
    state_update = 0.0
    for k_idx in range(K):
        state_update += new_v[k_idx] * k_h[k_idx]

    # Compute new_state per (V, K): old_state - state_remove[:, None] + state_update[:, None]
    for v_idx in range(V):
        # update column-wise for all K
        for k_idx in range(K):
            # old_state_mat[v_idx, k_idx] - state_remove + state_update
            new_val = old_state_mat[v_idx, k_idx] - state_remove + state_update
            tl.store(new_state_ptr + base_state + v_idx * K + k_idx, new_val)

    # output[b,h] = scale * (q_h @ new_state) where new_state is [V,K], q_h is [K]
    out_sum = 0.0
    for k_idx in range(K):
        col_sum = 0.0
        for v_idx in range(V):
            col_sum += new_state_ptr[base_state + v_idx * K + k_idx]
        out_sum += q_h[k_idx] * col_sum
    out_val = scale * out_sum
    tl.store(out_ptr + pid, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, K], k: [B, 1, 4, K], v: [B, 1, 8, V], state: [B, 8, V, K]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        Returns: output [B, H, V] (cast to bfloat16), new_state [B, H, V, K] (float32)
        """
        # Ensure device and dtype
        device = q.device
        B = q.size(0)
        H = v.size(1)  # num_v_heads
        V = v.size(-1)
        K = q.size(-1)

        # Cast parameters to float32 for Triton kernels
        a_f32 = a.float()
        b_f32 = b.float()
        A_log_f32 = A_log.float()
        db_f32 = dt_bias.float()

        # Build a_vec: [B, H] then flatten to [B*H]
        a_idx = torch.arange(H, device=device)
        a_2d = a_f32.index_select(dim=2, index=a_idx)  # [B, H]
        a_vec = a_2d.reshape(B * H).contiguous()  # [B*H]

        # Build b_vec: [B, H] then flatten to [B*H]
        b_idx = torch.arange(H, device=device)
        b_2d = b_f32.index_select(dim=2, index=b_idx)  # [B, H]
        b_vec = b_2d.reshape(B * H).contiguous()  # [B*H]

        # dt_bias_vec: [H]
        dt_bias_vec = db_f32  # already [H]

        # A_log_vec: [H]
        A_log_vec = A_log_f32  # already [H]

        # Prepare outputs and new_state
        out = torch.empty(B * H, dtype=torch.float32, device=device)
        g = torch.empty(B * H, dtype=torch.float32, device=device)
        beta = torch.empty(B * H, dtype=torch.float32, device=device)
        new_state = torch.empty((B * H * V * K), dtype=torch.float32, device=device)

        # Launch Triton kernels
        # 1) Compute g
        grid_g = (B * H,)
        _compute_g_kernel[grid_g](a_vec, dt_bias_vec, A_log_vec, g, B, H)

        # 2) Compute beta
        _compute_beta_kernel[grid_g](b_vec, beta, B, H)

        # 3) Update state and compute output per (b,h)
        grid_u = (B * H,)
        _update_all_kernel[grid_u](
            q.reshape(B * H * K).contiguous(),          # q_ptr: [B*H*K]
            k.reshape(B * H * K).contiguous(),          # k_ptr: [B*H*K]
            v.reshape(B * H * V).contiguous(),          # v_ptr: [B*H*V]
            state.reshape(B * H * V * K).contiguous(),  # state_ptr: [B*H*V*K]
            g, beta, out, new_state,
            B, H, V, K, float(scale)
        )

        # Reshape outputs
        out = out.view(B, H)
        # Cast output to bfloat16 [B, H, V]
        out_broadcast = out.unsqueeze(-1).expand(B, H, V).to(torch.bfloat16).contiguous()
        new_state = new_state.view(B, H, V, K).contiguous()

        return out_broadcast, new_state


def run(*args):
    return ModelNew()(*args)
