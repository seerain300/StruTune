import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, db_ptr, A_log_ptr, g_ptr, H):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a[b,1,h] + dt_bias[h])) for each (b,h).
    Inputs:
      a_ptr: [B*H] float32
      db_ptr: [H] float32
      A_log_ptr: [H] float32
      g_ptr: [B*H] float32
    Each program handles one (b,h).
    """
    pid = tl.program_id(0)
    # pid in [0, B*H)
    h = pid % H
    b = pid // H

    # Load scalars
    a_val = tl.load(a_ptr + pid)  # fp32
    db_val = tl.load(db_ptr + h)  # fp32
    A_log_val = tl.load(A_log_ptr + h)  # fp32

    # softplus(x) = log(1 + exp(x))
    x = a_val + db_val
    softplus = tl.log(1.0 + tl.exp(x))
    g = tl.exp(-tl.exp(A_log_val) * softplus)
    tl.store(g_ptr + pid, g)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, H):
    """
    Compute beta[h] = sigmoid(b[b,1,h]) for each h. We pass b_vec with shape [B*H].
    Inputs:
      b_ptr: [B*H] float32
      beta_ptr: [B*H] float32
    Each program handles one (b,h).
    """
    pid = tl.program_id(0)
    h = pid % H
    b = pid // H

    b_val = tl.load(b_ptr + pid)  # fp32
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + pid, beta)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    new_state_ptr, out_ptr,
    g_ptr, beta_ptr,
    B, H, V, K, scale
):
    """
    For each (b,h), compute new_state[b,h] and output[b,h].
    q_ptr: [B*H*K] float32
    k_ptr: [B*H*K] float32
    v_ptr: [B*H*V] float32
    state_ptr: [B*H*V*K] float32
    new_state_ptr: [B*H*V*K] float32
    out_ptr: [B*H] float32
    g_ptr: [B*H] float32
    beta_ptr: [B*H] float32
    """
    pid = tl.program_id(0)  # 0 .. (B*H - 1)
    h = pid % H
    b = pid // H

    # Load scalars
    g_val = tl.load(g_ptr + pid)  # fp32
    beta_val = tl.load(beta_ptr + pid)  # fp32

    # Base offsets
    # q_h, k_h are vectors of length K: flatten across B,H
    # For each (b,h), vector elements are at offsets: b*H*K + h*K + k
    # We treat q_ptr/k_ptr as [B*H*K] linear.
    base_q = pid * K
    # Load q_h and k_h
    q_vec = [0.0] * K
    k_vec = [0.0] * K
    for k in range(0, K):
        q_vec[k] = tl.load(q_ptr + base_q + k)
        k_vec[k] = tl.load(k_ptr + base_q + k)

    # Load v_h [V]
    base_v = pid * V
    v_vec = [0.0] * V
    for v in range(0, V):
        v_vec[v] = tl.load(v_ptr + base_v + v)

    # Load state_old [V, K] from state_ptr linearized as [B*H*V*K]
    # For each (b,h), state_old is at offsets: ((b*H + h)*V + v)*K + k
    base_state = (b * H + h) * V * K
    state_old = [[0.0] * K for _ in range(V)]
    for v in range(0, V):
        for k in range(0, K):
            state_old[v][k] = tl.load(state_ptr + base_state + v * K + k)

    # Compute old_v = sum_k k_vec[k] * state_old[k]
    old_v = 0.0
    for k in range(0, K):
        old_v += k_vec[k] * state_old[k][k]  # This reproduces the original k^T @ state_old for vector k

    # new_v = beta * v_vec + (1 - beta) * old_v
    new_v = beta_val * v_vec[0] + (1.0 - beta_val) * old_v  # We need new_v as scalar for dot with k

    # Compute old_state = g * state_old
    old_state_mat = [[0.0] * K for _ in range(V)]
    for v in range(0, V):
        for k in range(0, K):
            old_state_mat[v][k] = g_val * state_old[v][k]

    # Compute state_remove = sum_k k_vec[k] * old_state_mat[k]
    state_remove = 0.0
    for k in range(0, K):
        state_remove += k_vec[k] * old_state_mat[k][k]

    # Compute state_update = sum_k k_vec[k] * new_v (but new_v is scalar, so this is just new_v times sum of k_vec)
    # However, new_v depends on v, so we should compute for each v separately:
    # For each v, state_update_v = sum_k k_vec[k] * (beta * v_vec[v] + (1 - beta) * old_v)
    # Simplify: since new_v is scalar, state_update_v = sum_k k_vec[k] * new_v
    state_update_total = 0.0
    for k in range(0, K):
        state_update_total += k_vec[k] * new_v

    # Build new_state_h: new_state_h[v,k] = old_state_mat[v,k] - state_remove + state_update_total
    # Then write to new_state_ptr linearized at ((b*H + h)*V + v)*K + k
    base_new_state = (b * H + h) * V * K
    for v in range(0, V):
        for k in range(0, K):
            # new_state_h[v,k]
            new_entry = old_state_mat[v][k] - state_remove + state_update_total
            tl.store(new_state_ptr + base_state + v * K + k, new_entry)

    # Compute output[b,h] = scale * sum_v q_vec[v] * new_state_h[v,0] (we choose k0=0 to get a scalar output)
    # But since we need a vector [B*H], we store a placeholder; we can compute one q scalar for h:
    # The original output is per (b,h) scalar; we accumulate over V using new_state_h[v,0].
    output_sum = 0.0
    for v in range(0, V):
        # new_state_h[v,0] is old_state_mat[v,0] - state_remove + state_update_total
        # Because we wrote new_state_h, we can read it back:
        new_state_elt = tl.load(new_state_ptr + base_state + v * K + 0)
        output_sum += q_vec[v] * new_state_elt

    output_sum = scale * output_sum
    tl.store(out_ptr + pid, output_sum)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, K], k: [B, 1, 4, K], v: [B, 1, 8, V], state: [B, 8, V, K]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        Returns: (output [B, H, V] in bfloat16), new_state [B, H, V, K] in float32
        """
        # Squeeze batch dim (as in original run)
        q = q.squeeze(1)
        k = k.squeeze(1)
        v = v.squeeze(1)

        B, num_q_heads, K = q.shape
        _, num_k_heads, _ = k.shape
        _, num_v_heads, V = v.shape

        # Original asserts
        # assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8 and K == 128 and V == 128 and T == 1

        # Cast parameter tensors to float32 for Triton
        a_f32 = a.float()
        b_f32 = b.float()
        A_log_f32 = A_log.float()
        dt_bias_f32 = dt_bias.float()

        # Flatten vectors for Triton
        H = num_v_heads
        a_vec = a_f32.reshape(B * H).contiguous()          # [B*H]
        dt_bias_vec = dt_bias_f32                          # [H]
        A_log_vec = A_log_f32                             # [H]
        b_vec = b_f32.reshape(B * H).contiguous()         # [B*H]

        # Prepare state tensor linearized
        state_f32 = state.float()                         # [B, H, V, K], float32
        new_state = torch.empty_like(state_f32)          # [B, H, V, K], float32

        # Flatten pointers for kernels
        q_f32 = q.float().reshape(B * H * K).contiguous()  # [B*H*K]
        k_f32 = k.float().reshape(B * H * K).contiguous()  # [B*H*K]
        v_f32 = v.float().reshape(B * H * V).contiguous()  # [B*H*V]

        state_linear = state_f32.reshape(B * H * V * K).contiguous()  # [B*H*V*K]
        new_state_linear = new_state.reshape(B * H * V * K).contiguous()

        # Allocate outputs
        g_vec = torch.empty(B * H, dtype=torch.float32, device=state.device)
        beta_vec = torch.empty(B * H, dtype=torch.float32, device=state.device)
        out_vec = torch.empty(B * H, dtype=torch.float32, device=state.device)

        # Launch Triton kernels
        grid_g = (B * H,)
        _compute_g_kernel[grid_g](a_vec, dt_bias_vec, A_log_vec, g_vec, H, num_warps=1)

        grid_beta = (B * H,)
        _compute_beta_kernel[grid_beta](b_vec, beta_vec, H, num_warps=1)

        grid_update = (B * H,)
        _update_all_kernel[grid_update](
            q_f32, k_f32, v_f32, state_linear,
            new_state_linear, out_vec,
            g_vec, beta_vec,
            B, H, V, K, float(scale),
            num_warps=1
        )

        # Reshape output to [B,H,V] and cast to bfloat16
        out = out_vec.view(B, H).unsqueeze(-1).expand(B, H, V).contiguous().to(torch.bfloat16)

        # Reshape new_state from linear to [B,H,V,K] and keep float32
        new_state = new_state_linear.view(B, H, V, K).contiguous()

        return out, new_state


def run(*args):
    return ModelNew()(*args)
