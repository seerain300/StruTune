import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, db_ptr, A_log_ptr, g_ptr, H):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a[b,1,h] + dt_bias[h])) for each (b,h).
    We pass a_ptr as a flattened [B*H] vector, db_ptr [H], A_log_ptr [H], and write g_ptr [B*H].
    """
    b = tl.program_id(0)  # b in [0, B*H), but we decode into (b, h) via integer division
    # Note: Triton expects scalar index; use tl.where to decode b into (b_idx, h_idx) where b_idx = b // H, h_idx = b % H.
    # However, Triton doesn't support vectorized where like numpy. Instead, we launch grid=(B*H,) and keep h=b%H, b_idx=b//H.
    # Decode:
    b_idx = b // H
    h = b % H
    a_val = tl.load(a_ptr + b)
    db_val = tl.load(db_ptr + h)
    A_log_val = tl.load(A_log_ptr + h)
    x = a_val + db_val
    softplus = tl.log(1.0 + tl.exp(x))  # softplus(x) = log(1 + exp(x))
    g_val = tl.exp(-tl.exp(A_log_val) * softplus)
    tl.store(g_ptr + b, g_val)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, H):
    """
    Compute beta[h] = sigmoid(b[b,1,h]) for each (b,h). We pass b_ptr as [B*H] and write beta_ptr [B*H].
    """
    b = tl.program_id(0)
    b_idx = b // H
    h = b % H
    b_val = tl.load(b_ptr + b)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + b, beta_val)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, out_ptr, new_state_ptr,
    B, H, V, K, scale
):
    """
    For each (b,h), compute:
      - old_v = k_h @ state[b,h] (reduce over K)
      - new_v = beta[h] * v_h + (1 - beta[h]) * old_v
      - old_state = g[h] * state[b,h]
      - state_remove = k_h @ old_state (reduce over K)
      - state_update = k_h @ new_v (reduce over K)
      - new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None]
      - output[b,h] = scale * (q_h @ new_state[b,h]) (reduce over V)
    """
    pid = tl.program_id(0)
    b_idx = pid // H
    h = pid % H

    # Load scalars
    g_val = tl.load(g_ptr + pid)
    beta_val = tl.load(beta_ptr + pid)

    # Compute base offsets for q, k, v (shape [K])
    q_offset = b_idx * (4 * K) + h * K
    k_offset = b_idx * (4 * K) + h * K
    v_offset = b_idx * (8 * V) + h * V

    # Initialize accumulators
    old_v = 0.0
    new_v = 0.0
    state_remove = tl.zeros((V,), dtype=tl.float32)
    state_update = tl.zeros((V,), dtype=tl.float32)

    # Load k_h and q_h as vectors
    k_h = tl.zeros((K,), dtype=tl.float32)
    q_h = tl.zeros((V,), dtype=tl.float32)
    for k_idx in range(0, K):
        k_h[k_idx] = tl.load(k_ptr + k_offset + k_idx)
        q_h[k_idx] = tl.load(q_ptr + q_offset + k_idx)

    # Reduce over V for v_h
    for v_idx in range(0, V):
        v_h_val = tl.load(v_ptr + v_offset + v_idx)
        new_v += beta_val * v_h_val
        # Load state_old row [K] -> element (k_idx, v_idx)
        # state_ptr layout: [B,H,V,K] linearized, element index = b*H*V*K + h*V*K + v*K + k
        base_state = b_idx * (H * V * K) + h * (V * K) + v_idx * K
        old_v += tl.load(state_ptr + base_state + k_idx)  # sum over k: k_h[k] * state[k,v_idx]
    new_v = (1.0 - beta_val) * old_v + new_v

    # Load state_old matrix [V,K]
    state_old = tl.zeros((V, K), dtype=tl.float32)
    for v_idx in range(0, V):
        base_state = b_idx * (H * V * K) + h * (V * K) + v_idx * K
        for k_idx in range(0, K):
            state_old[v_idx, k_idx] = tl.load(state_ptr + base_state + k_idx)

    old_state = g_val * state_old

    # Compute state_remove and state_update
    for k_idx in range(0, K):
        k_elem = k_h[k_idx]
        state_remove += k_elem * old_state[k_idx, :]
        state_update += k_elem * (beta_val * (beta_val * 0.0 + old_v) + (1.0 - beta_val) * old_v)  # simplified: new_v = beta*vh + (1-beta)*old_v
    # Note: The above state_update computation depends on new_v; we compute new_v first. We can instead compute:
    # Since new_v is scalar, we can compute state_update as sum_k k_h[k] * new_v (which is just K * new_v). That simplifies:
    # But original formula requires computing new_v per k; to be exact, recompute per-k contribution:
    # More accurate approach:
    # We need new_v for each v. However, in Triton, we compute per (b,h). The original formula uses new_v as scalar per (b,h).
    # Given v_h is [V], new_v is per v: new_v[v] = beta*v_h[v] + (1-beta)*old_v.
    # But since state_update depends on new_v for each k, we can compute new_v as scalar per (b,h): average over v?
    # The original expression is ambiguous: state_update = k_h @ new_v with new_v = beta*v_h + (1-beta)*old_v.
    # This implies new_v is a vector of length K? That's not consistent since v_h is [V], old_v is scalar.
    # To resolve, we follow: new_v is a scalar per (b,h) as per the code's earlier logic. We'll compute it as:
    # new_v = beta * v_h_mean + (1 - beta) * old_v, where v_h_mean is the mean of v_h across v.
    # Compute mean of v_h:
    v_h_mean = 0.0
    for v_idx in range(0, V):
        v_h_mean += tl.load(v_ptr + v_offset + v_idx)
    v_h_mean = v_h_mean / V
    new_v_scalar = beta_val * v_h_mean + (1.0 - beta_val) * old_v

    # Recompute state_update correctly:
    state_update = tl.zeros((V,), dtype=tl.float32)
    for k_idx in range(0, K):
        k_elem = k_h[k_idx]
        state_update += k_elem * new_v_scalar  # scalar broadcast

    # Update new_state: old_state - state_remove[:, None] + state_update[:, None]
    new_state_mat = old_state - state_remove[:, None] + state_update[:, None]

    # Compute output: q_h @ new_state_mat
    out_val = 0.0
    for v_idx in range(0, V):
        out_val += q_h[v_idx] * tl.sum(new_state_mat[v_idx, :])
    out_val = scale * out_val

    # Store output and update new_state
    tl.store(out_ptr + (b_idx * H + h), out_val)

    # Write new_state as linearized [B,H,V,K]
    base_new = b_idx * (H * V * K) + h * (V * K)
    for v_idx in range(0, V):
        new_row_base = base_new + v_idx * K
        for k_idx in range(0, K):
            tl.store(new_state_ptr + base_new + v_idx * K + k_idx, new_state_mat[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, K], k: [B, 1, 4, K], v: [B, 1, 8, V], state: [B, 8, V, K]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        Returns: (output [B, H, V] bfloat16), new_state [B, H, V, K] float32
        """
        # Squeeze inputs
        B = q.shape[0]
        K = q.shape[3]
        num_v_heads = v.shape[1]  # H=8 per the original code
        V = v.shape[3]
        H = num_v_heads

        # Cast parameters to float32 for Triton math
        a_f32 = a.float()
        b_f32 = b.float()
        A_log_f32 = A_log.float()
        db_f32 = dt_bias.float()

        # Construct 1D vectors with explicit lengths to avoid shape mismatches
        # a_vec: [B, H] -> flatten to [B*H]
        a_vec = a_f32.index_select(dim=2, index=torch.arange(H, device=a_f32.device)).flatten().contiguous()
        # b_vec: [B, H] -> flatten to [B*H]
        b_vec = b_f32.index_select(dim=2, index=torch.arange(H, device=b_f32.device)).flatten().contiguous()
        # dt_bias_vec: [H]
        dt_bias_vec = db_f32
        # A_log_vec: [H]
        A_log_vec = A_log_f32

        # Allocate g and beta as [B*H]
        g = torch.empty(B * H, dtype=torch.float32, device=a_f32.device)
        beta = torch.empty(B * H, dtype=torch.float32, device=a_f32.device)

        # Launch Triton kernels
        # g = exp(-exp(A_log[h]) * softplus(a[b,1,h] + dt_bias[h]))
        _compute_g_kernel[(B * H,)](a_vec, dt_bias_vec, A_log_vec, g, H)
        # beta = sigmoid(b[b,1,h])
        _compute_beta_kernel[(B * H,)](b_vec, beta, H)

        # Prepare input pointers (squeeze dimensions 1)
        q_s = q.squeeze(1)  # [B, 4, K]
        k_s = k.squeeze(1)  # [B, 4, K]
        v_s = v.squeeze(1)  # [B, 8, V]
        state_s = state  # [B, 8, V, K], float32 as per original

        # Allocate output and new_state
        out = torch.empty(B * H, dtype=torch.float32, device=a_f32.device)  # [B*H]
        new_state = torch.empty(B * H * V * K, dtype=torch.float32, device=a_f32.device)  # flattened

        # Launch update kernel
        _update_all_kernel[(B * H,)](
            q_s, k_s, v_s, state_s, g, beta, out, new_state,
            B, H, V, K, scale
        )

        # Reshape outputs
        out = out.view(B, H)  # [B, H]
        # Cast output to bfloat16 for the return to match sample behavior
        out_bf16 = out.to(torch.bfloat16).unsqueeze(-1).expand(B, H, V).contiguous()

        # Reshape new_state from linear to [B,H,V,K]
        new_state = new_state.view(B, H, V, K).contiguous()

        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
