import torch
import triton
import triton.language as tl


@triton.jit
def compute_g_beta_kernel(a_ptr, dt_bias_ptr, A_log_ptr, b_ptr,
                          g_flat_ptr, beta_flat_ptr,
                          T, V):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v]))
    and beta[t, v] = sigmoid(b[t, v]) for all t in [0, T), v in [0, V),
    store into flat arrays g_flat_ptr and beta_flat_ptr of length T*V.
    softplus(x) = log(1 + exp(x)), sigmoid(x) = 1 / (1 + exp(-x)).
    """
    # One program per token t
    t = tl.program_id(0)
    for v in range(0, V):
        # Load scalars
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        # softplus and g
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        # beta
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        idx = t * V + v
        tl.store(g_flat_ptr + idx, g_val)
        tl.store(beta_flat_ptr + idx, beta_val)


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_flat_ptr, beta_flat_ptr,
                        new_state_ptr,
                        T, H, V, K, t, seq_idx):
    """
    Update state per token t within sequence block seq_idx:
    For each h in [0, H), v in [0, V):
      old_v[h, :] = sum_j k[t, h, j] * state_old[h, v, j]
      new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
      state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
      state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
      state_new[h, v, :] = g[h, v] * state_old[h, v, :] - state_remove[h, :] + state_update[h, :]
    All reductions are done explicitly over K=128 using vectorized ops.
    """
    # Constants
    K_vec = tl.arange(0, K)  # [0..127]
    V_vec = tl.arange(0, V)  # [0..7]
    H_vec = tl.arange(0, H)  # [0..3]

    # Load beta[v] and g[h, v] as scalars
    for h in range(0, H):
        for v_i in range(0, V):
            beta_val = tl.load(beta_flat_ptr + (t * V + v_i)).to(tl.float32)
            g_val = tl.load(g_flat_ptr + (h * V + v_i)).to(tl.float32)
            # Load vectors for q, k, v
            # q[t, h, :] shape is (K,)
            q_vec = tl.load(q_ptr + t * (H * K) + h * K + K_vec).to(tl.float32)
            # k[t, h, :] shape is (K,)
            k_row = tl.load(k_ptr + t * (H * K) + h * K + K_vec).to(tl.float32)
            # v[t, v_i, :] shape is (K,)
            v_vec = tl.load(v_ptr + t * (V * K) + v_i * K + K_vec).to(tl.float32)
            # state_old[h, v_i, :] shape is (K,)
            state_row = tl.load(state_old_ptr + h * (V * K) + v_i * K + K_vec).to(tl.float32)
            # Compute old_v[h, :] = k_row @ state_row
            old_v = tl.sum(k_row[:, None] * state_row[None, :], axis=0)  # (K,)
            # new_v[h, :] = beta * v_vec + (1 - beta) * old_v
            new_v = beta_val * v_vec + (1.0 - beta_val) * old_v
            # Compute state_remove[h, :] = sum_j k_row[j] * old_v[j]
            state_remove = tl.sum(k_row * old_v, axis=0)
            # Compute state_update[h, :] = sum_j k_row[j] * new_v[j]
            state_update = tl.sum(k_row * new_v, axis=0)
            # Update state_new[h, v_i, :] = g * state_row - state_remove + state_update
            state_new_vec = g_val * state_row - state_remove + state_update
            # Store updated state
            tl.store(new_state_ptr + seq_idx * (H * V * K) + h * (V * K) + v_i * K + K_vec,
                     state_new_vec, mask=K_vec < K)


@triton.jit
def compute_output_kernel(q_ptr, new_state_ptr, out_ptr, scale, T, H, V, K):
    """
    Compute output per token t for the last sequence block (seq_idx = T - 1):
    For each t in [0, T):
      For each h in [0, H):
        o[t, h, k] = scale * q[t, h, k] @ new_state[T-1, h, k] = scale * sum_j q[t, h, j] * new_state[T-1, h, j]
    Store out_ptr as float32 [T, H, K].
    """
    t = tl.program_id(0)  # one program per token
    # seq_idx = T - 1
    seq_idx = T - 1
    for h in range(0, H):
        K_vec = tl.arange(0, K)
        q_vec = tl.load(q_ptr + t * (H * K) + h * K + K_vec).to(tl.float32)
        new_vec = tl.load(new_state_ptr + seq_idx * (H * V * K) + h * (V * K) + K_vec).to(tl.float32)
        dot = tl.sum(q_vec * new_vec, axis=0)  # scalar
        out_vec = scale * dot  # scalar result for this (t, h)
        # Store as [T, H, K]; we write the same scalar across K to match output shape, but since
        # original output is [T, H, K] and q @ state_new produces [H, K], we need per-(h, k) output.
        # Instead, we store per (h, k) scalar by broadcasting:
        # Compute a vector where each element equals out_vec
        # However, Triton expects a (K,) vector; we'll broadcast:
        out_vec = tl.full((K,), out_vec, tl.float32)
        tl.store(out_ptr + t * (H * K) + h * K + K_vec, out_vec)


class ModelNew(nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-only forward:
        - compute_g_beta_kernel to get g_flat and beta_flat
        - update_state_kernel to update state per token per sequence block
        - compute_output_kernel to compute output vector per token
        """
        device = q.device
        T = q.shape[0]
        H = q.shape[1]
        K = q.shape[2]
        V = v.shape[1]

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if state is not None:
            state_old = state.contiguous().to(torch.float32)  # [H, V, K]
        else:
            state_old = torch.zeros((H, V, K), dtype=torch.float32, device=device)

        # Prepare outputs
        output = torch.empty((T, H, K), dtype=torch.float32, device=device)
        # Allocate new_state [num_seqs, H, V, K]
        num_seqs = cu_seqlens.numel() - 1
        new_state = torch.empty((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # 1) Compute g_flat and beta_flat (length T*V)
        g_flat = torch.empty((T * V), dtype=torch.float32, device=device)
        beta_flat = torch.empty((T * V), dtype=torch.float32, device=device)

        grid_g = (T,)
        compute_g_beta_kernel[grid_g](
            a, dt_bias, A_log, b, g_flat, beta_flat, T, V
        )

        # 2) Update state for each sequence block
        for seq_idx in range(0, num_seqs):
            # One program per token
            grid_update = (T,)
            update_state_kernel[grid_update](
                q, k, v, state_old, g_flat, beta_flat, new_state, T, H, V, K, t=T, seq_idx=seq_idx
            )

        # 3) Compute output for the last sequence block using Triton
        grid_out = (T,)
        compute_output_kernel[grid_out](
            q, new_state[-1], output, float(scale), T, H, V, K
        )

        # Return output as bfloat16 and new_state as float32
        return output.to(torch.bfloat16), new_state


def run(*args):
    return ModelNew()(*args)
