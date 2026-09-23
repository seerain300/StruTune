import torch
import triton
import triton.language as tl


@triton.jit
def compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, T, V):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v])) for all t, v.
    g_ptr: [T*V] float32
    """
    t = tl.program_id(0)  # one program per token t
    for v in range(0, V):
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        tl.store(g_ptr + t * V + v, g_val)


@triton.jit
def compute_beta_kernel(b_ptr, beta_ptr, T, V):
    """
    Compute beta[t, v] = sigmoid(b[t, v]) for all t, v.
    beta_ptr: [T*V] float32
    """
    t = tl.program_id(0)  # one program per token t
    for v in range(0, V):
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(beta_ptr + t * V + v, beta_val)


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_ptr, beta_ptr,
                        new_state_ptr, T, H, V, K, t, seq_idx):
    """
    Update state per token t within sequence block seq_idx:
    For each h in [0, H), v in [0, V):
      old_v[h, :] = sum_j k[t, h, j] * state_old[h, v, j]
      new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
      state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
      state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
      state_new[h, v, :] = g[h, v] * state_old[h, v, :] - state_remove[h, :] + state_update[h, :]
    """
    # One program per (t, seq_idx) processes all h, v via loops
    for h in range(0, H):
        for v_i in range(0, V):
            # g_val and beta_val are scalars for (h, v_i)
            g_val = tl.load(g_ptr + h * V + v_i).to(tl.float32)
            beta_val = tl.load(beta_ptr + h * V + v_i).to(tl.float32)
            # Compute old_v[h, :] = sum_j k[t, h, j] * state_old[h, v_i, j]
            old_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_k = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)  # k[t, h, j]
                state_old_kv = tl.load(state_old_ptr + (h * V + v_i) * K + j).to(tl.float32)  # state_old[h, v_i, j]
                old_v[j] = k_k * state_old_kv
            # new_v[h, :] = beta[v_i] * v[t, v_i, :] + (1 - beta[v_i]) * old_v
            new_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                v_k = tl.load(v_ptr + t * (V * K) + v_i * K + j).to(tl.float32)  # v[t, v_i, j]
                new_v[j] = beta_val * v_k + (1.0 - beta_val) * old_v[j]
            # state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
            state_remove = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_k = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)
                state_remove[j] = k_k * old_v[j]
            # state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
            state_update = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_k = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)
                state_update[j] = k_k * new_v[j]
            # new_state[h, v_i, :] = g * state_old - state_remove + state_update
            state_old_row = tl.load(state_old_ptr + (h * V + v_i) * K + tl.arange(0, K)).to(tl.float32)
            state_new = g_val * state_old_row - state_remove + state_update
            # Store to new_state[h, v_i, :]
            for j in range(0, K):
                tl.store(new_state_ptr + (h * V + v_i) * K + j, state_new[j])


@triton.jit
def compute_output_row(q_ptr, new_state_ptr, out_ptr, scale, T, H, V, K, t):
    """
    Compute out[t] = scale * q[t] @ new_state[seq_idx] for all t.
    out_ptr: [T*H*K] float32, stored as out[t, h, k] at index t * (H*K) + h*K + k
    """
    for h in range(0, H):
        q_row = tl.load(q_ptr + t * (H * K) + h * K + tl.arange(0, K)).to(tl.float32)  # [K]
        state_row = tl.load(new_state_ptr + h * (V * K) + tl.arange(0, K)).to(tl.float32)  # [K]
        dot = tl.sum(q_row * state_row, axis=0)
        out_index = t * (H * K) + h * K
        tl.store(out_ptr + out_index, dot * scale)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        device = q.device
        T, H, K = q.shape
        V = v.shape[1]
        num_seqs = cu_seqlens.shape[0] - 1

        # Prepare g and beta
        g_flat = torch.empty(T * V, dtype=torch.float32, device=device)
        beta_flat = torch.empty(T * V, dtype=torch.float32, device=device)

        grid_g = (T,)
        compute_g_kernel[grid_g](a, dt_bias, A_log, g_flat, T, V, num_warps=1)
        grid_beta = (T,)
        compute_beta_kernel[b](beta_flat, T, V, num_warps=1)

        # Initialize new_state
        new_state = torch.zeros((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # Update state for each token in each sequence block
        # We process tokens in the order of cu_seqlens, i.e., per seq_idx, tokens t in [start, end)
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            for t in range(seq_start, seq_end):
                # Run Triton kernel to update state for this token in this sequence block
                grid_upd = (1,)
                update_state_kernel[grid_upd](q, k, v, state[seq_idx].contiguous(), g_flat, beta_flat,
                                              new_state[seq_idx], T, H, V, K, t, seq_idx, num_warps=1)

        # Compute output per token using Triton row-wise matmul
        out = torch.empty((T, H, K), dtype=torch.float32, device=device)
        for t in range(T):
            grid_out = (1,)
            compute_output_row[grid_out](q[t], new_state[-1], out[t], float(scale), T, H, V, K, t, num_warps=1)

        # Return output as bfloat16 (to match original), and new_state as float32
        return out.to(torch.bfloat16), new_state


def run(*args):
    return ModelNew()(*args)
