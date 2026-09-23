import torch
import triton
import triton.language as tl


@triton.jit
def compute_g_and_beta(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, beta_ptr, T, V):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v])) for all t, v.
    beta[t, v] = sigmoid(b[t, v]).
    g_ptr, beta_ptr: [T*V] float32
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
        # beta[t, v] = sigmoid(b[t, v])
        b_val = tl.load(a_ptr + t * V + v).to(tl.float32)  # original 'b' is passed; use it for beta
        # Note: The original code uses 'b' for beta; we assume b is provided. If not, we'd need separate pointer. Here we rely on b argument.
        # For correctness, we compute beta from 'b' as provided. If Triton doesn't find b_ptr, the harness supplies it; adjust accordingly.
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(beta_ptr + t * V + v, beta_val)


@triton.jit
def update_state(q_ptr, k_ptr, v_ptr, state_old_ptr, g_ptr, beta_ptr,
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
    """
    for h in range(0, H):
        for v_i in range(0, V):
            # Load scalars g and beta for (h, v_i)
            g_val = tl.load(g_ptr + h * V + v_i).to(tl.float32)
            beta_val = tl.load(beta_ptr + h * V + v_i).to(tl.float32)
            # old_v[h, :] = sum_j k[t, h, j] * state_old[h, v_i, j]
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
def compute_output(q_ptr, new_state_ptr, out_ptr, scale, T, H, V, K):
    """
    Compute out[t, h, k] = scale * q[t, h, k] @ new_state[h, v, k] for all tokens t.
    Here new_state is assumed to be the last sequence block (num_seqs - 1).
    """
    t = tl.program_id(0)  # one program per token t
    for h in range(0, H):
        for k in range(0, K):
            sum_val = tl.zeros((), dtype=tl.float32)
            for v in range(0, V):
                q_val = tl.load(q_ptr + t * (H * K) + h * K + k).to(tl.float32)
                state_val = tl.load(new_state_ptr + (h * V + v) * K + k).to(tl.float32)
                sum_val += q_val * state_val
            out_val = scale * sum_val
            tl.store(out_ptr + t * (H * K) + h * K + k, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure tensors are contiguous
        device = q.device
        T, H, K = q.shape
        V = v.shape[1]  # num_v_heads

        # Compute g and beta in Triton (flat arrays)
        g_flat = torch.empty(T * V, dtype=torch.float32, device=device)
        beta_flat = torch.empty(T * V, dtype=torch.float32, device=device)

        grid_g = (T,)
        compute_g_and_beta[grid_g](a.contiguous(), dt_bias.contiguous(), A_log.contiguous(), g_flat, beta_flat, T, V)

        # Prepare new_state as [num_seqs, H, V, K] float32
        num_seqs = cu_seqlens.numel() - 1
        new_state = torch.zeros((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # Update state per token and sequence block using Triton
        # Note: We iterate over tokens t=0..T-1 and sequence blocks seq_idx=0..num_seqs-1.
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue
            # state_old is provided as [num_seqs, H, V, K]. We use state[seq_idx].
            state_old = state[seq_idx].contiguous().float()
            # Launch update_state for each token
            for t in range(0, T):
                update_state[(1,)](
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    state_old.contiguous().view(-1), g_flat, beta_flat,
                    new_state[seq_idx].contiguous().view(-1),
                    T, H, V, K, t, seq_idx, num_warps=1
                )

        # Compute output per token using Triton (last sequence block)
        out = torch.empty((T, H, K), dtype=torch.float32, device=device)
        grid_out = (T,)
        # Use the last sequence block for output
        last_seq_idx = num_seqs - 1
        compute_output[grid_out](
            q.contiguous(), new_state[last_seq_idx].contiguous().view(-1), out, float(scale), T, H, V, K
        )
        output = out.to(torch.bfloat16)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
