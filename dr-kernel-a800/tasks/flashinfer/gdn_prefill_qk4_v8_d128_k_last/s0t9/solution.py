import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_g_beta_kernel(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    H, V, T,
):
    """
    Triton kernel to compute:
      g[i] = exp(-exp(A_log[i // V]) * softplus(a[t, i % V] + dt_bias[i % V])) for i in [0, H*V)
      beta[j] = sigmoid(b[t, j]) for j in [0, V)
    Writes g to g_ptr[H*V] and beta to beta_ptr[V], both float32.
    """
    # We use a single program instance and loop over H*V for g and over V for beta.
    for i in range(0, H * V):
        v_idx = i % V
        val_a = tl.load(a_ptr + i).to(tl.float32)
        val_dt = tl.load(dt_bias_ptr + v_idx).to(tl.float32)
        softplus = tl.log(1.0 + tl.exp(val_a + val_dt))  # softplus(x)
        g_val = tl.exp(-tl.exp(tl.load(A_log_ptr + v_idx).to(tl.float32)) * softplus)
        tl.store(g_ptr + i, g_val)
    for j in range(0, V):
        val_b = tl.load(b_ptr + j).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-val_b))
        tl.store(beta_ptr + j, beta_val)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-featured forward:
        - Computes g and beta via Triton.
        - Updates state using PyTorch matmul/dot (for stability and correctness).
        - Computes output per token as scale * q @ state_new.
        Shapes (assumed from original):
          q: [T, H, K] -> H=4, K=128
          k: [T, H, K]
          v: [T, V, K] -> V=8
          state: [num_seqs, V, K, K] (the provided code uses [H,V,K], but the tensors align by indexing; we use [V,K,K]).
          A_log: [V]
          a: [T, V]
          dt_bias: [V]
          b: [T, V]
          cu_seqlens: [num_seqs+1]
          scale: float
        Output:
          output: [T, H, K] (bfloat16), new_state: [num_seqs, H, V, K] (float32)
        """
        device = q.device
        dtype_qk = q.dtype
        T = q.shape[0]
        H = q.shape[1]
        K = q.shape[2]
        V = v.shape[1]
        num_seqs = cu_seqlens.numel() - 1

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        A_log = A_log.contiguous()
        a = a.contiguous()
        dt_bias = dt_bias.contiguous()
        b = b.contiguous()

        # Allocate outputs
        output = torch.empty((T, H, K), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # 1) Compute g and beta with Triton
        g_flat = torch.empty((H * V,), dtype=torch.float32, device=device)
        beta_flat = torch.empty((V,), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        compute_g_beta_kernel[(1,)](
            A_log, a, dt_bias, b,
            g_flat, beta_flat,
            H, V, T,
        )

        # 2) Initialize new_state from state if provided
        # Note: original code handles state=None; here we assume state is provided as in get_inputs().
        # But given state shape [num_seqs, V, K, K], we need to match [H, V, K, K]. The original code uses [H,V,K].
        # Here, we interpret state as [H, V, K, K] as in the benchmark inputs. So we initialize new_state from state.
        # If state is None, initialize zeros.
        if state is None:
            # For correctness, initialize new_state zeros; the per-token loop below will compute new_state.
            pass
        else:
            # state is [num_seqs, V, K, K]; we need to build [H, V, K, K] per block. Since H=4 from q, and V=8, we
            # can replicate state across H dimension by assumption (H==4 in provided get_inputs). In general,
            # we rely on the per-token update which doesn't need initial new_state; we can skip this.
            pass

        # 3) For each sequence block and each token t, compute state_new and output
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # We compute per-token, reusing g and beta computed by Triton
            for t in range(seq_start, seq_end):
                # For each (h, v), compute:
                #   old_v = k[t, h, :] @ state[seq_idx, h, v, :]
                #   new_v = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v
                #   state_remove = k[t, h, :] @ old_v
                #   state_update = k[t, h, :] @ new_v
                #   state_new = g[h, v] * state[seq_idx, h, v, :] - state_remove + state_update
                for h in range(H):
                    for v_i in range(V):
                        # Load k_vec[t, h, :]
                        k_vec = k[t, h, :].float()  # [K]
                        # Load state_old[seq_idx, h, v_i, :]
                        # state is [num_seqs, V, K, K] in get_inputs. We access it as:
                        state_old = state[seq_idx, v_i, :, :]  # [K, K]
                        # old_v = k_vec @ state_old  -> [K]
                        old_v = torch.matmul(k_vec.view(1, K), state_old)  # [1, K]
                        old_v = old_v.squeeze(0)  # [K]

                        # v_vec[t, v_i, :]
                        v_vec = v[t, v_i, :].float()  # [K]

                        # g_val and beta_val
                        g_idx = h * V + v_i
                        g_val = g_flat[g_idx]
                        beta_val = beta_flat[v_i]

                        # new_v = beta * v + (1 - beta) * old_v
                        new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

                        # state_remove = k_vec @ old_v
                        state_remove = torch.matmul(k_vec.view(1, K), old_v.view(K, 1)).squeeze(1)  # scalar? wait, old_v is [K], but k_vec @ old_v is [K]
                        # Correct: state_remove is [K], computed as inner product: sum_j k_vec[j] * old_v[j]
                        # Implement explicitly to avoid ambiguity
                        state_remove = torch.zeros(K, dtype=torch.float32, device=device)
                        for j in range(K):
                            dot_j = 0.0
                            for kk in range(K):
                                dot_j += float(k_vec[kk].item()) * float(old_v[j].item())
                            state_remove[j] = dot_j

                        # state_update = k_vec @ new_v
                        state_update = torch.zeros(K, dtype=torch.float32, device=device)
                        for j in range(K):
                            dot_j = 0.0
                            for kk in range(K):
                                dot_j += float(k_vec[kk].item()) * float(new_v[kk].item())
                            state_update[j] = dot_j

                        # state_new_vec = g * state_old_flat - state_remove + state_update
                        state_old_flat = state_old.view(K)  # [K]
                        state_new_vec = g_val * state_old_flat - state_remove + state_update  # [K]

                        # Store into new_state[seq_idx, h, v_i, :]
                        new_state[seq_idx, h, v_i, :] = state_new_vec

                # 4) Compute output[t, h, K] = scale * q[t, h, :] @ state_new[h, v, :]
                # For each h, we need q_vec[h, :] @ state_new[h, v, :]. Note state_new has shape [H, V, K].
                # We compute per v_i:
                for v_i in range(V):
                    state_new_vec = new_state[seq_idx, h, v_i, :]  # [K]
                    q_vec = q[t, h, :].to(torch.float32)  # [K]
                    out_vec = scale * torch.matmul(q_vec.view(1, K), state_new_vec.view(K, 1))
                    output[t, h, :] = out_vec.squeeze(1)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
