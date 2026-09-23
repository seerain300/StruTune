import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_g_beta_kernel(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    H, V,
):
    """
    Compute:
      g[h, v] = exp(-exp(A_log[v]) * softplus(a[h, v] + dt_bias[v])) for h in [0..H-1], v in [0..V-1]
      beta[v] = sigmoid(b[h, v]) for v in [0..V-1] (same v for all h)
    Write results to g_ptr[H*V] and beta_ptr[V] as float32.
    """
    for h in range(0, H):
        for v_i in range(0, V):
            a_val = tl.load(a_ptr + h * V + v_i).to(tl.float32)
            dt_bias_val = tl.load(dt_bias_ptr + v_i).to(tl.float32)
            A_log_val = tl.load(A_log_ptr + v_i).to(tl.float32)

            # softplus(x) = log(1 + exp(x))
            softplus_val = tl.log(1.0 + tl.exp(a_val + dt_bias_val))
            # g = exp(-exp(A_log) * softplus(a + dt_bias))
            g_val = tl.exp(-tl.exp(A_log_val) * softplus_val)
            tl.store(g_ptr + h * V + v_i, g_val)

            # beta = sigmoid(b) = 1 / (1 + exp(-b))
            b_val = tl.load(b_ptr + h * V + v_i).to(tl.float32)
            beta_val = 1.0 / (1.0 + tl.exp(-b_val))
            tl.store(beta_ptr + v_i, beta_val)


@triton.jit
def update_state_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr,
    T, H, V, K,
    t,          # token index within current sequence block
    seq_idx,    # sequence block index
):
    """
    Update state for token t and sequence block seq_idx.
    q: [T, H, K], k: [T, H, K], v: [T, V, K]
    state: [num_seqs, H, V, K] (k-last layout: last dim is K)
    g_ptr: [H*V] float32, beta_ptr: [V] float32
    """
    for h in range(0, H):
        for v_i in range(0, V):
            # Load g[h, v] and beta[v]
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_val = tl.load(beta_ptr + v_i).to(tl.float32)

            # Build q_vec[t, h, K] and k_vec[t, h, K]
            q_vec = tl.zeros([K], dtype=tl.float32)
            for k_i in range(0, K):
                q_off = t * (H * K) + h * K + k_i
                q_elem = tl.load(q_ptr + q_off).to(tl.float32)
                q_vec[k_i] = q_elem

            k_vec = tl.zeros([K], dtype=tl.float32)
            for k_i in range(0, K):
                k_off = t * (H * K) + h * K + k_i
                k_elem = tl.load(k_ptr + k_off).to(tl.float32)
                k_vec[k_i] = k_elem

            # Load v_vec[t, v_i, K]
            v_vec = tl.zeros([K], dtype=tl.float32)
            for k_i in range(0, K):
                v_off = t * (V * K) + v_i * K + k_i
                v_elem = tl.load(v_ptr + v_off).to(tl.float32)
                v_vec[k_i] = v_elem

            # Load state_old[h, v_i, K] as vector: state_old_vec[j] = state[seq_idx, h, v_i, j]
            state_old_vec = tl.zeros([K], dtype=tl.float32)
            for j in range(0, K):
                state_off = seq_idx * (H * V * K) + h * (V * K) + v_i * K + j
                state_elem = tl.load(state_ptr + state_off).to(tl.float32)
                state_old_vec[j] = state_elem

            # Compute state_old as 2D matrix [K, K] for this (h, v_i)
            state_old_mat = tl.zeros([K, K], dtype=tl.float32)
            for i in range(0, K):
                for j in range(0, K):
                    state_off_mat = seq_idx * (H * V * K) + h * (V * K) + v_i * K + j
                    state_elem_mat = tl.load(state_ptr + state_off_mat).to(tl.float32)
                    state_old_mat[i, j] = state_elem_mat

            # old_v = k_vec @ state_old_mat (1xK @ KxK -> 1xK), elementwise inner product:
            old_v = tl.zeros([K], dtype=tl.float32)
            for k_j in range(0, K):
                dot_val = 0.0
                for kk in range(0, K):
                    dot_val += k_vec[kk] * state_old_mat[kk, k_j]
                old_v[k_j] = dot_val

            # new_v = beta * v + (1 - beta) * old_v
            new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

            # Compute state_remove = k^T @ old_v (k_vec is [K], old_v is [K])
            state_remove = tl.zeros([K], dtype=tl.float32)
            for k_j in range(0, K):
                dot_j = 0.0
                for kk in range(0, K):
                    dot_j += k_vec[kk] * old_v[k_j]
                state_remove[k_j] = dot_j

            # state_update = k^T @ new_v
            state_update = tl.zeros([K], dtype=tl.float32)
            for k_j in range(0, K):
                dot_j = 0.0
                for kk in range(0, K):
                    dot_j += k_vec[kk] * new_v[k_j]
                state_update[k_j] = dot_j

            # Update state_new = g * state_old - state_remove + state_update
            state_new_vec = g_val * state_old_vec - state_remove + state_update

            # Store back to state[seq_idx, h, v_i, K]
            for k_i in range(0, K):
                state_off = seq_idx * (H * V * K) + h * (V * K) + v_i * K + k_i
                tl.store(state_ptr + state_off, state_new_vec[k_i])


class ModelNew(nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized forward:
        - compute_g_beta_kernel in Triton to compute g and beta in float32.
        - update_state_kernel in Triton to update state per token per sequence block.
        - Output computed in PyTorch (small).
        """
        device = q.device
        T = q.shape[0]
        H = q.shape[1]
        V = v.shape[1]
        K = q.shape[2]
        num_seqs = cu_seqlens.numel() - 1

        # Ensure contiguous tensors
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        a = a.contiguous()
        b = b.contiguous()
        A_log = A_log.contiguous()
        dt_bias = dt_bias.contiguous()

        # Allocate g_flat and beta_flat (float32) on device
        g_flat = torch.empty(H * V, dtype=torch.float32, device=device)
        beta_flat = torch.empty(V, dtype=torch.float32, device=device)

        # Launch kernel to compute g and beta
        compute_g_beta_kernel[(1,)](
            A_log, a, dt_bias, b,
            g_flat, beta_flat,
            H, V,
            num_warps=4, num_stages=2
        )

        # Prepare state tensor: original code doesn't use 'state' arg; initialize zeros.
        new_state = torch.zeros((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # Output tensor [T, H, K] bfloat16
        output = torch.empty((T, H, K), dtype=torch.bfloat16, device=device)

        # Update state per sequence block and token
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Run Triton update for each token
            for t in range(seq_start, seq_end):
                update_state_kernel[(1,)](
                    q, k, v, new_state[seq_idx], g_flat, beta_flat,
                    T, H, V, K,
                    t, seq_idx,
                    num_warps=4, num_stages=2
                )

                # Compute output for this token using PyTorch (small and correct).
                # output[t, h, k] = scale * sum_v (q[t, h, k] @ state[seq_idx, h, v, k])
                for h in range(H):
                    out_vec = torch.zeros(K, dtype=torch.float32, device=device)
                    for v_i in range(V):
                        q_vec = q[t, h, :].float()
                        state_vec = new_state[seq_idx, h, v_i, :].float()
                        out_vec += torch.dot(q_vec, state_vec)
                    output[t, h, :] = (out_vec * float(scale)).to(torch.bfloat16)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
