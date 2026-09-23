import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, A_log_ptr, b_ptr,
                              g_ptr, beta_ptr,
                              T, V):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v])) for all t, v
    and beta[t, v] = sigmoid(b[t, v]), writing results to g_ptr and beta_ptr.
    g_ptr: [T*V] float32
    beta_ptr: [T*V] float32
    """
    t = tl.program_id(0)  # one program per token t
    for v in range(0, V):
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        idx = t * V + v
        tl.store(g_ptr + idx, g_val)
        # beta = sigmoid(b) = 1 / (1 + exp(-b))
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(beta_ptr + idx, beta_val)


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_ptr, beta_ptr,
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
    # One program per (t, seq_idx) processes all h, v via loops
    for h in range(0, H):
        for v_i in range(0, V):
            # Load per-token vectors for q and k
            q_vec = tl.zeros((K,), dtype=tl.float32)
            k_vec = tl.zeros((K,), dtype=tl.float32)
            v_vec = tl.zeros((K,), dtype=tl.float32)
            for kk in range(0, K):
                q_vec[kk] = tl.load(q_ptr + t * H * K + h * K + kk).to(tl.float32)
                k_vec[kk] = tl.load(k_ptr + t * H * K + h * K + kk).to(tl.float32)
                v_vec[kk] = tl.load(v_ptr + t * V * K + v_i * K + kk).to(tl.float32)

            # Load g and beta scalars for (h, v_i)
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_val = tl.load(beta_ptr + g_idx).to(tl.float32)

            # Compute old_v[h, :]
            old_v = tl.zeros((K,), dtype=tl.float32)
            for kk in range(0, K):
                acc = 0.0
                for jj in range(0, K):
                    # state_old_ptr flattened as [H, V, K] contiguous: offset = (h * V + v_i) * K + jj
                    offset = (h * V + v_i) * K + jj
                    s_old = tl.load(state_old_ptr + offset).to(tl.float32)
                    k_elem = tl.load(k_ptr + t * H * K + h * K + jj).to(tl.float32)
                    acc += s_old * k_elem
                old_v[kk] = acc

            # Compute new_v[h, :]
            new_v = tl.zeros((K,), dtype=tl.float32)
            for kk in range(0, K):
                new_v[kk] = beta_val * v_vec[kk] + (1.0 - beta_val) * old_v[kk]

            # Compute state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
            state_remove = tl.zeros((K,), dtype=tl.float32)
            for jj in range(0, K):
                k_elem = tl.load(k_ptr + t * H * K + h * K + jj).to(tl.float32)
                state_remove += k_elem * old_v[jj]

            # Compute state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
            state_update = tl.zeros((K,), dtype=tl.float32)
            for jj in range(0, K):
                k_elem = tl.load(k_ptr + t * H * K + h * K + jj).to(tl.float32)
                state_update += k_elem * new_v[jj]

            # Compute state_new[h, v_i, :] = g * state_old - state_remove + state_update
            for kk in range(0, K):
                offset = (h * V + v_i) * K + kk
                s_old = tl.load(state_old_ptr + offset).to(tl.float32)
                s_new = g_val * s_old - state_remove[kk] + state_update[kk]
                tl.store(new_state_ptr + offset, s_new)


@triton.jit
def compute_output_kernel(q_ptr, new_state_ptr, out_ptr, scale, T, H, V, K):
    """
    Compute output[t, h, k] = scale * sum_v q[t, h, k] * new_state[t, h, v, k]
    One program per token t; output is [T, H, K] float32 (later cast to bfloat16).
    """
    t = tl.program_id(0)
    for h in range(0, H):
        out_vec = tl.zeros((K,), dtype=tl.float32)
        for v_i in range(0, V):
            for k in range(0, K):
                q_val = tl.load(q_ptr + t * H * K + h * K + k).to(tl.float32)
                # new_state layout: [H, V, K] flattened
                state_val = tl.load(new_state_ptr + (t * H + h) * V * K + v_i * K + k).to(tl.float32)
                out_vec[k] += scale * q_val * state_val
        for k in range(0, K):
            tl.store(out_ptr + t * H * K + h * K + k, out_vec[k])


class ModelNew(nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-only implementation:
        - Compute g and beta in Triton
        - Update state in Triton
        - Compute output in Triton
        Returns (output, new_state)
        """
        device = q.device
        T, H, K = q.shape
        V = v.shape[1]
        assert H == 4, "num_q_heads must be 4"
        assert K == 128, "head_size must be 128"
        assert V == 8, "num_v_heads must be 8"

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if state is not None:
            state = state.contiguous()

        # Allocate g and beta as 1D float32
        g_flat = torch.empty(T * V, dtype=torch.float32, device=device)
        beta_flat = torch.empty(T * V, dtype=torch.float32, device=device)

        # Launch compute_g_and_beta_kernel
        grid_g = (T,)
        compute_g_and_beta_kernel[grid_g](
            a.float(), dt_bias.float(), A_log.float(), b.float(),
            g_flat, beta_flat, T, V
        )

        # Prepare new_state: [num_seqs, H, V, K] float32
        num_seqs = cu_seqlens.numel() - 1
        new_state = torch.empty((num_seqs, H, V, K), dtype=torch.float32, device=device)
        if state is not None:
            # Copy state to new_state as float32 per seq_idx (k-last layout)
            for seq_idx in range(num_seqs):
                new_state[seq_idx].copy_(state[seq_idx].float())

        # Update state per sequence block and token
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            if seq_end - seq_start <= 0:
                continue
            # For each token t in this block
            for t in range(seq_start, seq_end):
                grid_update = (1,)
                update_state_kernel[grid_update](
                    q, k, v, new_state[seq_idx].contiguous().view(-1), g_flat, beta_flat,
                    new_state[seq_idx].contiguous().view(-1),
                    T, H, V, K, t, seq_idx
                )

        # Compute output using Triton for the last sequence block (num_seqs - 1)
        out = torch.empty((T, H, K), dtype=torch.float32, device=device)
        grid_out = (T,)
        compute_output_kernel[grid_out](
            q, new_state[-1].contiguous().view(-1), out, float(scale), T, H, V, K
        )
        output = out.to(torch.bfloat16)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
