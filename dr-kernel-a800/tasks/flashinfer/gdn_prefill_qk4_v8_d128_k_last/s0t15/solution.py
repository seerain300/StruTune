import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, T, V):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v])) for all t, v.
    g_ptr: [T*V] float32
    """
    t = tl.program_id(0)
    for v in range(0, V):
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        idx = t * V + v
        tl.store(g_ptr + idx, g_val)


@triton.jit
def compute_beta_kernel(b_ptr, beta_ptr, T, V):
    """
    Compute beta[t, v] = sigmoid(b[t, v]) for all t, v.
    beta_ptr: [T*V] float32
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    t = tl.program_id(0)
    for v in range(0, V):
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        idx = t * V + v
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
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_val = tl.load(beta_ptr + g_idx).to(tl.float32)

            # Initialize accumulators for vector operations
            old_v = tl.zeros((K,), dtype=tl.float32)

            # Compute old_v = k @ state_old[h, v, :]
            for j in range(0, K):
                k_elem = tl.load(k_ptr + t * H * K + h * K + j).to(tl.float32)
                state_old_j = tl.load(state_old_ptr + seq_idx * (H * V * K) + h * V * K + v_i * K + j).to(tl.float32)
                old_v[j] = k_elem * state_old_j

            # v_vec = v[t, v_i, :]
            v_vec = tl.load(v_ptr + t * V * K + v_i * K).to(tl.float32)
            new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

            # state_remove = sum_j k[t, h, j] * old_v[h, j]
            state_remove = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_elem = tl.load(k_ptr + t * H * K + h * K + j).to(tl.float32)
                old_v_j = old_v[j]
                state_remove += k_elem * old_v_j

            # state_update = sum_j k[t, h, j] * new_v[h, j]
            state_update = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_elem = tl.load(k_ptr + t * H * K + h * K + j).to(tl.float32)
                new_v_j = new_v[j]
                state_update += k_elem * new_v_j

            # state_new[h, v_i, :] = g_val * state_old[h, v_i, :] - state_remove + state_update
            for j in range(0, K):
                state_old_j = tl.load(state_old_ptr + seq_idx * (H * V * K) + h * V * K + v_i * K + j).to(tl.float32)
                state_new_j = g_val * state_old_j - state_remove[j] + state_update[j]
                tl.store(new_state_ptr + seq_idx * (H * V * K) + h * V * K + v_i * K + j, state_new_j)


@triton.jit
def row_matmul_kernel(q_ptr, state_ptr, out_ptr, scale,
                      T, H, V, K):
    """
    Compute out[t, h, k] = scale * sum_v q[t, h, k] @ state[h, v, k] for all h,v,k.
    state_ptr corresponds to last sequence block (num_seqs - 1).
    out_ptr: [T * H * K] float32
    """
    t = tl.program_id(0)  # one program per token
    for h in range(0, H):
        q_row = tl.zeros((K,), dtype=tl.float32)
        for k in range(0, K):
            q_row[k] = tl.load(q_ptr + t * H * K + h * K + k).to(tl.float32)
        out_vec = tl.zeros((K,), dtype=tl.float32)
        for v in range(0, V):
            state_vec = tl.zeros((K,), dtype=tl.float32)
            base_state = (0)  # last sequence block index not needed explicitly since state_ptr already points to last block
            for k in range(0, K):
                state_vec[k] = tl.load(state_ptr + h * V * K + v * K + k).to(tl.float32)
            out_vec += scale * (q_row @ state_vec)
        base_out = t * H * K + h * K
        for k in range(0, K):
            tl.store(out_ptr + base_out + k, out_vec[k])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed head sizes from the original assertions
        self.H = 4
        self.V = 8
        self.K = 128

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-only implementation of the forward.
        Returns (output, new_state).
        """
        device = q.device
        T = q.shape[0]
        H = self.H
        V = self.V
        K = self.K

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Prepare new_state: initialize from provided state if any; else zeros
        num_seqs = cu_seqlens.shape[0] - 1
        new_state = torch.empty((num_seqs, H, V, K), dtype=torch.float32, device=device)

        if state is not None:
            # Copy state to new_state: state is [num_seqs, H, V, K]
            if state.shape[0] != num_seqs:
                new_state.zero_()
            else:
                for seq_idx in range(num_seqs):
                    new_state[seq_idx].copy_(state[seq_idx].contiguous().float())
        else:
            new_state.zero_()

        # Compute g and beta with Triton
        g_flat = torch.empty((T * V,), dtype=torch.float32, device=device)
        beta_flat = torch.empty((T * V,), dtype=torch.float32, device=device)

        grid_g = (T,)
        compute_g_kernel[grid_g](a, dt_bias, A_log, g_flat, T, V, num_warps=1)
        grid_beta = (T,)
        compute_beta_kernel[grid_beta](b, beta_flat, T, V, num_warps=1)

        # Update state for each sequence block and token
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            if seq_end - seq_start <= 0:
                continue
            for t in range(seq_start, seq_end):
                grid_upd = (1,)
                update_state_kernel[grid_upd](
                    q, k, v, new_state.view(-1), g_flat, beta_flat,
                    new_state.view(-1),
                    T, H, V, K, t, seq_idx, num_warps=1
                )

        # Compute output using Triton: use last sequence block
        out = torch.empty((T, H, K), dtype=torch.float32, device=device)
        grid_out = (T,)
        row_matmul_kernel[grid_out](
            q, new_state[-1].view(-1), out, float(scale),
            T, H, V, K, num_warps=1
        )
        output = out.to(torch.bfloat16)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
