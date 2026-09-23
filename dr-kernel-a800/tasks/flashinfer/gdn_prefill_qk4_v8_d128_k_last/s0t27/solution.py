import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_g_flat_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, T, V):
    """
    Compute g_flat[t*V + v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v]))
    for all t in [0, T), v in [0, V).
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
def compute_beta_flat_kernel(b_ptr, beta_ptr, T, V):
    """
    Compute beta_flat[t*V + v] = sigmoid(b[t, v]) for all t in [0, T), v in [0, V).
    beta_ptr: [T*V] float32
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    t = tl.program_id(0)  # one program per token t
    for v in range(0, V):
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(beta_ptr + t * V + v, beta_val)


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_flat_ptr, beta_flat_ptr,
                        new_state_ptr,
                        T, H, V, K, t, seq_idx):
    """
    Update state for sequence block seq_idx at token t:
    For each h in [0, H), v in [0, V):
      old_v[h, :] = sum_j k[t, h, j] * state_old[h, v, j]
      new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
      state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
      state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
      state_new[h, v, :] = g[h, v] * state_old[h, v, :] - state_remove[h, :] + state_update[h, :]
    All tensors are contiguous with strides:
      q: [T, H, K], k: [T, H, K], v: [T, V, K], state_old/new: [H, V, K]
    """
    # H, V, K are compile-time constants or known at host; we use fixed loops for correctness.
    for h in range(0, H):
        for v_i in range(0, V):
            # Load scalars for this (h, v_i)
            g_val = tl.load(g_flat_ptr + h * V + v_i).to(tl.float32)
            beta_val = tl.load(beta_flat_ptr + h * V + v_i).to(tl.float32)

            # old_v[h, :] = sum_j k[t, h, j] * state_old[h, v_i, j]
            old_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_val = tl.load(k_ptr + t * H * K + h * K + j).to(tl.float32)
                state_val = tl.load(state_old_ptr + h * V * K + v_i * K + j).to(tl.float32)
                old_v[j] = k_val * state_val

            # v_vec = v[t, v_i, :]
            v_vec = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                v_val = tl.load(v_ptr + t * V * K + v_i * K + j).to(tl.float32)
                v_vec[j] = v_val

            # new_v[h, :] = beta * v + (1 - beta) * old_v
            new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

            # state_remove[h, :] = sum_j k[t, h, j] * old_v[j]
            state_remove = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_val = tl.load(k_ptr + t * H * K + h * K + j).to(tl.float32)
                state_remove[j] = k_val * old_v[j]

            # state_update[h, :] = sum_j k[t, h, j] * new_v[j]
            state_update = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_val = tl.load(k_ptr + t * H * K + h * K + j).to(tl.float32)
                state_update[j] = k_val * new_v[j]

            # state_new[h, v_i, :] = g * state_old - state_remove + state_update
            state_old_vec = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                state_old_vec[j] = tl.load(state_old_ptr + h * V * K + v_i * K + j).to(tl.float32)

            for j in range(0, K):
                new_elem = (g_val * state_old_vec[j]) - state_remove[j] + state_update[j]
                tl.store(new_state_ptr + seq_idx * (H * V * K) + h * V * K + v_i * K + j, new_elem)


@triton.jit
def compute_output_kernel(q_ptr, new_state_ptr, out_ptr, scale, T, H, V, K, t, seq_idx):
    """
    Compute output for token t and sequence block seq_idx:
    For each h in [0, H):
      out[t, h, :] = scale * q[t, h, :] @ new_state[seq_idx, h, :]
      Implement matmul via loop over K: out[h, :] = sum_j q[t, h, j] * new_state[seq_idx, h, j]
    out_ptr: [T, H, K] float32
    """
    for h in range(0, H):
        out_vec = tl.zeros((K,), dtype=tl.float32)
        for j in range(0, K):
            q_j = tl.load(q_ptr + t * H * K + h * K + j).to(tl.float32)
            new_j = tl.load(new_state_ptr + seq_idx * (H * V * K) + h * V * K + j).to(tl.float32)
            out_vec[j] = scale * q_j * new_j
        # Store out_vec at out[t, h, :]
        for j in range(0, K):
            tl.store(out_ptr + t * H * K + h * K + j, out_vec[j])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed dims per original code
        self.H = 4
        self.V = 8
        self.K = 128

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        q: [T, H, K], bfloat16
        k: [T, H, K], bfloat16
        v: [T, V, K], bfloat16
        state: [num_seqs, H, V, K], float32 (initial or previous state)
        A_log: [V], float32
        a: [T, V], bfloat16
        dt_bias: [V], float32
        b: [T, V], bfloat16
        cu_seqlens: [num_seqs+1], int64
        scale: float
        Output: (output: [T, H, K], new_state: [num_seqs, H, V, K])
        """
        device = q.device
        T = q.shape[0]
        num_seqs = cu_seqlens.shape[0] - 1
        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        A_log = A_log.contiguous()
        a = a.contiguous()
        dt_bias = dt_bias.contiguous()
        b = b.contiguous()

        # Allocate flat g and beta
        g_flat = torch.empty((T * self.V,), dtype=torch.float32, device=device)
        beta_flat = torch.empty((T * self.V,), dtype=torch.float32, device=device)

        # Compute g_flat and beta_flat with Triton
        grid = (T,)
        compute_g_flat_kernel[grid](a, dt_bias, A_log, g_flat, T, self.V, num_warps=1)
        compute_beta_flat_kernel[grid](b, beta_flat, T, self.V, num_warps=1)

        # Prepare output tensor as float32 for computation
        output = torch.empty((T, self.H, self.K), dtype=torch.float32, device=device)

        # Prepare new state tensor [num_seqs, H, V, K] in float32
        new_state = torch.zeros((num_seqs, self.H, self.V, self.K), dtype=torch.float32, device=device)

        # For each sequence block, update state and compute output
        for seq_idx in range(0, num_seqs):
            # Run update kernel for all tokens
            for t in range(0, T):
                update_state_kernel[(1,)](
                    q, k, v, state[seq_idx], g_flat, beta_flat, new_state[seq_idx],
                    T, self.H, self.V, self.K, t, seq_idx, num_warps=1
                )
            # Compute output for last seq block using Triton
            # Note: output is per token t for each seq block. We compute it here for the last seq block.
            compute_output_kernel[(T,)](q, new_state[-1], output, float(scale), T, self.H, self.V, self.K, 0, seq_idx, num_warps=1)

        # Return output in bfloat16 to match original signature, and new_state
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
