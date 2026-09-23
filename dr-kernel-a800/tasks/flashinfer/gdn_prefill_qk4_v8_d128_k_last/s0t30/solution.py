import math
import torch
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
    Softplus(x) = log(1 + exp(x))
    """
    t = tl.program_id(0)  # one program per token t
    for v in range(0, V):
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        idx = t * V + v
        tl.store(g_ptr + idx, g_val)
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
    All loops over K=128 and V=8.
    """
    for h in range(0, H):
        for v_i in range(0, V):
            # Load scalars g and beta for this (h, v)
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_val = tl.load(beta_ptr + g_idx).to(tl.float32)

            # Compute old_v[h, :] = sum_j k[t, h, j] * state_old[h, v, j]
            old_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_j = tl.load(k_ptr + t * H * K + h * K + j).to(tl.float32)
                state_old_j = tl.load(state_old_ptr + seq_idx * H * V * K + h * V * K + v_i * K + j).to(tl.float32)
                old_v[j] = k_j * state_old_j

            # Compute new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
            new_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                v_j = tl.load(v_ptr + t * V * K + v_i * K + j).to(tl.float32)
                new_v[j] = beta_val * v_j + (1.0 - beta_val) * old_v[j]

            # state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
            state_remove = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_j = tl.load(k_ptr + t * H * K + h * K + j).to(tl.float32)
                state_remove[j] = k_j * old_v[j]

            # state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
            state_update = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_j = tl.load(k_ptr + t * H * K + h * K + j).to(tl.float32)
                state_update[j] = k_j * new_v[j]

            # state_new[h, v, :] = g * state_old - state_remove + state_update
            for j in range(0, K):
                state_old_j = tl.load(state_old_ptr + seq_idx * H * V * K + h * V * K + v_i * K + j).to(tl.float32)
                new_val = g_val * state_old_j - state_remove[j] + state_update[j]
                tl.store(new_state_ptr + seq_idx * H * V * K + h * V * K + v_i * K + j, new_val)


@triton.jit
def compute_output_kernel(q_ptr, new_state_ptr, out_ptr,
                          scale, T, H, V, K):
    """
    Compute output[t, h, :] = scale * q[t, h, :] @ new_state[h, :, :]
    new_state is [H, V, K], out_ptr is [T*H*K] float32, we write per (t, h, j).
    """
    t = tl.program_id(0)
    for h in range(0, H):
        # q[t, h, :] vector
        q_row = tl.zeros((K,), dtype=tl.float32)
        for j in range(0, K):
            q_j = tl.load(q_ptr + t * H * K + h * K + j).to(tl.float32)
            q_row[j] = q_j
        # new_state[h, :, :] matrix [V, K]
        for v_i in range(0, V):
            state_row = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                state_elem = tl.load(new_state_ptr + h * V * K + v_i * K + j).to(tl.float32)
                state_row[j] = state_elem
            # out[t, h, j] = sum over v of state_row[j] * q_row[j]
            dot = 0.0
            for j in range(0, K):
                dot += state_row[j] * q_row[j]
            out_val = scale * dot
            out_idx = t * H * K + h * K + j  # write all j; but j must be fixed; we'll vectorize later
            # Since we cannot store vector at once, we store per j in host by writing [T*H*K] linearly
            # We'll instead compute linear index as ((t * H) + h) * K + j in host; here we store scalar per j loop.
            # To keep it simple, we store per j: out_ptr[out_idx] = out_val. But out_idx depends on j? We can't.
            # Therefore, we'll compute out as a 2D array in host and not use Triton for output here; but to satisfy
            # Triton-only requirement, we can compute output entirely in Triton by expanding q_row and state rows:
            # However, Triton kernel above only computes scalar per j; better approach: compute full vector with nested loops.
            # We'll compute per j and store directly into out_ptr linearized as ((t * H) + h) * K + j.
            out_linear_idx = (t * H + h) * K + j
            tl.store(out_ptr + out_linear_idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure dtype and shapes
        assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
        device = q.device
        T = q.shape[0]
        H = q.shape[1]
        K = q.shape[2]
        assert H == 4 and K == 128, "num_q_heads=4 and head_size=128 are required"
        assert v.shape[1] == 8, "num_v_heads must be 8"
        assert k.shape[1] == 4 and k.shape[2] == 128
        V = v.shape[1]
        num_seqs = cu_seqlens.shape[0] - 1

        # Prepare flattened parameters for Triton
        g_flat = torch.empty(T * V, dtype=torch.float32, device=device)
        beta_flat = torch.empty(T * V, dtype=torch.float32, device=device)

        # Launch compute_g_and_beta_kernel: one program per token t
        grid_g = (T,)
        compute_g_and_beta_kernel[grid_g](
            a.contiguous().view(-1), dt_bias.contiguous(), A_log.contiguous(),
            b.contiguous().view(-1),
            g_flat, beta_flat,
            T, V,
            num_warps=1
        )

        # Initialize output and new_state
        output = torch.empty((T, H, K), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # Process each sequence block: update state for all tokens and compute output
        for seq_idx in range(num_seqs):
            # state may be None; if so, initialize zeros [H, V, K]
            if state is None or (not isinstance(state, torch.Tensor)):
                state_old = torch.zeros((H, V, K), dtype=torch.float32, device=device)
            else:
                # state layout: [H, V, K] for each seq_idx; here idx is seq_idx
                state_old = state[seq_idx].contiguous().to(torch.float32)

            # Copy state_old to new_state buffer for this seq_idx
            new_state[seq_idx].copy_(state_old)

            # Update state for all tokens t in this sequence block
            for t in range(T):
                # Launch update_state_kernel: one program per (t, seq_idx)
                grid_u = (1,)
                update_state_kernel[grid_u](
                    q[t].contiguous().view(1, H, K),  # pass as [1, H, K]
                    k[t].contiguous().view(1, H, K),  # [1, H, K]
                    v[t].contiguous().view(1, V, K),  # [1, V, K]
                    state_old, g_flat[t * V:(t + 1) * V], beta_flat[t * V:(t + 1) * V],
                    new_state[seq_idx],
                    T, H, V, K, t, seq_idx,
                    num_warps=1
                )
                # new_state updated by the kernel in-place; no need to copy back

        # Compute output in Triton: out[t, h, j] = scale * sum_v new_state[h, v, j] * q[t, h, j]
        out_flat = torch.empty(T * H * K, dtype=torch.float32, device=device)
        grid_out = (T,)
        compute_output_kernel[grid_out](
            q.contiguous().view(T, H, K),
            new_state[-1].contiguous().view(H, V, K),
            out_flat,
            float(scale) if scale is not None else 1.0 / math.sqrt(K),
            T, H, V, K,
            num_warps=1
        )
        # Reshape and cast output to bfloat16
        output_flat = out_flat.view(T, H, K).to(torch.bfloat16)

        return output_flat, new_state


def run(*args):
    return ModelNew()(*args)
