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
    # one program per token t
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
        # beta = sigmoid(b)
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
    state_old_ptr points to [H, V, K] for seq_idx
    new_state_ptr points to [H, V, K] for seq_idx
    """
    # one program per (t, seq_idx), loops over H and V
    for h in range(0, H):
        for v_i in range(0, V):
            # scalar gating
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_val = tl.load(beta_ptr + g_idx).to(tl.float32)

            # old_v[h, :]
            old_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_elem = tl.load(k_ptr + (t + seq_idx * T) * H * K + h * K + j).to(tl.float32)  # t could be > seq_start; use seq_idx*t mapping
                state_elem = tl.load(state_old_ptr + h * V * K + v_i * K + j).to(tl.float32)
                old_v += k_elem * state_elem

            # new_v[h, :]
            new_v = tl.zeros((K,), dtype=tl.float32)
            v_elem = tl.load(v_ptr + (t + seq_idx * T) * V * K + v_i * K).to(tl.float32)  # [K]
            new_v = beta_val * v_elem + (1.0 - beta_val) * old_v

            # state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
            state_remove = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_elem = tl.load(k_ptr + (t + seq_idx * T) * H * K + h * K + j).to(tl.float32)
                state_remove += k_elem * tl.load(old_v + j).to(tl.float32)

            # state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
            state_update = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_elem = tl.load(k_ptr + (t + seq_idx * T) * H * K + h * K + j).to(tl.float32)
                state_update += k_elem * tl.load(new_v + j).to(tl.float32)

            # new_state[h, v_i, :] = g_val * state_old[h, v_i, :] - state_remove + state_update
            state_old_row = tl.load(state_old_ptr + h * V * K + v_i * K).to(tl.float32)  # [K]
            new_state_row = (g_val * state_old_row) - state_remove + state_update
            # Store new_state[h, v_i, :]
            for j in range(0, K):
                tl.store(new_state_ptr + h * V * K + v_i * K + j, new_state_row[j])


@triton.jit
def compute_sqrt_scale_kernel(x_ptr, out_ptr, N):
    """
    Compute y[i] = sqrt(x[i]) for i in [0, N), write to out_ptr.
    Used to implement sqrt(scale) in Triton (host passes a single-element tensor).
    """
    i = tl.program_id(0)
    x = tl.load(x_ptr + i).to(tl.float32)
    y = tl.sqrt(x)
    tl.store(out_ptr + i, y)


@triton.jit
def compute_out_seq_kernel(q_ptr, new_state_ptr, out_ptr, scale,
                           T, H, V, K, t, seq_idx):
    """
    Compute output for token t in sequence block seq_idx:
    out[t, h, k] = scale * q[t, h, k] @ sum_v new_state[seq_idx, h, v, k].
    """
    h = 0  # one program per t; loop over h
    # For this kernel, we assume grid = (T,) and we compute for each t.
    # We need to loop over H to store full [H, K] row for each t.
    # But Triton grid limits per launch, so we launch one program per (t, h) would be better.
    # Instead, we launch grid=(T, H) and do per (t, h) compute.
    # However, Triton requires static loops; better approach: compute per (t, h) in separate launches, or write out per t.
    # Here, we compute per t and per h in the same program; H is small (4), so we loop over h.

    # Sum over v: total[h, k] = sum_v new_state[seq_idx, h, v, k]
    for h in range(0, H):
        total = tl.zeros((K,), dtype=tl.float32)
        for v_i in range(0, V):
            state_row = tl.load(new_state_ptr + h * V * K + v_i * K).to(tl.float32)  # [K]
            total += state_row
        # q_row = q[t, h, :] [K]
        q_row = tl.load(q_ptr + t * H * K + h * K).to(tl.float32)  # [K]
        out_row = scale * total  # scalar scale * vector
        # Store out[t, h, :]
        for j in range(0, K):
            tl.store(out_ptr + t * H * K + h * K + j, out_row[j])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure contiguity
        device = q.device
        T, H, K = q.shape
        assert H == 4, "num_q_heads must be 4"
        V = v.shape[1]
        assert V == 8, "num_v_heads must be 8"
        assert K == 128, "head_size must be 128"
        num_seqs = cu_seqlens.size(0) - 1

        # Prepare g and beta as 1D arrays [T*V]
        g_flat = torch.empty(T * V, dtype=torch.float32, device=device)
        beta_flat = torch.empty(T * V, dtype=torch.float32, device=device)

        # Launch compute_g_and_beta_kernel: one program per token t
        grid_g_beta = (T,)
        compute_g_and_beta_kernel[grid_g_beta](
            a, dt_bias, A_log, b, g_flat, beta_flat, T, V, num_warps=1
        )

        # Output accumulator: [T, H, K] float32, cast to bfloat16 at the end
        output_accum = torch.zeros((T, H, K), dtype=torch.float32, device=device)

        # new_state buffer: [num_seqs, H, V, K] float32
        new_state = torch.empty((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # For each sequence block, update state and compute output per token
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Initialize state_old for this block: [H, V, K] float32
            if state is not None:
                state_old = state[seq_idx].clone().contiguous().float()  # [H, V, K]
            else:
                state_old = torch.zeros((H, V, K), dtype=torch.float32, device=device)

            # Process each token in this block
            for t in range(seq_len):
                # Update state
                update_state_kernel[(1,)](
                    q, k, v, state_old, g_flat, beta_flat,
                    new_state[seq_idx],
                    T, H, V, K, t + seq_start, seq_idx, num_warps=1
                )
            # Compute and accumulate output per token t for this block
            out_block = torch.empty((T, H, K), dtype=torch.float32, device=device)
            # Launch compute_out_seq_kernel: one program per (t, h) pair
            # Triton does not support 2D grid easily in simple codegen, so we loop in host over h
            # However, Triton kernel supports grid tuple; we can launch per t and compute for each h.
            for t in range(seq_len):
                # For each h, compute out[t, h, :]
                # We need to pass h as well; Triton expects a grid. Use a small loop over h and grid=(T,) and rely on host loop.
                # Better: write a grid=(T, H) kernel. Triton’s simple interface does not expose 2D grid easily, so we do:
                # We will launch compute_out_seq_kernel per t and compute for each h inside the kernel by looping.
                # But to keep correctness and Triton-only, we will do per h inside kernel using a static range over H.
                # We pass H and V; kernel loops over H. We can set grid=(T,) and kernel handles H internally.

                # One program per t
                compute_out_seq_kernel[(1,)](
                    q, new_state[seq_idx].contiguous().view(-1),
                    out_block[t], float(scale), T, H, V, K, t + seq_start, seq_idx, num_warps=1
                )
            # Accumulate output: add this block’s out for each token
            # Note: out_block has shape [seq_len, H, K]; we add it to output_accum at positions [t + seq_start, :, :]
            for t_i in range(seq_len):
                t = t_i + seq_start
                output_accum[t] += out_block[t_i]

        # Cast output to bfloat16 as required
        output = output_accum.to(torch.bfloat16)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
