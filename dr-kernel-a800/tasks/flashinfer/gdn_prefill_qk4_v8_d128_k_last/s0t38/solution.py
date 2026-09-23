import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr,
                     T, V: tl.constexpr):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v]))
    for t in [0, T), v in [0, V), store into g_ptr as 1D array of length T*V.
    """
    t = tl.program_id(0)  # one program per token
    for v in range(0, V):
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        tl.store(g_ptr + t * V + v, g_val)


@triton.jit
def compute_beta_kernel(b_ptr, beta_ptr, T, V: tl.constexpr):
    """
    Compute beta[t, v] = sigmoid(b[t, v]) for t in [0, T), v in [0, V),
    store into beta_ptr as 1D array of length T*V.
    """
    t = tl.program_id(0)  # one program per token
    for v in range(0, V):
        b_val = tl.load(b_ptr + t * V + v).to(tl.float32)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(beta_ptr + t * V + v, beta_val)


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_ptr, beta_ptr,
                        new_state_ptr,
                        T, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr, t, seq_idx):
    """
    Update state per token t within sequence block seq_idx:
    For each h in [0, H), v in [0, V):
      old_v[h, :] = sum_j k[t, h, j] * state_old[h, v, j]
      new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
      state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
      state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
      state_new[h, v, :] = g[h, v] * state_old[h, v, :] - state_remove[h, :] + state_update[h, :]
    """
    # Each program handles one token t for this seq_idx and loops over h, v
    for h in range(0, H):
        for v_i in range(0, V):
            # Load scalars g and beta for (h, v_i)
            g_val = tl.load(g_ptr + h * V + v_i).to(tl.float32)
            beta_val = tl.load(beta_ptr + h * V + v_i).to(tl.float32)

            # old_v[h, :] = sum_j k[t, h, j] * state_old[h, v_i, j]
            old_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_k = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)      # k[t, h, j]
                state_old_kv = tl.load(state_old_ptr + (h * V + v_i) * K + j).to(tl.float32)  # state_old[h, v_i, j]
                old_v[j] = k_k * state_old_kv

            # new_v[h, :] = beta[v_i] * v[t, v_i, :] + (1 - beta[v_i]) * old_v[h, :]
            new_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                v_k = tl.load(v_ptr + t * (V * K) + v_i * K + j).to(tl.float32)      # v[t, v_i, j]
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

            # Load state_old[h, v_i, :] and compute new_state
            state_old_row = tl.load(state_old_ptr + (h * V + v_i) * K + tl.arange(0, K)).to(tl.float32)
            state_new = g_val * state_old_row - state_remove + state_update

            # Store to new_state[h, v_i, :]
            for j in range(0, K):
                tl.store(new_state_ptr + (h * V + v_i) * K + j, state_new[j])


@triton.jit
def row_matmul_kernel(q_ptr, new_state_ptr, out_ptr, scale, T, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr, t, seq_idx):
    """
    Compute out[t] = scale * q[t] @ new_state[seq_idx].
    q_ptr points to q[t] row: shape [H, K] (in contiguous flattened form).
    new_state_ptr points to new_state[seq_idx] as [H, V, K] flattened.
    out_ptr is 1D array of length H*K.
    """
    # Flatten q[t] as 1D [H*K]
    q_flat = tl.load(q_ptr + t * (H * K) + tl.arange(0, H * K)).to(tl.float32)

    # For each (h, k), out[h*K + k] = sum_v new_state[h, v, k] * q[h, k]
    for h in range(0, H):
        for k in range(0, K):
            dot_sum = 0.0
            # Loop over V (8) and accumulate
            for v in range(0, V):
                state_val = tl.load(new_state_ptr + (h * V + v) * K + k).to(tl.float32)
                q_val = tl.load(q_ptr + t * (H * K) + h * K + k).to(tl.float32)
                dot_sum += state_val * q_val
            out_elem = scale * dot_sum
            tl.store(out_ptr + t * (H * K) + h * K + k, out_elem)


class ModelNew(nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure tensors are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if state is not None:
            state = state.contiguous()
        device = q.device
        dtype = torch.float32

        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        H = 4
        V = 8
        K = 128

        # Allocate arrays for g and beta (1D)
        g_flat = torch.empty((total_seq_len * V,), dtype=torch.float32, device=device)
        beta_flat = torch.empty((total_seq_len * V,), dtype=torch.float32, device=device)

        # Launch Triton kernels to compute g and beta
        grid_g = (total_seq_len,)
        compute_g_kernel[grid_g](a, dt_bias, A_log, g_flat, total_seq_len, V=V, num_warps=1)
        compute_beta_kernel[grid_g](b, beta_flat, total_seq_len, V=V, num_warps=1)

        # Output and new_state tensors
        output = torch.empty((total_seq_len, H, K), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((cu_seqlens.shape[0], H, V, K), dtype=torch.float32, device=device)

        # Process each sequence block
        for seq_idx in range(cu_seqlens.shape[0] - 1):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            if seq_start >= total_seq_len:
                continue
            # Initialize new_state for this block
            # If state is provided for this block, clone and transpose appropriately
            if state is not None and seq_idx < state.shape[0]:
                # state[seq_idx] has shape [H, V, K] (k-last at the end)
                state_clone = state[seq_idx].contiguous()  # [H, V, K]
                # Copy into new_state[seq_idx]
                # Flatten and copy row by row
                for h in range(H):
                    for v in range(V):
                        src_ptr = state_clone[h, v, :].contiguous()
                        dst_ptr = new_state[seq_idx, h, v, :].contiguous()
                        # Copy using torch operations (we're not doing torch matmul though)
                        new_state[seq_idx, h, v, :] = state_clone[h, v, :].clone()
            else:
                # Initialize to zeros
                new_state[seq_idx] = 0.0

            # Update state for each token in this block
            # We'll use the last block to compute output; the original code computes output using final state
            for t in range(seq_start, seq_end):
                # Launch update kernel for this token and block
                # new_state is updated in-place by the kernel
                # Note: Triton requires meta-args for H,V,K as constexpr
                grid_update = (1,)
                update_state_kernel[grid_update](
                    q, k, v, new_state[seq_idx], g_flat, beta_flat, new_state[seq_idx],
                    total_seq_len, H=H, V=V, K=K, t=t, seq_idx=seq_idx, num_warps=1
                )

        # Compute output per token using Triton row_matmul
        # For each t, compute out[t] = scale * q[t] @ new_state[-1]
        out_rows = torch.empty((total_seq_len, H * K), dtype=torch.float32, device=device)
        for t in range(total_seq_len):
            grid_out = (1,)
            row_matmul_kernel[grid_out](
                q, new_state[-1].contiguous().view(-1), out_rows[t], float(scale),
                total_seq_len, H=H, V=V, K=K, t=t, seq_idx=0, num_warps=1
            )
        output = out_rows.to(torch.bfloat16)  # [T, H*K] -> we need [T, H, K]
        # Reshape to [T, H, K]
        # out_rows has shape [T, H*K], convert to [T, H, K]
        output = out_rows.view(total_seq_len, H, K)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
