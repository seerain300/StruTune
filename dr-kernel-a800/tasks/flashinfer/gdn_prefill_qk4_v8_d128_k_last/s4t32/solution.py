import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F


@triton.jit
def softplus_torch_like(x_ptr, out_ptr, N):
    # Elementwise softplus over 1D vector x_ptr of length N
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    offs = tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    absx = tl.abs(x)
    soft = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-absx))
    tl.store(out_ptr + offs, soft, mask=mask)


@triton.jit
def sigmoid_torch_like(x_ptr, out_ptr, N):
    # Elementwise sigmoid over 1D vector
    offs = tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, sig, mask=mask)


@triton.jit
def exp_vec(x_ptr, out_ptr, N):
    # Elementwise exp over 1D vector
    offs = tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def gate_torch_like(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, N_COLS, BLOCK_N: tl.constexpr):
    # Compute g_expanded of length N_COLS = L * 8
    # Mapping: hh = offs // 2; use A_log[hh] for each head index
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N_COLS
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    hh = offs // 2  # head index in 0..7
    A_log_val = tl.load(A_log_ptr + hh, mask=mask, other=0.0)
    x = a + tl.load(dt_bias_ptr + hh, mask=mask, other=0.0)
    absx = tl.abs(x)
    soft = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-absx))
    g = tl.exp(-tl.exp(A_log_val) * soft)
    tl.store(g_ptr + offs, g, mask=mask)


@triton.jit
def gemv_kernel_v128(q_ptr, state_ptr, out_ptr, scale):
    # Compute out[j] = sum_i q[i] * state[j, i], j in 0..127, i in 0..127
    # q_ptr: [128], state_ptr: [16384] (row-major [128,128])
    q = tl.load(q_ptr)  # [128]
    V = 128
    acc = tl.zeros([V], dtype=tl.float32)
    # Accumulate over K=128
    for i in range(128):
        q_i = q[i]
        col = tl.arange(0, V)  # each j
        # state[j, i] is at offset i*V + j
        s_j = tl.load(state_ptr + i * V + col)  # [128]
        acc += q_i * s_j
    acc *= scale
    tl.store(out_ptr + tl.arange(0, V), acc)


@triton.jit
def update_state_kernel(k_ptr, old_state_ptr, v_ptr, beta, g, new_state_ptr, K, V, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    # K=128, V=128; update new_state using k, old_state, v, beta (scalar), g (scalar)
    # Compute old_v = k @ old_state (GEMV over K)
    old_v = tl.zeros([V], dtype=tl.float32)
    for i0 in range(0, K, BLOCK_K):
        offs_k = i0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_tile = tl.load(k_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]
        acc = tl.zeros([V], dtype=tl.float32)
        for kk in range(BLOCK_K):
            ii = i0 + kk
            s_row = tl.load(old_state_ptr + ii * V + tl.arange(0, V), mask=(ii < K) & (tl.arange(0, V) < V), other=0.0)
            acc += k_tile[kk] * s_row
        old_v += acc

    # new_v = beta * v + (1 - beta) * old_v
    v_vec = tl.load(v_ptr + tl.arange(0, V))
    new_v = beta * v_vec + (1.0 - beta) * old_v

    # delta = k @ new_v
    delta = tl.zeros((), dtype=tl.float32)  # scalar
    for i0 in range(0, K, BLOCK_K):
        offs_k = i0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_tile = tl.load(k_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for kk in range(BLOCK_K):
            ii = i0 + kk
            s_row = tl.load(old_state_ptr + ii * V + tl.arange(0, V), mask=(ii < K) & (tl.arange(0, V) < V), other=0.0)
            acc += k_tile[kk] * s_row[kk]
        delta += tl.sum(acc)

    # Update new_state in-place: new_state += g * old_state
    # We don't have pointer-to-pointer update in Triton; we store back to new_state_ptr row by row.
    for i0 in range(0, V, BLOCK_V):
        offs_v = i0 + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        row = tl.load(old_state_ptr + tl.arange(0, V) * V + offs_v, mask=mask_v, other=0.0)
        row = row * g + delta
        tl.store(new_state_ptr + tl.arange(0, V) * V + offs_v, row, mask=mask_v)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized forward. All heavy math is done via Triton kernels.
        Inputs:
          q: [L, 4, 128] bfloat16
          k: [L, 4, 128] bfloat16
          v: [L, 8, 128] bfloat16
          state: [num_seqs, 8, 128, 128] float32
          A_log: [8] float32
          a: [L, 32] bfloat16
          dt_bias: [8] float32
          b: [L, 32] bfloat16
          cu_seqlens: [num_seqs+1] int64
          scale: float
        Returns:
          output: [L, 8, 128] bfloat16
          new_state: [num_seqs, 8, 128, 128] float32
        """
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, "q,k,v must be 3D"
        assert q.shape[1] == 4 and k.shape[1] == 4 and v.shape[1] == 8, "Heads mismatch: q=4, k=4, v=8"
        assert q.shape[2] == 128 and k.shape[2] == 128 and v.shape[2] == 128, "head_size must be 128"

        L = q.shape[0]
        num_seqs = cu_seqlens.shape[0] - 1
        device = q.device

        # Ensure dtypes and contiguity
        a = a.contiguous().to(torch.float32)  # [L, 32]
        dt_bias = dt_bias.contiguous().to(torch.float32)  # [8]
        b = b.contiguous().to(torch.float32)  # [L, 32]
        A_log = A_log.contiguous().to(torch.float32)  # [8]
        state = state.contiguous().to(torch.float32)  # [num_seqs, 8, 128, 128]

        # Compute g_expanded and beta using Triton kernels
        # g is per head: length L*8
        N_HEADS = 8
        N_COLS = L * N_HEADS
        g_expanded = torch.empty((N_COLS,), dtype=torch.float32, device=device)
        gate_torch_like[(1,)](a.view(-1), dt_bias, A_log, g_expanded, N_COLS, BLOCK_N=N_COLS)

        # beta: per head, length L*8
        beta_expanded = torch.empty((N_COLS,), dtype=torch.float32, device=device)
        sigmoid_torch_like[(1,)](b.view(-1), beta_expanded, N_COLS, BLOCK_N=N_COLS)

        # Initialize outputs and new state
        output = torch.empty((L, N_HEADS, 128), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((num_seqs, N_HEADS, 128, 128), dtype=torch.float32, device=device)

        # For each sequence segment
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Initialize new_state for this seq to zeros
            new_state[seq_idx] = 0.0  # placeholder; we'll update per t

            # Loop over time steps in the segment
            for i in range(seq_len):
                t = seq_start + i
                # Compute expanded heads q_exp/k_exp from original 4 heads
                # Mapping: h -> 2h, 2h+1 share q[k] from original h
                q_t = q[t]  # [4,128]
                k_t = k[t]  # [4,128]
                v_t = v[t]  # [8,128]

                # Build q_exp/k_exp for 8 heads
                # q_exp[k,8,128] where q_exp[k, 2h] = q[k,h], q_exp[k, 2h+1] = q[k,h]
                # Implement by selecting rows
                q_h0 = q_t[0]  # [128]
                q_h1 = q_t[1]
                q_h2 = q_t[2]
                q_h3 = q_t[3]
                k_h0 = k_t[0]
                k_h1 = k_t[1]
                k_h2 = k_t[2]
                k_h3 = k_t[3]

                # Load g_expanded[t, h] and beta_expanded[t, h] for h in 0..7
                # Index mapping: idx = t * 8 + h
                idx0 = t * 8 + 0
                idx1 = t * 8 + 1
                idx2 = t * 8 + 2
                idx3 = t * 8 + 3
                idx4 = t * 8 + 4
                idx5 = t * 8 + 5
                idx6 = t * 8 + 6
                idx7 = t * 8 + 7
                g0 = g_expanded[idx0]
                g1 = g_expanded[idx1]
                g2 = g_expanded[idx2]
                g3 = g_expanded[idx3]
                g4 = g_expanded[idx4]
                g5 = g_expanded[idx5]
                g6 = g_expanded[idx6]
                g7 = g_expanded[idx7]
                b0 = beta_expanded[idx0]
                b1 = beta_expanded[idx1]
                b2 = beta_expanded[idx2]
                b3 = beta_expanded[idx3]
                b4 = beta_expanded[idx4]
                b5 = beta_expanded[idx5]
                b6 = beta_expanded[idx6]
                b7 = beta_expanded[idx7]

                # Process each head h
                for h in range(8):
                    idx = t * 8 + h
                    g = g_expanded[idx]
                    beta = beta_expanded[idx]

                    # Prepare q_exp vector for this head
                    if h == 0 or h == 1:
                        q_vec = q_h0
                    elif h == 2 or h == 3:
                        q_vec = q_h1
                    elif h == 4 or h == 5:
                        q_vec = q_h2
                    else:
                        q_vec = q_h3  # h == 6 or 7

                    # Initial state_old for this head in this segment is previous new_state (first t uses zeros)
                    state_old = new_state[seq_idx, h]  # [128,128]
                    state_new = torch.empty_like(state_old)

                    # Load k_exp row for this head
                    if h == 0 or h == 1:
                        k_vec = k_h0
                    elif h == 2 or h == 3:
                        k_vec = k_h1
                    elif h == 4 or h == 5:
                        k_vec = k_h2
                    else:
                        k_vec = k_h3  # h == 6 or 7

                    # Compute old_v = k_vec @ state_old (GEMV), Triton kernel
                    old_state_flat = state_old.reshape(128 * 128)  # [16384]
                    out_old = torch.empty((128,), dtype=torch.float32, device=device)
                    # Pass k_vec as flattened length 128
                    k_flat = k_vec.reshape(128)
                    gemv_kernel_v128[(1,)](k_flat, old_state_flat, out_old, 1.0)  # scale=1.0 for update

                    # Load v_vec for this head
                    v_vec = v_t[h]  # [128]

                    # Compute new_v = beta * v + (1 - beta) * old_v (elementwise), then delta = k_vec @ new_v (GEMV), Triton kernel
                    new_v_vec = beta * v_vec + (1.0 - beta) * out_old
                    # update delta: k_vec @ new_v_vec
                    new_v_flat = new_v_vec.reshape(128)
                    delta_val = torch.empty((), dtype=torch.float32, device=device)
                    gemv_kernel_v128[(1,)](k_flat, new_v_flat, delta_val, 1.0)  # scale=1.0

                    # Update state_new = g * state_old + delta
                    # We need to write updated rows into state_new
                    # Implement via Triton update_state_kernel
                    # Prepare pointers
                    old_state_ptr = state_old.reshape(128 * 128)  # [16384]
                    new_state_row_ptr = state_new.reshape(128 * 128)  # [16384]
                    # We need to store back updated rows; Triton kernel will compute and store
                    update_state_kernel[(1,)](k_flat, old_state_ptr, new_v_flat, beta, g, new_state_row_ptr, 128, 128, BLOCK_K=128, BLOCK_V=128)

                    # Compute output = scale * q_vec @ state_new (GEMV), Triton kernel
                    q_vec_flat = q_vec.reshape(128)
                    state_new_flat = state_new.reshape(128 * 128)
                    out_vec = torch.empty((128,), dtype=torch.float32, device=device)
                    # scale: original scale or 1.0; use scale argument
                    gemv_kernel_v128[(1,)](q_vec_flat, state_new_flat, out_vec, float(scale))

                    # Store output[t, h, :]
                    output[t, h, :] = out_vec.to(torch.bfloat16)

                    # Save updated state_new back to new_state
                    new_state[seq_idx, h] = state_new

        return output, new_state


def run(*args):
    return ModelNew()(*args)
