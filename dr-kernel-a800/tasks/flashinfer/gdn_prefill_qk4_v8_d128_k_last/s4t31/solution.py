import torch
import triton
import triton.language as tl


@triton.jit
def softplus_torch_like(x_ptr, out_ptr, N):
    # Elementwise softplus for x of length N: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    offs = tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    absx = tl.abs(x)
    soft = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-absx))
    tl.store(out_ptr + offs, soft, mask=mask)


@triton.jit
def sigmoid_torch_like(x_ptr, out_ptr, N):
    # Elementwise sigmoid: y = 1 / (1 + exp(-x))
    offs = tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def exp_vec(x_ptr, out_ptr, N):
    # Elementwise exp over a vector of length N
    offs = tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def gate_torch_like(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, N_COLS, BLOCK_N: tl.constexpr):
    # Compute g_expanded of length N_COLS = L * 8
    # Mapping: for each head hh in 0..7, columns [2*hh, 2*hh+1] map to A_log[hh]
    # a_ptr: [L*8], dt_bias_ptr: [32], A_log_ptr: [8], g_ptr: [L*8]
    # We process 1024 elements per launch, mask for actual N_COLS
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N_COLS
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    # Load corresponding A_log based on head index
    hh = offs // 2  # head index
    # A_log index is hh (hh in 0..7), but our column only maps to 0..7
    A_log_val = tl.load(A_log_ptr + hh, mask=mask, other=0.0)
    x = a + tl.load(dt_bias_ptr + hh, mask=mask, other=0.0)  # dt_bias has length 8
    # softplus(x) = max(x,0) + log(1 + exp(-|x|))
    absx = tl.abs(x)
    soft = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-absx))
    # g = exp(-exp(A_log) * softplus(x))
    g = tl.exp(-tl.exp(A_log_val) * soft)
    tl.store(g_ptr + offs, g, mask=mask)


@triton.jit
def gemv_kernel(q_ptr, state_ptr, out_ptr, K, V, scale, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    # q_ptr: [K], state_ptr: [V, K], out_ptr: [V]
    # Compute out[j] = sum_i q[i] * state[j, i], j in 0..V-1, i in 0..K-1
    offs_j = tl.arange(0, BLOCK_V)
    for i0 in range(0, K, BLOCK_K):
        offs_i = i0 + tl.arange(0, BLOCK_K)
        # Load q tile
        q_tile = tl.load(q_ptr + offs_i, mask=offs_i < K, other=0.0)  # [BLOCK_K]
        # Accumulate over i in this tile
        acc = tl.zeros([BLOCK_V], dtype=tl.float32)
        # Loop over this tile dimension (compile-time loop)
        for kk in range(BLOCK_K):
            i = i0 + kk
            # Load state row j across all columns in tile
            # We need to compute out_j += q[i] * state[j, i]
            # For each j, state[j, i] is accessed at (j*stride_v + i)
            # We'll do an inner vector load for all j
            # Note: state is [V, K] row-major; pointer state_ptr + j*stride_v + i
            # Build pointers for all j in tile
            # We use a vectorized approach: create 2D pointers for each j in offs_j and each i in offs_i
            # but Triton doesn't support 2D load directly in this context; we compute acc via loop:
            # Instead, loop over j in tile and update acc
            # However, Triton allows only vectorized operations; we'll restructure: for each kk (i), we compute a vector dot by iterating j manually.
            # This design is suboptimal in Triton for a 2D matmul, but here V=K=128 so we can use fixed BLOCK=128 to avoid inner loop.
            # To keep it simple and correct, we assume BLOCK_K == BLOCK_V == 128 and unroll.
            pass  # Placeholder; see below for corrected implementation


# Corrected GEMV kernel for V=K=128: process entire vector at once
@triton.jit
def gemv_kernel_v128(q_ptr, state_ptr, out_ptr, scale):
    # Assumes K=128, V=128
    q = tl.load(q_ptr)  # [128]
    V = 128
    acc = tl.zeros([128], dtype=tl.float32)
    # For each i in 0..127, acc += q[i] * state[i, :]
    for i in range(128):
        q_i = q[i]
        # state[i, :] is at offset i*V + 0..V-1 in row-major [V, K]
        # Load row i across all columns
        # Note: Triton needs pointers; we'll treat state_ptr as 1D base and use index arithmetic
        # Load vector for row i across columns
        col = tl.arange(0, 128)
        # Pointer to state row i: state_ptr + i * V + col
        s_row = tl.load(state_ptr + i * V + col)
        acc += q_i * s_row
    # Scale
    acc *= scale
    # Store
    tl.store(out_ptr + tl.arange(0, 128), acc)


@triton.jit
def update_state_kernel(k_ptr, old_state_ptr, v_ptr, beta_ptr, g_ptr, new_state_ptr, K, V, scale, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    # For each head h in 0..7: update new_state[h] using k[t,h], old_state[h], v[t,h], beta[h], g[h]
    # We implement for a single head; caller can call 8 times. Triton supports loops with compile-time constants.
    # This kernel assumes V=K=128 and uses tiles.
    # Compute old_v = k @ old_state (GEMV)
    offs_j = tl.arange(0, BLOCK_V)
    offs_k = tl.arange(0, BLOCK_K)
    old_v = tl.zeros([128], dtype=tl.float32)
    for i0 in range(0, K, BLOCK_K):
        i = i0 + offs_k
        mask_i = i < K
        q_tile = tl.load(k_ptr + i, mask=mask_i, other=0.0)  # [BLOCK_K]
        acc = tl.zeros([BLOCK_V], dtype=tl.float32)
        for kk in range(BLOCK_K):
            ii = i0 + kk
            s_row = tl.load(old_state_ptr + ii * V + offs_j, mask=(ii < K) & (offs_j < V), other=0.0)
            acc += q_tile[kk] * s_row
        old_v += acc
    # Compute new_v = beta * v + (1 - beta) * old_v
    beta_h = tl.load(beta_ptr)
    v_row = tl.load(v_ptr)  # [128]
    new_v = beta_h * v_row + (1.0 - beta_h) * old_v
    # Compute delta = k @ (new_v - old_v) using GEMV on (new_v - old_v)
    diff = new_v - old_v
    delta = tl.zeros([128], dtype=tl.float32)
    for i0 in range(0, K, BLOCK_K):
        i = i0 + offs_k
        mask_i = i < K
        q_tile = tl.load(k_ptr + i, mask=mask_i, other=0.0)  # [BLOCK_K]
        acc = tl.zeros([BLOCK_V], dtype=tl.float32)
        for kk in range(BLOCK_K):
            ii = i0 + kk
            # diff[ii] is scalar
            diff_val = tl.load(diff + ii, mask=(ii < K), other=0.0)  # but diff is [128], so scalar load is fine
            s_row = tl.load(old_state_ptr + ii * V + offs_j, mask=(ii < K) & (offs_j < V), other=0.0)
            acc += q_tile[kk] * s_row * diff_val
        delta += acc
    # Update new_state: g_h * old_state - k @ old_v + k @ (new_v - old_v) = g_h * old_state + delta
    g_h = tl.load(g_ptr)  # single scalar per head
    for i0 in range(0, K, BLOCK_K):
        i = i0 + offs_k
        mask_i = i < K
        q_tile = tl.load(k_ptr + i, mask=mask_i, other=0.0)  # [BLOCK_K]
        acc = tl.zeros([BLOCK_V], dtype=tl.float32)
        for kk in range(BLOCK_K):
            ii = i0 + kk
            s_row = tl.load(old_state_ptr + ii * V + offs_j, mask=(ii < K) & (offs_j < V), other=0.0)
            acc += q_tile[kk] * s_row
        # Write new_state row ii: new_state[ii] = g_h * old_state[ii] + delta[ii]
        old_row = tl.load(old_state_ptr + ii * V + offs_j, mask=(ii < K) & (offs_j < V), other=0.0)
        new_row = g_h * old_row + delta + 0.0  # placeholder; Triton will handle broadcasting
        tl.store(new_state_ptr + ii * V + offs_j, new_row, mask=(ii < K) & (offs_j < V))


def _launch_for_all_heads(t, k_ptr_t, old_state, v_ptr_t, beta_t, g_t, new_state, K, V, scale):
    # Run update for all 8 heads: hh in 0..7, A_log[hh]
    for hh in range(8):
        # Prepare pointers
        # k_ptr_t: 1D [K], old_state: [V,V], v_ptr_t: 1D [V]
        # beta_t, g_t: scalars
        # Launch update kernel for head hh
        # We need to index A_log[hh] externally
        # Note: Triton kernel expects tensors; we pass flattened pointers and shapes.
        # We assume BLOCK sizes 128.
        # Launch update_state_kernel for this (t, hh)
        # Triton kernel uses masks, we provide K=128, V=128
        # We'll create temporary pointers for beta and g (scalars), but Triton expects 1D; pass 1-element tensors
        beta_buf = torch.tensor([beta_t], dtype=torch.float32, device=old_state.device)
        g_buf = torch.tensor([g_t], dtype=torch.float32, device=old_state.device)
        grid = (1,)
        update_state_kernel[grid](
            k_ptr_t, old_state, v_ptr_t, beta_buf, g_buf, new_state, K, V, scale,
            BLOCK_K=128, BLOCK_V=128
        )


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure inputs are on CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be on CUDA for Triton execution."
        L = q.shape[0]
        # Initialize outputs
        output = torch.empty((L, 8, 128), dtype=torch.bfloat16, device=q.device)
        new_state = torch.empty((state.shape[0], 8, 128, 128), dtype=torch.float32, device=q.device)

        # Compute g_expanded (shape [L, 8]) using Triton
        # a has shape [L, 32], dt_bias has shape [8], A_log has shape [8]
        # We need to map a[:, h] to head hh via repeat_interleave(2). The Triton kernel gate_torch_like computes g_expanded directly.
        # Prepare flattened a_expanded: [L*8] where a_expanded[:, 2h] = a[:, h], a_expanded[:, 2h+1] = a[:, h]
        a_expanded = a.repeat(1, 4)  # [L, 8]
        g_expanded = torch.empty((L, 8), dtype=torch.float32, device=q.device)
        # Launch gate_torch_like: a_expanded [L*8], dt_bias [8], A_log [8]
        # We need dt_bias of length 8 expanded to 32? The original logic uses dt_bias per head (8), not per column 32. The code snippet did softplus(a + dt_bias) but dt_bias is length 8. To match reference, we use dt_bias per head: dt_bias_expanded = dt_bias[hh] for each head hh. In the forward, dt_bias is [8], so we can use it as is.
        # Launch gate_torch_like on a_expanded.view(-1) and dt_bias
        N_COLS = L * 8
        grid_g = (triton.cdiv(N_COLS, 1024),)
        gate_torch_like[grid_g](a_expanded.view(-1), dt_bias, A_log, g_expanded.view(-1), N_COLS, BLOCK_N=1024)

        # Compute beta using Triton sigmoid for b of shape [L, 32], then map to 8 heads via repeat_interleave(2)
        b_expanded = b.repeat(1, 4)  # [L, 8]
        beta_expanded = torch.empty((L, 8), dtype=torch.float32, device=q.device)
        grid_s = (triton.cdiv(L * 8, 1024),)
        sigmoid_torch_like[grid_s](b_expanded.view(-1), beta_expanded.view(-1), L * 8)

        # Now process sequences using cu_seqlens
        num_seqs = cu_seqlens.numel() - 1
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start

            # Initialize old_state for each head with state[seq_idx] or zeros if None
            # state is [num_seqs, 8, 128, 128]; index seq_idx
            if state[seq_idx] is not None:
                old_state = state[seq_idx].to(torch.float32).contiguous()
            else:
                old_state = torch.zeros((8, 128, 128), dtype=torch.float32, device=q.device).transpose(-1, -2).contiguous()  # [H, V, K]

            # For each t in segment
            for i in range(seq_len):
                t = seq_start + i
                # Compute outputs for each head
                # We need q_exp[k,8,128] and k_exp[k,8,128] from q[k,4,128], k[k,4,128] by repeat_interleave(2, dim=1)
                q_exp_h = q[t].repeat_interleave(2, dim=0).float().contiguous()  # [8, 128]
                k_exp_h = k[t].repeat_interleave(2, dim=0).float().contiguous()  # [8, 128]
                v_h = v[t].float().contiguous()  # [8, 128] (but we need per-head vector [128])

                # For Triton GEMV, we need q_exp_h[:,0] -> q_vec for head 0..7. But original GEMV uses q_exp[t,h], which is [128]. Triton kernel accepts a 1D vector. We'll compute output[t, h, :] for each h:
                for h in range(8):
                    # Prepare pointers
                    q_vec = q_exp_h[h].float().contiguous()  # [128]
                    state_h = old_state[h].float().contiguous()  # [128, 128], row-wise as [V, K]
                    out_vec = torch.empty((128,), dtype=torch.float32, device=q.device)
                    # Launch gemv kernel for this (t, h)
                    grid_gmv = (1,)
                    gemv_kernel_v128[grid_gmv](q_vec, state_h.view(-1), out_vec, 1.0)
                    # Store output as bfloat16
                    output[t, h] = out_vec.to(torch.bfloat16)

                # Update new_state for each head using Triton update_state_kernel
                # First, compute g_h and beta_h for each head h
                g_h = g_expanded[t, h]  # scalar per head h
                beta_h = beta_expanded[t, h]
                # Select k[t,h], v[t,h] vectors
                k_vec = k_exp_h[h].float().contiguous()  # [128]
                v_vec = v_h[h].float().contiguous()  # [128]
                # Launch update for this (t) across 8 heads in _launch_for_all_heads helper, but we need to pass new_state properly.
                # The helper requires 8 separate launches; we will inline per-head logic using Triton update_state_kernel:
                for hh in range(8):
                    # new_state[seq_idx, hh] is a 128x128 matrix; we need a pointer to its row-major storage. Compute it as a contiguous [128,128] buffer.
                    new_state_seq = new_state[seq_idx]  # [8,128,128]
                    # Launch update_state_kernel for (t, hh)
                    grid_u = (1,)
                    update_state_kernel[grid_u](
                        k_vec, old_state[hh], v_vec, beta_expanded[t, hh], g_expanded[t, hh], new_state_seq[hh], 128, 128, 1.0,
                        BLOCK_K=128, BLOCK_V=128
                    )
                    # Update old_state for next iteration
                    old_state[hh] = new_state_seq[seq_idx, hh]

        return output, new_state


def run(*args):
    return ModelNew()(*args)
