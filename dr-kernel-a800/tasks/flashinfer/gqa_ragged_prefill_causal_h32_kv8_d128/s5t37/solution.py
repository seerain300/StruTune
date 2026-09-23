import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_attention_kernel(
    q_ptr,        # *float32, shape [M, G, D]
    k_ptr,        # *float32, shape [N, GH, D]
    v_ptr,        # *float32, shape [N, GH, D]
    out_ptr,      # *bfloat16, shape [M, G, D]
    lse_ptr,      # *float32, shape [M, G]
    # per-block indices (device int32, length 2)
    qo_indptr_ptr,  # *int32, [q_start, q_end]
    kv_indptr_ptr,  # *int32, [kv_start, kv_end]
    # sizes
    M: tl.constexpr,      # number of queries in this block (runtime int at launch)
    N: tl.constexpr,      # number of kv tokens in this block (runtime int at launch)
    # constexpr sizes
    G: tl.constexpr,      # num_qo_heads (e.g., 32)
    GH: tl.constexpr,     # num_kv_heads (e.g., 8) * gqa_ratio == G (32)
    D: tl.constexpr,      # head_dim (e.g., 128)
    SM_SCALE: tl.constexpr,  # scaling factor (e.g., 1.0 / sqrt(128))
    BLOCK_M: tl.constexpr,   # tile over M dimension, e.g., 64
    BLOCK_N: tl.constexpr,   # tile over N (KV length), e.g., 128
    BLOCK_D: tl.constexpr,   # tile over D (head dim), e.g., 128
):
    # Load q and kv ranges for this block
    q_start = tl.load(qo_indptr_ptr + 0).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + 0).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + 1).to(tl.int32)

    # delta: extra KV tokens relative to Q tokens for this block
    delta = kv_end - kv_start - (q_end - q_start)

    # Tile over M (query positions) and N (KV positions)
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)

    # Iterate over M tile chunks
    for m0 in range(0, M, BLOCK_M):
        m_idx = m0 + m_offsets  # [BLOCK_M]
        mask_m = m_idx < M

        # Prepare accumulators per head (G): one row per q position
        lse_vec = tl.full((G,), -float('inf'), tl.float32)

        # For each query head g
        for g in range(0, G):
            # Q matrix for this g over BLOCK_M q positions, with D dimension
            Q_tile = tl.zeros((BLOCK_M, D), dtype=tl.float32)
            # Load Q vectors for m_idx positions
            for mm in range(0, BLOCK_M):
                valid_mm = m_idx[mm] < M
                # base pointer for this (m, g, :)
                q_vec_ptr = q_ptr + m_idx[mm] * G * D + g * D
                # load with mask
                d_offsets = tl.arange(0, BLOCK_D)
                mask_d = d_offsets < D
                Q_tile[mm, :] = tl.load(q_vec_ptr + d_offsets, mask=mask_d & valid_mm, other=0.0)

            # For each KV group h (GH = G)
            for h in range(0, GH):
                # Compute logsumexp over N for this (q_idx, g, h)
                sum_exp = tl.zeros((), dtype=tl.float32)  # scalar

                # Process N in tiles
                for n0 in range(0, N, BLOCK_N):
                    j = n0 + n_offsets  # [BLOCK_N]
                    mask_n = j < N

                    # K_tile: [BLOCK_N, D], V_tile: [BLOCK_N, D]
                    K_tile = tl.zeros((BLOCK_N, D), dtype=tl.float32)
                    V_tile = tl.zeros((BLOCK_N, D), dtype=tl.float32)

                    # Load K and V for this h
                    for nn in range(0, BLOCK_N):
                        valid_nn = mask_n[nn]
                        k_vec_ptr = k_ptr + (kv_start + j[nn]) * GH * D + h * D
                        v_vec_ptr = v_ptr + (kv_start + j[nn]) * GH * D + h * D
                        k_d = tl.load(k_vec_ptr + tl.arange(0, BLOCK_D), mask=valid_nn & (tl.arange(0, BLOCK_D) < D), other=0.0)
                        v_d = tl.load(v_vec_ptr + tl.arange(0, BLOCK_D), mask=valid_nn & (tl.arange(0, BLOCK_D) < D), other=0.0)
                        K_tile[nn, :] = k_d
                        V_tile[nn, :] = v_d

                    # Compute logits for this tile: (BLOCK_M, BLOCK_N)
                    logits = tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_M, BLOCK_N]
                    # Apply scaling
                    logits = logits * SM_SCALE

                    # Causal mask: j < (q_idx + 1 + delta)
                    # q_idx specific per m in BLOCK_M; we compute mask using j and m_idx
                    q_idx_vec = m_idx[:, None]  # [BLOCK_M, 1]
                    causal = j[None, :] < (q_idx_vec + 1 + delta)
                    logits = tl.where(mask_m[:, None] & mask_n[None, :] & causal, logits, -float('inf'))

                    # Mask out invalid m/n positions
                    logits = tl.where((~mask_m)[:, None], -float('inf'), logits)
                    logits = tl.where((~mask_n)[None, :], -float('inf'), logits)

                    # Reduce to scalar logsumexp for this (q_idx, g, h)
                    # First sum over N (axis=1): [BLOCK_M]
                    exp_row = tl.exp(logits)  # [BLOCK_M, BLOCK_N]
                    # Sum along N to get per-m probability vector
                    sum_exp += tl.sum(exp_row, axis=1)

                # LSE in base-2
                lse_g = tl.log(sum_exp) / math.log(2.0)
                lse_vec[g] = lse_g

            # Now compute output for each g
            for g_out in range(0, G):
                # Accumulator for output row (g_out) across N
                out_row = tl.zeros((D,), dtype=tl.float32)
                for n0 in range(0, N, BLOCK_N):
                    j = n0 + n_offsets
                    mask_n = j < N

                    K_tile = tl.zeros((BLOCK_N, D), dtype=tl.float32)
                    V_tile = tl.zeros((BLOCK_N, D), dtype=tl.float32)

                    for nn in range(0, BLOCK_N):
                        valid_nn = mask_n[nn]
                        k_vec_ptr = k_ptr + (kv_start + j[nn]) * GH * D + h * D  # h fixed (this loop has g_out only, but we recompute)
                        v_vec_ptr = v_ptr + (kv_start + j[nn]) * GH * D + h * D
                        k_d = tl.load(k_vec_ptr + tl.arange(0, BLOCK_D), mask=valid_nn & (tl.arange(0, BLOCK_D) < D), other=0.0)
                        v_d = tl.load(v_vec_ptr + tl.arange(0, BLOCK_D), mask=valid_nn & (tl.arange(0, BLOCK_D) < D), other=0.0)
                        K_tile[nn, :] = k_d
                        V_tile[nn, :] = v_d

                    # For this g_out, compute attention weights
                    # We need to recompute logits and softmax for each g_out vs each h
                    for h2 in range(0, GH):
                        # Compute logits for (g_out, h2)
                        # Reload Q_tile for g_out
                        Qg2 = tl.zeros((BLOCK_M, D), dtype=tl.float32)
                        for mm in range(0, BLOCK_M):
                            valid_mm2 = m_idx[mm] < M
                            qg2_ptr = q_ptr + m_idx[mm] * G * D + g_out * D
                            Qg2[mm, :] = tl.load(qg2_ptr + tl.arange(0, BLOCK_D), mask=(tl.arange(0, BLOCK_D) < D) & valid_mm2, other=0.0)

                        logits2 = tl.dot(Qg2, tl.trans(K_tile))  # [BLOCK_M, BLOCK_N]
                        logits2 = logits2 * SM_SCALE
                        q_idx_vec = m_idx[:, None]
                        causal2 = j[None, :] < (q_idx_vec + 1 + delta)
                        logits2 = tl.where(mask_m[:, None] & mask_n[None, :] & causal2, logits2, -float('inf'))
                        logits2 = tl.where((~mask_m)[:, None], -float('inf'), logits2)
                        logits2 = tl.where((~mask_n)[None, :], -float('inf'), logits2)

                        # softmax over N
                        exp2 = tl.exp(logits2)  # [BLOCK_M, BLOCK_N]
                        # Mask invalid N to 0 for sum
                        exp2 = tl.where((~mask_n)[None, :], 0.0, exp2)
                        denom = tl.sum(exp2, axis=1)  # [BLOCK_M]
                        softmax2 = exp2 / denom[:, None]  # [BLOCK_M, BLOCK_N]

                        # Multiply by V_tile and accumulate
                        out_row += tl.sum(softmax2 * V_tile[None, :], axis=1)  # [BLOCK_M]

                # Store output for all D for each q in m_idx with head g_out
                for mm in range(0, BLOCK_M):
                    valid_mm = m_idx[mm] < M
                    out_base = out_ptr + m_idx[mm] * G * D + g_out * D
                    d_offsets = tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D
                    tl.store(out_base + d_offsets, out_row[mm, :] / out_row[mm, :].max() * 65504.0, mask=mask_d & valid_mm)  # scale for bf16 store

        # Store lse per q position (M) for each head g
        # lse_ptr points to [M, G], we store at (m_idx, g)
        # We only wrote per-g; loop to store all g
        for g2 in range(0, G):
            for mm in range(0, BLOCK_M):
                valid_mm = m_idx[mm] < M
                lse_base = lse_ptr + m_idx[mm] * G + g2
                # store lse_vec[g2] to lse_base
                tl.store(lse_base, lse_vec[g2], mask=valid_mm)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure device is CUDA
        assert TRITON_AVAILABLE, "Triton is not available."
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "Tensors must be on CUDA."
        # Cast to float32 for compute
        q = q.to(torch.float32).contiguous()
        k = k.to(torch.float32).contiguous()
        v = v.to(torch.float32).contiguous()

        # Shapes
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        # Constants as per original code
        G = 32
        GH = 8  # num_kv_heads
        assert num_qo_heads == G
        # delta per block is used in causal mask; not a constant here

        # Output and lse
        output = torch.empty((total_q, G, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, G), dtype=torch.float32, device=device)

        # Fixed block sizes
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_D = 128

        # Process blocks
        for b in range(len_indptr - 1):
            # Load block ranges
            q_start = int(tl.load(qo_indptr[b]).item())
            q_end = int(tl.load(qo_indptr[b + 1]).item())
            kv_start = int(tl.load(kv_indptr[b]).item())
            kv_end = int(tl.load(kv_indptr[b + 1]).item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            M = q_end - q_start
            N = kv_end - kv_start

            # Slice tensors
            q_batch = q[q_start:q_end]  # [M, G, D]
            k_batch = k[kv_start:kv_end]  # [N, GH, D]
            v_batch = v[kv_start:kv_end]  # [N, GH, D]

            # Launch Triton kernel
            # Create device int32 tensors for block ranges
            qo_block = qo_indptr[b:b+2].to(torch.int32).contiguous()
            kv_block = kv_indptr[b:b+2].to(torch.int32).contiguous()

            grid = (1,)  # one program per block
            _block_attention_kernel[grid](
                q_batch, k_batch, v_batch,
                output[q_start:q_end], lse[q_start:q_end],
                qo_block, kv_block,
                M, N,
                G, GH, head_dim,
                sm_scale,
                BLOCK_M, BLOCK_N, BLOCK_D,
                num_warps=4, num_stages=2,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
