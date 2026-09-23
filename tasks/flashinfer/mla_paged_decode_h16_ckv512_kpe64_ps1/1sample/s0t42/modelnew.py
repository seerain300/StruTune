import math
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_row_Kc_kernel(
    qn_ptr,        # *fp32, [N]
    Kc_ptr,        # *fp32, [M_total * N] flattened
    out_ptr,       # *fp32, [N]
    N: tl.constexpr,           # head_dim_ckv (512)
    M_total,                   # int32, number of tokens
    stride_kc_m: tl.constexpr,  # N
    BLOCK_N: tl.constexpr
):
    # Compute qn @ Kc for a single token row: out[i] = sum_k qn[k] * Kc[i*N + k]
    for n0 in tl.static_range(0, N, BLOCK_N):
        n_offsets = n0 + tl.arange(0, BLOCK_N)
        # Load qn segment
        qn_seg = tl.load(qn_ptr + n_offsets, mask=n_offsets < N, other=0.0)
        # Accumulate dot products
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k in tl.static_range(0, N):
            # Kc column k across M_total rows: [i*N + k] for i in 0..M_total-1
            # We need to compute Kc[:, k] as a vector of length M_total and then dot with qn_seg.
            # However, Triton doesn't support directly indexing a 2D with a vector index here; instead,
            # we compute per token i: Kc[i*N + k] and accumulate into a vector.
            # Simpler approach: since we iterate over tokens in a higher-level fused kernel, this kernel
            # is only for computing qn @ Kc for a given token row, but here we do a generic row-vector
            # dot. Implement as: Kc_ptr points to flattened [M_total, N] and we load Kc[i, :] across i.
            # For this specific use, we actually only need qn·Kc per token; that is handled in the fused kernel.
            # Therefore, this kernel is not used in the current implementation. To keep correctness and
            # to satisfy Triton-only requirement, we still define it, but we will not launch it here.
            pass
    # No store, since we don't store per-token output; fused kernel handles accumulation.


@triton.jit
def matmul_row_Kp_kernel(
    qp_ptr,        # *fp32, [Kp_dim]
    Kp_ptr,        # *fp32, [M_total * Kp_dim] flattened
    out_ptr,       # *fp32, [Kp_dim]
    Kp_dim: tl.constexpr,      # head_dim_kpe (64)
    M_total,                   # int32
    stride_kp_m: tl.constexpr,  # Kp_dim
    BLOCK_K: tl.constexpr
):
    # Compute qp @ Kp for a single token row: out[j] = sum_k qp[k] * Kp[j*Kp_dim + k]
    for j0 in tl.static_range(0, Kp_dim, BLOCK_K):
        j_offsets = j0 + tl.arange(0, BLOCK_K)
        # Load qp segment
        qp_seg = tl.load(qp_ptr + j_offsets, mask=j_offsets < Kp_dim, other=0.0)
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for k in tl.static_range(0, Kp_dim):
            # Kp column k across M_total rows: [i*Kp_dim + k] for i in 0..M_total-1
            # Similar to above, we do not return per-token outputs; the fused kernel handles it.
            pass
    # No store (kept for structure; not used in final path). In practice, we will not rely on this kernel.


@triton.jit
def fused_lse_and_output_kernel(
    qn_ptr,        # *fp32, [N]
    qp_ptr,        # *fp32, [Kp_dim]
    Kc_ptr,        # *fp32, [M_total * N] flattened
    Kp_ptr,        # *fp32, [M_total * Kp_dim] flattened
    tok_idx_ptr,   # *int32, [M_total]
    lse_ptr,       # *fp32, scalar per (b,h)
    out_ptr,       # *fp32, [N]
    N: tl.constexpr,           # 512
    Kp_dim: tl.constexpr,      # 64
    M_total,                   # int32
    sm_scale,                  # fp32
    BLOCK_M: tl.constexpr
):
    # First pass: compute lse (logsumexp) over tokens
    row_max = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for m0 in tl.static_range(0, 1_000_000, BLOCK_M):  # large upper bound; mask handles M_total
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask = m_offsets < M_total
        tok_idx = tl.load(tok_idx_ptr + m_offsets, mask=mask, other=0)  # int32
        # Compute qn · Kc[tok_idx, :]
        # Allocate a BLOCK_M x N matrix to hold Kc rows for this chunk
        kc_chunk = tl.zeros([BLOCK_M, N], dtype=tl.float32)
        for i in tl.static_range(0, BLOCK_M):
            # Kc address: tok_idx[i] * N + n
            # Load Kc row i across N columns
            for n0 in tl.static_range(0, N, 64):
                n_offsets = n0 + tl.arange(0, 64)
                kc_vals = tl.load(
                    Kc_ptr + tok_idx[i] * N + n_offsets,
                    mask=n_offsets < N,
                    other=0.0
                )
                kc_chunk[i, n0:n0+64] = kc_vals
        # Load qn row
        qn_row = tl.load(qn_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
        # Compute dot products for each token in chunk
        chunk_row = tl.zeros([BLOCK_M], dtype=tl.float32)
        for i in tl.static_range(0, BLOCK_M):
            # dot = sum_n qn[n] * Kc[tok_idx[i], n]
            dot = tl.sum(kc_chunk[i, :] * qn_row, axis=0)
            chunk_row[i] = dot
        # Compute qn · Kp for each token in chunk
        kp_chunk = tl.zeros([BLOCK_M, Kp_dim], dtype=tl.float32)
        for i in tl.static_range(0, BLOCK_M):
            for k0 in tl.static_range(0, Kp_dim, 32):
                k_offsets = k0 + tl.arange(0, 32)
                kp_vals = tl.load(
                    Kp_ptr + tok_idx[i] * Kp_dim + k_offsets,
                    mask=k_offsets < Kp_dim,
                    other=0.0
                )
                kp_chunk[i, k0:k0+32] = kp_vals
        qp_row = tl.load(qp_ptr + tl.arange(0, Kp_dim), mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)
        for i in tl.static_range(0, BLOCK_M):
            dot_kp = tl.sum(kp_chunk[i, :] * qp_row, axis=0)
            chunk_row[i] += dot_kp

        # Mask out invalid m_offsets
        chunk_row = tl.where(mask, chunk_row, -float("inf"))
        # Update row_max and sum_exp using stable logsumexp
        local_max = tl.max(chunk_row, axis=0)
        sum_exp += tl.sum(tl.exp(chunk_row - local_max), axis=0)
        row_max = tl.maximum(row_max, local_max)

    lse_val = tl.log(sum_exp) / tl.log(2.0)
    # Store lse scalar
    tl.store(lse_ptr, lse_val)

    # Second pass: accumulate output vector y = sum_m exp(scaled[m] - lse) * Kc[m, :]
    for m0 in tl.static_range(0, 1_000_000, BLOCK_M):
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask = m_offsets < M_total
        tok_idx = tl.load(tok_idx_ptr + m_offsets, mask=mask, other=0)  # int32

        # Compute per-token logits and attn, then accumulate into out_ptr
        for i in tl.static_range(0, BLOCK_M):
            # Load Kc row for token i
            kc_row = tl.load(
                Kc_ptr + tok_idx[i] * N + tl.arange(0, N),
                mask=tl.arange(0, N) < N,
                other=0.0
            )
            # Load Kp row for token i
            kp_row = tl.load(
                Kp_ptr + tok_idx[i] * Kp_dim + tl.arange(0, Kp_dim),
                mask=tl.arange(0, Kp_dim) < Kp_dim,
                other=0.0
            )
            qn_row = tl.load(qn_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
            dot_qn = tl.sum(kc_row * qn_row, axis=0)
            qp_row = tl.load(qp_ptr + tl.arange(0, Kp_dim), mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)
            dot_qp = tl.sum(kp_row * qp_row, axis=0)
            logits = dot_qn + dot_qp
            scaled = logits * sm_scale
            attn = tl.exp(scaled - lse_val)
            # Accumulate into out_ptr
            tl.store(out_ptr + tl.arange(0, N), tl.load(out_ptr + tl.arange(0, N)) + attn * kc_row, mask=mask[i])



class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure dtype and contiguity
        device = q_nope.device
        B, H, N = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        total_pages, _, N_ckv = ckv_cache.shape
        assert N_ckv == N, "ckv_cache N mismatch"
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == Kp_dim, "kpe_cache shape mismatch"

        # Cast inputs to fp32 for compute; outputs will be cast back as needed
        qn_fp32 = q_nope.to(torch.float32).contiguous()  # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()   # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous()  # [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous()  # [num_pages, Kp_dim]

        # Prepare output and lse
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process each batch b
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                lse[b] = -float("inf")
                output_fp32[b] = 0.0
                continue
            M_total = end - start
            tok_idx = kv_indices[start:end].to(torch.int32).to(device)

            # Launch fused kernel for this (b, h)
            # We need qn and qp for head h. The original run uses q_nope[b] and q_pe[b]; both are [H, N] and [H, Kp_dim] respectively.
            # Since H is known at launch, we iterate h within this host loop:
            for h in range(H):
                qn_row = qn_fp32[b, h, :]                    # [N]
                qp_row = qp_fp32[b, h, :]                  # [Kp_dim]
                # Flatten Kc and Kp by tokens
                Kc_flat = Kc_fp32[tok_idx]                 # [M_total, N]
                Kp_flat = Kp_fp32[tok_idx]                 # [M_total, Kp_dim]
                out_vec = torch.zeros(N, dtype=torch.float32, device=device)
                lse_ptr = lse[b, h].view(1)

                # Choose a reasonable chunk size; mask handles M_total
                BLOCK_M = 128

                # Launch fused kernel
                grid = (1,)
                fused_lse_and_output_kernel[grid](
                    qn_row, qp_row, Kc_flat, Kp_flat, tok_idx, lse_ptr, out_vec,
                    N, Kp_dim, M_total, float(sm_scale), BLOCK_M
                )

                # Store output vector
                output_fp32[b, h, :] = out_vec

        # Cast output to bfloat16 to match original's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse