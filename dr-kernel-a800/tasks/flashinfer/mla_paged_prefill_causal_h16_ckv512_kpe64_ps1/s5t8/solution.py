import torch
import math
import triton
import triton.language as tl


@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[H, L] = qn[H, D] @ kc[L, D]^T
    m = H
    n = L
    k = D

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # qn tile [BLOCK_M, BLOCK_K]
        q = tl.load(
            qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        # kc tile as [BLOCK_K, BLOCK_N], note kc is [L, D]
        k_tile = tl.load(
            kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(q, k_tile)

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


@triton.jit
def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                 H: tl.constexpr, P: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[H, L] = qp[H, P] @ kp[L, P]^T
    m = H
    n = L
    k = P

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q = tl.load(
            qp_ptr + (offs_m[:, None] * P + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        k_tile = tl.load(
            kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(q, k_tile)

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(a_ptr + offs_m[:, None] * L + offs_n[None, :],
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    b = tl.load(b_ptr + offs_m[:, None] * L + offs_n[None, :],
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    out = a + b

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets, out,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(in_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    inp = tl.load(in_ptr + offs_m[:, None] * L + offs_n[None, :],
                  mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                  other=0.0)
    out = inp * scale

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets, out,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def apply_mask(logits_ptr, out_ptr, L: tl.constexpr, query_abs_pos: tl.int32,
               BLOCK: tl.constexpr):
    # Apply per-row mask: positions j <= query_abs_pos -> -inf, else keep
    # We process per head row; H=16 is a constexpr here, so loop is safe.
    for h in range(0, 16):
        for k in range(0, L, BLOCK):
            offs = k + tl.arange(0, BLOCK)
            logits = tl.load(logits_ptr + h * L + offs, mask=offs < L, other=0.0)
            keep = offs > query_abs_pos
            neg_inf = -float("inf")
            new = tl.where(keep, logits, neg_inf)
            tl.store(out_ptr + h * L + offs, new, mask=offs < L)


@triton.jit
def row_logsumexp_masked(masked_ptr, lse_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    # Compute lse[h] = log(sum_j exp(masked[h, j] - m[h])) + m[h], per row.
    # We launch one program per head h.
    h = tl.program_id(0)
    # First pass: max
    m = -float("inf")
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(masked_ptr + h * L + offs, mask=offs < L, other=-float("inf"))
        m = tl.maximum(m, tl.max(x, axis=0))
    # Second pass: sumexp
    sumexp = 0.0
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(masked_ptr + h * L + offs, mask=offs < L, other=-float("inf"))
        sumexp += tl.sum(tl.exp(x - m), axis=0)
    lse = tl.log(sumexp) + m  # natural logsumexp
    # Store as float32
    tl.store(lse_ptr + h, lse)


@triton.jit
def softmax_row_masked(masked_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr,
                        BLOCK: tl.constexpr):
    # Compute softmax per row h for masked logits using precomputed lse[h]
    for h in range(0, 16):
        # Load lse[h]
        lse = tl.load(lse_ptr + h)
        # First pass: compute denominator sum
        denom = 0.0
        for k in range(0, L, BLOCK):
            offs = k + tl.arange(0, BLOCK)
            x = tl.load(masked_ptr + h * L + offs, mask=offs < L, other=0.0)
            denom += tl.sum(tl.exp(x - lse), axis=0)
        # Second pass: write normalized outputs
        for k in range(0, L, BLOCK):
            offs = k + tl.arange(0, BLOCK)
            x = tl.load(masked_ptr + h * L + offs, mask=offs < L, other=0.0)
            soft = tl.exp(x - lse) / denom
            tl.store(out_ptr + h * L + offs, soft, mask=offs < L)


@triton.jit
def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # out[H, D] = attn[H, L] @ kc[L, D]
    m = H
    k = L
    n = D

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            attn_ptr + (offs_m[:, None] * L + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        c_tile = tl.load(
            kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(a, c_tile)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on the same device
        assert q_nope.device.type == 'cuda' and q_pe.device.type == 'cuda', "Use CUDA tensors"
        device = q_nope.device

        # Prepare Kc_all and Kp_all as 2-D tensors [num_pages, D] and [num_pages, P], float32
        # ckv_cache shape [num_pages, 1, D], kpe_cache [num_pages, 1, P]
        num_pages = ckv_cache.shape[0]
        D = q_nope.shape[-1]
        P = q_pe.shape[-1]
        H = q_nope.shape[1]
        # Make caches 2-D
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)

        # Output buffers
        total_q = int(qo_indptr[-1].item())
        output = torch.empty((total_q, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        # Precompute L for each batch b: L = kv_indptr[b+1] - kv_indptr[b]
        # We need len_indptr to be > 1; if not, there are no batches.
        len_indptr = qo_indptr.numel()
        assert len_indptr >= 2, "len_indptr must be >= 2"

        # Process each batch
        batch_start = 0
        while batch_start < len_indptr - 1:
            b = batch_start
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                batch_start += 1
                continue

            # Select token indices for this batch and slice Kc/Kp accordingly
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)
            L = tok_idx.numel()

            # Slice caches for this batch
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Slice q_nope and q_pe for this batch
            # q_nope: [q_start:q_end, H, D], q_pe: [q_start:q_end, H, P]
            q_nope_batch = q_nope[q_start:q_end].contiguous().to(torch.float32)  # [Q, H, D]
            q_pe_batch = q_pe[q_start:q_end].contiguous().to(torch.float32)     # [Q, H, P]
            Q = q_nope_batch.shape[0]

            # Compute outputs for each query i in the batch
            for i in range(Q):
                qn = q_nope_batch[i]  # [H, D]
                qp = q_pe_batch[i]    # [H, P]

                # Allocate intermediates
                logits = torch.empty((H, L), dtype=torch.float32, device=device)
                masked = torch.empty((H, L), dtype=torch.float32, device=device)

                # matmul qn @ Kc.T -> logits
                # Use BLOCK_M=64, BLOCK_N=128, BLOCK_K=64 for D=512
                grid_mat1 = (triton.cdiv(H, 64), triton.cdiv(L, 128))
                matmul_qn_kc[grid_mat1](
                    qn, Kc, logits,
                    H=16, D=512, L=L,
                    BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
                )

                # matmul qp @ Kp.T
                grid_mat2 = (triton.cdiv(H, 64), triton.cdiv(L, 128))
                matmul_qp_kp[grid_mat2](
                    qp, Kp, logits,  # overwrite logits
                    H=16, P=64, L=L,
                    BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
                )

                # Add two contributions
                add_logits[(triton.cdiv(H, 64), triton.cdiv(L, 128))](
                    logits, logits, logits, H=16, L=L, BLOCK_M=64, BLOCK_N=128
                )

                # Scale
                scaled = torch.empty_like(logits)
                scale_logits[(triton.cdiv(H, 64), triton.cdiv(L, 128))](
                    logits, scaled, sm_scale, H=16, L=L, BLOCK_M=64, BLOCK_N=128
                )

                # Apply mask: keep j where j > (L - Q + i)
                query_abs_pos = (L - Q) + i  # int
                masked.copy_(scaled)
                apply_mask[(16,)](masked, masked, L=L, query_abs_pos=query_abs_pos, BLOCK=128)

                # Row-wise logsumexp
                lse_i = torch.empty((H,), dtype=torch.float32, device=device)
                row_logsumexp_masked[(H,)](masked, lse_i, L=L, BLOCK=128)

                # Softmax and store output
                soft = torch.empty((H, L), dtype=torch.float32, device=device)
                softmax_row_masked[(16,)](masked, lse_i, soft, H=16, L=L, BLOCK=128)

                # final output = soft @ Kc -> [H, D]
                out_vec = torch.empty((H, D), dtype=torch.float32, device=device)
                grid_mat3 = (triton.cdiv(H, 64), triton.cdiv(D, 128))
                matmul_attn_kc[grid_mat3](
                    soft, Kc, out_vec,
                    H=16, L=L, D=512,
                    BLOCK_M=64, BLOCK_K=128, BLOCK_N=128
                )
                output[q_start + i] = out_vec.to(torch.bfloat16)

                # Store lse for this query
                lse[q_start + i] = lse_i[0]  # single head's lse

            batch_start += 1

        return output, lse


def run(*args):
    return ModelNew()(*args)
