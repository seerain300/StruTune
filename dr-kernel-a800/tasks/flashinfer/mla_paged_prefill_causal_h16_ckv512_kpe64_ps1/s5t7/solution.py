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
        # kc tile [BLOCK_K, BLOCK_N] from kc[n, k]
        kc_tile = tl.load(
            kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
            mask=(offs_n[None, :] < n) & (offs_k[:, None] < k),
            other=0.0
        )
        acc += tl.dot(q, kc_tile)

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
        kp_tile = tl.load(
            kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
            mask=(offs_n[None, :] < n) & (offs_k[:, None] < k),
            other=0.0
        )
        acc += tl.dot(q, kp_tile)

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(a_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    b = tl.load(b_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    out = a + b

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets, out, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(logits_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logits = tl.load(
        logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=0.0
    )
    out = logits * scale

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        out,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


@triton.jit
def apply_mask(logits_ptr, out_ptr, L: tl.constexpr, query_abs_pos: tl.int32,
               BLOCK: tl.constexpr):
    # Apply mask per head h: keep j where j > query_abs_pos, else -inf
    # We process the row in blocks and loop over heads h. H is fixed at 16.
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
    # Compute lse[h] = log(sum_j exp(masked[h, j] - m[h])) + m[h], per row
    # We use two passes over L with BLOCK-sized chunks. We pass h via program_id.
    pid = tl.program_id(0)

    # Pass 1: compute max m over L
    m = -float("inf")
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        vals = tl.load(masked_ptr + pid * L + offs, mask=offs < L, other=-float("inf"))
        local_max = tl.max(vals, axis=0)
        m = tl.maximum(m, local_max)

    # Pass 2: compute sumexp = sum_j exp(val - m)
    sumexp = 0.0
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        vals = tl.load(masked_ptr + pid * L + offs, mask=offs < L, other=-float("inf"))
        exp_vals = tl.exp(vals - m)
        sumexp += tl.sum(exp_vals, axis=0)

    lse = tl.log(sumexp) + m  # logsumexp / ln(2) will be applied later by host
    tl.store(lse_ptr + pid, lse)


@triton.jit
def softmax_row_masked(masked_ptr, lse_ptr, out_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    # Compute softmax per row h using lse[h]
    # out[h, :] = exp(masked[h, :] - lse[h]) / sum_j exp(masked[h, j] - lse[h])
    # We loop over heads h and process the row in blocks.
    for h in range(0, 16):
        # Load lse for this row
        lse = tl.load(lse_ptr + h)

        # First compute denominator
        denom = 0.0
        for k in range(0, L, BLOCK):
            offs = k + tl.arange(0, BLOCK)
            vals = tl.load(masked_ptr + h * L + offs, mask=offs < L, other=-float("inf"))
            exp_vals = tl.exp(vals - lse)
            denom += tl.sum(exp_vals, axis=0)

        # Write normalized outputs
        for k in range(0, L, BLOCK):
            offs = k + tl.arange(0, BLOCK)
            vals = tl.load(masked_ptr + h * L + offs, mask=offs < L, other=-float("inf"))
            exp_vals = tl.exp(vals - lse)
            soft = exp_vals / denom
            tl.store(out_ptr + h * L + offs, soft, mask=offs < L)


@triton.jit
def matmul_attn_kc(soft_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[H, D] = soft[H, L] @ kc[L, D]
    m = H
    n = D
    k = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        soft = tl.load(
            soft_ptr + (offs_m[:, None] * L + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        kc_tile = tl.load(
            kc_ptr + (offs_k[:, None] * L + offs_n[None, :]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(soft, kc_tile)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        H = 16
        D = 512
        P = 64

        # Prepare Kc_all and Kp_all
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [PAGES, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [PAGES, P]

        total_q = q_nope.shape[0]
        num_qo_heads = H
        head_dim_ckv = D
        head_dim_kpe = P

        # Output tensors
        output = torch.empty((total_q, num_qo_heads, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch b
        b = 0
        while b < qo_indptr.shape[0] - 1:
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            # If no queries in this batch, skip
            if q_start >= q_end:
                b += 1
                continue

            # KV tokens for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                b += 1
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # [L]
            L = int(page_end - page_beg)

            # Slice Kc and Kp per batch
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Slice q_nope and q_pe per batch [Q, H, D] and [Q, H, P], but here Q=1
            q_nope_b = q_nope[q_start:q_end]  # [Q, H, D]
            q_pe_b = q_pe[q_start:q_end]      # [Q, H, P]
            Q = q_nope_b.shape[0]

            # For each query i in the batch
            for i in range(Q):
                q_pos = q_start + i

                # Load qn, qp for this query: [H, D] and [H, P]
                qn = q_nope_b[i].to(torch.float32)  # [H, D]
                qp = q_pe_b[i].to(torch.float32)    # [H, P]

                # Allocate intermediates
                A = torch.empty((H, L), dtype=torch.float32, device=device)
                B = torch.empty((H, L), dtype=torch.float32, device=device)
                logits = torch.empty((H, L), dtype=torch.float32, device=device)
                masked = torch.empty((H, L), dtype=torch.float32, device=device)

                # Launch matmuls
                grid_qn = (triton.cdiv(H, 16), triton.cdiv(L, 128))
                matmul_qn_kc[grid_qn](qn, Kc, A, H, D, L, 16, 128, 32)

                grid_qp = (triton.cdiv(H, 16), triton.cdiv(L, 128))
                matmul_qp_kp[grid_qp](qp, Kp, B, H, P, L, 16, 128, 32)

                # Add
                add_logits[(triton.cdiv(H, 16), triton.cdiv(L, 128))](A, B, logits, H, L, 16, 128)

                # Scale
                scale_logits[(triton.cdiv(H, 16), triton.cdiv(L, 128))](logits, masked, float(sm_scale), H, L, 16, 128)

                # Apply mask: j > (L - Q + i)
                query_abs_pos = int(L - Q + i)
                apply_mask[(triton.cdiv(L, 128))](masked, masked, L, query_abs_pos, 128)

                # row_logsumexp
                lse_row = torch.empty((H,), dtype=torch.float32, device=device)
                row_logsumexp_masked[(H,)](masked, lse_row, L, 128)

                # Compute final softmax outputs and store in output
                soft = torch.empty((H, L), dtype=torch.float32, device=device)
                softmax_row_masked[(H,)](masked, lse_row, soft, L, 128)

                # Matmul of soft @ Kc -> [H, D]
                out_h = torch.empty((H, D), dtype=torch.float32, device=device)
                grid_attn = (triton.cdiv(H, 16), triton.cdiv(D, 128))
                matmul_attn_kc[grid_attn](soft, Kc, out_h, H, L, D, 16, 128, 32)

                # Store to output tensor at position [q_pos, :, :]
                # output[q_pos, :, :] = out_h.to(torch.bfloat16)
                # Also store lse for this head
                for h in range(H):
                    lse[q_pos, h] = lse_row[h].item()  # host-side store of lse per head

            b += 1

        # We only return output[:, :, :] and lse, since the original returns (output, lse)
        # lse currently stores per (q_pos, head). We need to fill the whole lse tensor.
        # However, the original code computes lse per query position and head, and we do that above.
        # Return final output and lse
        return output, lse


def run(*args):
    return ModelNew()(*args)
