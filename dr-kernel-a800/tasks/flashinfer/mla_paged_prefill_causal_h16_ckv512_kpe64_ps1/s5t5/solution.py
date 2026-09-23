import torch
import math
import triton
import triton.language as tl


@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute C = qn[H, D] @ Kc[L, D]^T -> [H, L]
    # qn_ptr: [H, D], row-major; Kc_ptr: [L, D], row-major
    # out_ptr: [H, L], row-major
    m = H
    n = L
    k = D

    # program ids
    pid_m = tl.program_id(0)  # head index h
    pid_n = tl.program_id(1)  # block index along L

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        q = tl.load(
            qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        k_block = tl.load(
            kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
            mask=(offs_n[None, :] < n) & (offs_k[:, None] < k),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(q, k_block)  # [BLOCK_M, BLOCK_N]

    # write out
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
    # Compute C = qp[H, P] @ Kp[L, P]^T -> [H, L]
    m = H
    n = L
    k = P

    pid_m = tl.program_id(0)  # head index h
    pid_n = tl.program_id(1)  # block index along L

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        q = tl.load(
            qp_ptr + (offs_m[:, None] * P + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        k_block = tl.load(
            kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
            mask=(offs_n[None, :] < n) & (offs_k[:, None] < k),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(q, k_block)  # [BLOCK_M, BLOCK_N]

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr,
               H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute C = A + B, A,B are [H, L]
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(
        a_ptr + (offs_m[:, None] * L + offs_n[None, :]),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=0.0
    )
    b = tl.load(
        b_ptr + (offs_m[:, None] * L + offs_n[None, :]),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=0.0
    )
    c = a + b

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        c,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


@triton.jit
def scale_logits(logits_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute C = Logits * scale, store to out_ptr
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
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Apply mask: positions j <= query_abs_pos -> -inf, else keep
    m = 1  # one row, i.e., one head's logits vector
    # We process L in blocks; since H=16 (outer program id 0), we could tile by m as well, but here m=1 by design.
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logits = tl.load(logits_ptr + offs_n, mask=offs_n < L, other=0.0)
    # mask true where j > query_abs_pos
    keep = offs_n > query_abs_pos
    neg_inf = -float("inf")
    new = tl.where(keep, logits, neg_inf)
    tl.store(out_ptr + offs_n, new, mask=offs_n < L)


@triton.jit
def row_logsumexp_masked(masked_ptr, lse_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    # Compute lse[h] = log(sum_j exp(masked[h, j] - m[h])) + m[h], per row
    # We process one row per program; here m=1 as we compute per-head per query.
    pid = tl.program_id(0)
    # First pass: max
    max_val = -float("inf")
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(masked_ptr + offs, mask=offs < L, other=-float("inf"))
        cur_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, cur_max)
    # Second pass: sumexp
    sum_exp = 0.0
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(masked_ptr + offs, mask=offs < L, other=-float("inf"))
        e = tl.exp(x - max_val)
        sum_exp += tl.sum(e, axis=0)
    lse = tl.log(sum_exp) + max_val
    # store lse[h]
    tl.store(lse_ptr + pid, lse)


@triton.jit
def softmax_row_masked(masked_ptr, lse_ptr, softmax_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    # Compute softmax for one row using precomputed lse[h]
    pid = tl.program_id(0)
    # Load lse
    lse = tl.load(lse_ptr + pid)
    # Two-phase: compute denom, then write normalized
    denom = 0.0
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(masked_ptr + offs, mask=offs < L, other=-float("inf"))
        e = tl.exp(x - lse)
        denom += tl.sum(e, axis=0)
    # write normalized softmax
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(masked_ptr + offs, mask=offs < L, other=-float("inf"))
        e = tl.exp(x - lse)
        soft = e / denom
        tl.store(softmax_ptr + offs, soft, mask=offs < L)


@triton.jit
def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute C = attn[H, L] @ Kc[L, D] -> [H, D]
    m = H
    n = D
    k = L

    pid_m = tl.program_id(0)  # head index h
    pid_n = tl.program_id(1)  # block index along D

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        attn_block = tl.load(
            attn_ptr + (offs_m[:, None] * L + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        kc_block = tl.load(
            kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(attn_block, kc_block)  # [BLOCK_M, BLOCK_N]

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Device: assume all inputs are on same device
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
        batch_size = qo_indptr.shape[0] - 1

        # Prepare Kc_all and Kp_all: squeeze the batch dim and keep float32
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, P]

        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Constants
        D = head_dim_ckv
        P = head_dim_kpe

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or page_beg >= page_end:
                continue

            L = page_end - page_beg  # number of selected tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.long).to(device)  # indices into Kc_all/Kp_all
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [Q, H, D]
            q_pe_batch = q_pe[q_start:q_end].to(torch.float32)      # [Q, H, P]
            Q = q_end - q_start

            for i in range(Q):
                # 1) matmul qn @ Kc.T -> A [H, L]
                # 2) matmul qp @ Kp.T -> B [H, L]
                # 3) add A+B -> C
                # 4) scale C by sm_scale -> D
                # 5) apply mask per row (for head h, keep j > (L - Q + i), else -inf)
                # 6) lse per head row
                # 7) softmax per row
                # 8) matmul attn @ Kc -> output [H, D]

                qn = q_nope_batch[i].to(torch.float32)  # [H, D]
                qp = q_pe_batch[i].to(torch.float32)   # [H, P]

                # Allocate intermediate buffers
                A = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                B = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                C = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                D_scaled = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                masked = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                softmax_row = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                out_tmp = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

                # Triton matmul: qn @ Kc.T -> A
                BLOCK_M = 16  # H
                BLOCK_N = 128  # tile along L
                BLOCK_K = 64  # tile along D
                grid_a = (num_qo_heads, triton.cdiv(L, BLOCK_N))
                matmul_qn_kc[grid_a](qn, Kc, A, H=num_qo_heads, D=D, L=L,
                                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

                # Triton matmul: qp @ Kp.T -> B
                grid_b = (num_qo_heads, triton.cdiv(L, BLOCK_N))
                matmul_qp_kp[grid_b](qp, Kp, B, H=num_qo_heads, P=P, L=L,
                                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

                # Triton add A + B -> C
                grid_add = (num_qo_heads, triton.cdiv(L, BLOCK_N))
                add_logits[grid_add](A, B, C, H=num_qo_heads, L=L, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

                # Triton scale C by sm_scale -> D_scaled
                grid_scale = (num_qo_heads, triton.cdiv(L, BLOCK_N))
                scale_logits[grid_scale](C, D_scaled, sm_scale, H=num_qo_heads, L=L, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

                # Triton apply mask: per row, j > (L - Q + i)
                query_abs_pos = (L - Q) + i  # j > query_abs_pos keep, else -inf
                # We need masked as a contiguous vector for one head. We use head 0; since H=16, we write per head.
                grid_mask = (triton.cdiv(L, BLOCK_N),)
                # Launch once per row with single m dimension; Triton doesn't have a 2D grid here, so we replicate per head
                for h in range(num_qo_heads):
                    masked_row_ptr = masked + h * L  # view as row base
                    apply_mask[grid_mask](D_scaled[h], masked_row_ptr, L=L, query_abs_pos=query_abs_pos)

                # Triton row_logsumexp over masked[h, :]
                grid_lse = (num_qo_heads,)
                lse_vec = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                row_logsumexp_masked[grid_lse](masked, lse_vec, L=L, BLOCK=128)
                # Store lse for this query
                lse[q_start + i] = lse_vec[0]  # lse is vectorized per head, but we update single head per loop; adjust below

                # Triton softmax per row
                grid_soft = (num_qo_heads,)
                softmax_row[grid_soft](masked, lse_vec, softmax_row, L=L, BLOCK=128)

                # Triton matmul attn @ Kc -> output row
                grid_mm = (num_qo_heads, triton.cdiv(head_dim_ckv, 128))
                matmul_attn_kc[grid_mm](softmax_row, Kc, out_tmp, H=num_qo_heads, L=L, D=D,
                                        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=64)

                # Write to output tensor
                output[q_start + i] = out_tmp  # out_tmp is [H, D] for this head, but loop i handles per query; adjust

                # Update lse per head for this query: use lse_vec[h] for each head h
                # Since we only write one head per iteration, we can do:
                for h in range(num_qo_heads):
                    lse[q_start + i, h] = lse_vec[h]

        return output, lse


def run(*args):
    return ModelNew()(*args)
