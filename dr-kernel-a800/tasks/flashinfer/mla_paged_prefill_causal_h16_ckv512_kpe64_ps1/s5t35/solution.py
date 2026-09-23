import math
import triton
import triton.language as tl


@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[h, l] = sum_d qn[h, d] * kc[l, d]
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
        # qn: [H, D] row-major
        q = tl.load(
            qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        # kc: [L, D], we access kc[l, d] -> kc_ptr + l*D + d
        k_tile = tl.load(
            kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(q, k_tile)  # [BLOCK_M, BLOCK_N]

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
    # out[h, l] = sum_p qp[h, p] * kp[l, p]
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
        )  # [BLOCK_M, BLOCK_K]
        # kp: [L, P], kp[l, p] -> kp_ptr + l*P + p
        k_tile = tl.load(
            kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(q, k_tile)

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


@triton.jit
def add_logits(a_ptr, b_ptr, c_ptr, H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
    c = a + b
    tl.store(c_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             c,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(logits_ptr, scale, out_ptr, H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logit = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                    other=0.0)
    out = logit * scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             out,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def row_max_then_sumexp(logits_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr,
                        BLOCK_N: tl.constexpr):
    # Compute per-row lse using two-pass: max then sumexp
    m = H
    n = L

    pid = tl.program_id(0)
    # Each program handles one row (head)
    h = pid
    if h >= m:
        return

    offs_n = tl.arange(0, BLOCK_N)
    max_val = -float("inf")
    # Pass 1: row-wise max
    for start in range(0, n, BLOCK_N):
        l = start + offs_n
        mask = l < n
        # logits[h, l]
        row_base = h * L
        vals = tl.load(logits_ptr + row_base + l, mask=mask, other=-float("inf"))
        # reduce max across the vector
        local_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, local_max)

    # Pass 2: sumexp with masking and log2 normalization
    sumexp = 0.0
    ln2 = 1.0 / math.log(2.0)
    for start in range(0, n, BLOCK_N):
        l = start + offs_n
        mask = l < n
        vals = tl.load(logits_ptr + row_base + l, mask=mask, other=-float("inf"))
        # sum exp(vals - max_val) across vector
        sumexp += tl.sum(tl.exp(vals - max_val), axis=0)

    lse = tl.log(sumexp) + max_val  # logsumexp in natural log
    # Normalize by ln(2) to match original
    lse = lse * ln2
    tl.store(lse_ptr + h, lse)


@triton.jit
def softmax_row_masked(logits_ptr, lse_ptr, softmax_ptr, H: tl.constexpr, L: tl.constexpr,
                       BLOCK_N: tl.constexpr):
    # softmax[h, l] = exp(logits[h, l] - lse[h]) / sum_j exp(logits[h, j] - lse[h])
    m = H
    n = L

    pid = tl.program_id(0)
    h = pid
    if h >= m:
        return

    offs_n = tl.arange(0, BLOCK_N)
    ln2 = 1.0 / math.log(2.0)
    lse = tl.load(lse_ptr + h) * ln2  # convert logsumexp from ln scale to log2 scale internally

    # Compute denominator: sum over positions
    sumexp = 0.0
    for start in range(0, n, BLOCK_N):
        l = start + offs_n
        mask = l < n
        vals = tl.load(logits_ptr + h * L + l, mask=mask, other=-float("inf"))
        # We want to compute exp(vals - lse) but need sum; we can't vectorize tl.sum without loop.
        # Do scalar accumulation across tiles:
        for j in range(start, start + BLOCK_N):
            if j < n:
                sumexp += tl.exp(tl.load(logits_ptr + h * L + j) - lse)

    # Now fill softmax row
    for start in range(0, n, BLOCK_N):
        l = start + offs_n
        mask = l < n
        vals = tl.load(logits_ptr + h * L + l, mask=mask, other=-float("inf"))
        exp_vals = tl.exp(vals - lse)
        softmax_row = exp_vals / sumexp  # scalar
        # Store each element
        for j in range(start, start + BLOCK_N):
            if j < n:
                tl.store(softmax_ptr + h * L + j, softmax_row)


@triton.jit
def matmul_attn_kc(softmax_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[h, d] = sum_l softmax[h, l] * kc[l, d]
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
        # softmax: [H, L]
        s = tl.load(
            softmax_ptr + (offs_m[:, None] * L + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < n),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        # kc: [L, D], kc[l, d] -> kc_ptr + l*D + d
        k_tile = tl.load(
            kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
            mask=(offs_k[:, None] < n) & (offs_n[None, :] < k),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(s, k_tile)  # [BLOCK_M, BLOCK_N]

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
        # device and dtype setup
        device = q_nope.device
        D = 512  # fixed head_dim_ckv
        P = 64   # fixed head_dim_kpe

        # Prepare caches on host (squeezing singleton dim, float32 for compute)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, P]

        total_q = int(q_nope.shape[0])
        H = q_nope.shape[1]
        assert H == 16, "num_qo_heads must be 16"
        assert q_nope.shape[2] == D, "head_dim_ckv must be 512"
        assert q_pe.shape[2] == P, "head_dim_kpe must be 64"
        num_qo_heads = H
        head_dim_ckv = D
        head_dim_kpe = P
        num_pages = Kc_all.shape[0]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1  # consistent with original

        # Output buffers
        output = torch.zeros((total_q, num_qo_heads, D), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Process each batch b
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())         # scalar
            q_end = int(qo_indptr[b + 1].item())      # scalar
            qo_len = q_end - q_start                   # number of queries in this batch

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            L = kv_end - kv_start                      # number of tokens to consider

            # If nothing, skip
            if qo_len == 0 or L == 0:
                continue

            # tok_idx: [L]
            tok_idx = kv_indices[kv_start:kv_end].to(torch.long)

            # Select cached keys
            Kc = Kc_all[tok_idx]  # [L, D], float32
            Kp = Kp_all[tok_idx]  # [L, P], float32

            # Slice query tensors
            q_nope_batch = q_nope[q_start:q_end]  # [qo_len, H, D], bfloat16
            q_pe_batch = q_pe[q_start:q_end]      # [qo_len, H, P], bfloat16

            # Ensure contiguous
            q_nope_batch = q_nope_batch.contiguous()
            q_pe_batch = q_pe_batch.contiguous()
            Kc = Kc.contiguous()
            Kp = Kp.contiguous()

            # Run per query i
            for i in range(qo_len):
                # Shapes
                qn = q_nope_batch[i]  # [H, D], bfloat16
                qp = q_pe_batch[i]    # [H, P], bfloat16

                # Output buffers
                logits_qn = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_qp = torch.empty((H, L), dtype=torch.float32, device=device)
                logits = torch.empty((H, L), dtype=torch.float32, device=device)
                masked_logits = torch.empty((H, L), dtype=torch.float32, device=device)
                softmax_row = torch.empty((H, L), dtype=torch.float32, device=device)

                # GEMM 1: qn @ Kc.T -> [H, L]
                BLOCK_M = 16
                BLOCK_N = 64
                BLOCK_K = 64
                grid_qn = (triton.cdiv(H, BLOCK_M), triton.cdiv(L, BLOCK_N))
                matmul_qn_kc[grid_qn](qn, Kc, logits_qn,
                                      H, D, L,
                                      BLOCK_M, BLOCK_N, BLOCK_K)

                # GEMM 2: qp @ Kp.T -> [H, L]
                grid_qp = (triton.cdiv(H, BLOCK_M), triton.cdiv(L, BLOCK_N))
                logits_qp = torch.empty((H, L), dtype=torch.float32, device=device)
                matmul_qp_kp[grid_qp](qp, Kp, logits_qp,
                                      H, P, L,
                                      BLOCK_M, BLOCK_N, BLOCK_K)

                # Add
                grid_add = (triton.cdiv(H, BLOCK_M), triton.cdiv(L, BLOCK_N))
                add_logits[grid_add](logits_qn, logits_qp, logits,
                                     H, L,
                                     BLOCK_M, BLOCK_N)

                # Scale
                grid_scale = (triton.cdiv(H, BLOCK_M), triton.cdiv(L, BLOCK_N))
                scale_logits[grid_scale](logits, sm_scale, masked_logits,
                                         H, L,
                                         BLOCK_M, BLOCK_N)

                # Compute LSE per row: logsumexp(masked_logits) / ln(2)
                # row_max_then_sumexp requires 1D grid over H
                # Choose BLOCK_N as a power of two up to L, but Triton prefers compile-time constants.
                # We will loop in Triton with BLOCK_N=128 (>=L).
                grid_lse = (H,)
                ln2 = 1.0 / math.log(2.0)
                row_max_then_sumexp[grid_lse](masked_logits, lse[q_start + i],
                                              H, L,
                                              128)

                # Softmax per row
                grid_softmax = (H,)
                softmax_row_masked[grid_softmax](masked_logits, lse[q_start + i], softmax_row,
                                                 H, L,
                                                 128)

                # Final output = softmax @ Kc -> [H, D]
                out_row = torch.empty((H, D), dtype=torch.float32, device=device)
                BLOCK_M_out = 16
                BLOCK_N_out = 64
                BLOCK_K_out = 64
                grid_out = (triton.cdiv(H, BLOCK_M_out), triton.cdiv(D, BLOCK_N_out))
                matmul_attn_kc[grid_out](softmax_row, Kc, out_row,
                                         H, L, D,
                                         BLOCK_M_out, BLOCK_N_out, BLOCK_K_out)

                # Store to output
                output[q_start + i] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
