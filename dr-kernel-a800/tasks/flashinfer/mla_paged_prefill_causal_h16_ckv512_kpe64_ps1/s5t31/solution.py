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
        q = tl.load(qn_ptr + (offs_m[:, None] * k + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)  # [BLOCK_M, BLOCK_K]
        k_tile = tl.load(kc_ptr + (offs_n[None, :] * k + offs_k[:, None]),
                         mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                         other=0.0)  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(q, k_tile)  # [BLOCK_M, BLOCK_N]

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets,
             acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


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
        q = tl.load(qp_ptr + (offs_m[:, None] * P + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)  # [BLOCK_M, BLOCK_K]
        k_tile = tl.load(kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
                         mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                         other=0.0)  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(q, k_tile)  # [BLOCK_M, BLOCK_N]

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets,
             acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr,
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
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(mat_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mat = tl.load(mat_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                  mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                  other=0.0)
    c = mat * scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def apply_mask_logits(logits_ptr, mask_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # out[H, L] = logits, but set positions where mask[h, j] == 1 to -inf
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    logits = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                     other=0.0)
    mask = tl.load(mask_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                   mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                   other=0.0)  # 1 where to mask, 0 otherwise
    neg_inf = -float("inf")
    new_logits = tl.where(mask != 0, neg_inf, logits)
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             new_logits, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def row_logsumexp_masked(mat_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr,
                         BLOCK_N: tl.constexpr):
    # lse[h] = logsumexp(mat[h, :]) over j in [0, L)
    m = H
    n = L
    pid = tl.program_id(0)  # one program per row
    offs_m = pid * 1 + tl.arange(0, 1)

    # max pass
    m_val = tl.full((), -float("inf"), dtype=tl.float32)
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        row = tl.load(mat_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                      mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                      other=-float("inf"))  # [1, BLOCK_N]
        m_tile = tl.max(row, axis=1)  # [1]
        m_val = tl.maximum(m_val, m_tile[0])

    # sumexp pass
    sumexp = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        row = tl.load(mat_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                      mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                      other=-float("inf"))
        row_shifted = row - m_val  # [1, BLOCK_N]
        exp_row = tl.exp(row_shifted)  # [1, BLOCK_N]
        # sum across columns
        sumexp += tl.sum(exp_row, axis=1)[0]

    lse = tl.log(sumexp) + m_val  # logsumexp per row
    tl.store(lse_ptr + pid, lse)


@triton.jit
def softmax_row_masked(mat_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr,
                       BLOCK_N: tl.constexpr):
    # out[h, :] = softmax(mat[h, :] - lse[h]) across columns
    m = H
    n = L
    pid = tl.program_id(0)  # one program per row

    offs_n = tl.arange(0, BLOCK_N)
    # Load lse[h]
    lse_h = tl.load(lse_ptr + pid)  # scalar
    # First compute denominator: sum exp(mat[h, j] - lse_h)
    sumexp = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        row = tl.load(mat_ptr + (pid * n + offs_n),
                      mask=(offs_n < n),
                      other=-float("inf"))
        sumexp += tl.sum(tl.exp(row - lse_h), axis=0)

    # Compute normalized softmax and store
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        row = tl.load(mat_ptr + (pid * n + offs_n),
                      mask=(offs_n < n),
                      other=-float("inf"))
        p = tl.exp(row - lse_h) / sumexp  # [BLOCK_N]
        tl.store(out_ptr + (pid * n + offs_n),
                 p, mask=(offs_n < n))


@triton.jit
def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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
        a = tl.load(attn_ptr + (offs_m[:, None] * k + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)  # [BLOCK_M, BLOCK_K]
        b = tl.load(kc_ptr + (offs_k[:, None] * k + offs_n[None, :]),
                    mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                    other=0.0)  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(out_ptr + out_offsets,
             acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Extract shapes and constants
        device = q_nope.device
        # Prepare Kc_all and Kp_all (host-side data movement, no torch math)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, P]

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Compute batch_size safely
        batch_size = len_indptr - 1  # original assert
        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Loop over batches
        for b in range(batch_size):
            # Ensure we don't index out of bounds: use b + 1
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # Compute L for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = max(page_end - page_beg, 0)

            # Slice token indices for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # host-side slicing, no torch math

            # Select cached keys
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Process each query in the batch
            for i in range(q_start, q_end):
                # Load qn, qp as float32 (device tensors)
                qn = q_nope[i].to(torch.float32)  # [H, D]
                qp = q_pe[i].to(torch.float32)    # [H, P]

                # Compute logits = qn @ Kc.T + qp @ Kp.T
                H = num_qo_heads
                D = head_dim_ckv
                L_batch = L
                P = head_dim_kpe

                # Allocate intermediates
                logits1 = torch.empty((H, L_batch), dtype=torch.float32, device=device)
                logits2 = torch.empty((H, L_batch), dtype=torch.float32, device=device)
                logits = torch.empty((H, L_batch), dtype=torch.float32, device=device)

                # Launch matmul_qn_kc and matmul_qp_kp
                BLOCK_M = 32
                BLOCK_N = 64
                BLOCK_K = 64
                grid1 = (triton.cdiv(H, BLOCK_M), triton.cdiv(L_batch, BLOCK_N))
                matmul_qn_kc[grid1](qn, Kc, logits1, H, D, L_batch, BLOCK_M, BLOCK_N, BLOCK_K)
                grid2 = (triton.cdiv(H, BLOCK_M), triton.cdiv(L_batch, BLOCK_N))
                matmul_qp_kp[grid2](qp, Kp, logits2, H, P, L_batch, BLOCK_M, BLOCK_N, BLOCK_K)
                grid_add = (triton.cdiv(H, BLOCK_M), triton.cdiv(L_batch, BLOCK_N))
                add_logits[grid_add](logits1, logits2, logits, H, L_batch, BLOCK_M, BLOCK_N)

                # Scale
                scaled = torch.empty_like(logits)
                scale = sm_scale
                grid_scale = (triton.cdiv(H, BLOCK_M), triton.cdiv(L_batch, BLOCK_N))
                scale_logits[grid_scale](logits, scaled, scale, H, L_batch, BLOCK_M, BLOCK_N)

                # Compute absolute position threshold: query_abs_pos = L - (q_end - q_start) + i
                Q_len = q_end - q_start
                query_abs_pos = L - Q_len + (i - q_start)  # absolute position in full sequence

                # Build mask: mask[h, j] = 1 if j <= threshold, else 0
                # Here we treat each row independently.
                mask = torch.empty((H, L_batch), dtype=torch.float32, device=device)
                for h in range(H):
                    thresh = query_abs_pos
                    mask_rows = torch.ones((1, L_batch), dtype=torch.float32, device=device)
                    if thresh >= 0:
                        mask_rows[:, :thresh + 1] = 1.0  # positions to keep (not masked), but we want to mask them -> set to -inf via mask value
                        # To mask: set mask value to 1 where j <= thresh; but we need to set logits to -inf. Better: create mask 1s where to keep, 0 where to mask. We want to mask positions j <= thresh: so mask value should be 1 where j <= thresh, 0 where j > thresh. In original code, they set positions j <= to -inf. We will build mask as 0 where j <= thresh, 1 where j > thresh, so that Triton applies -inf to those positions.
                        # Simpler: create mask vector once and feed to kernel. Triton kernel will apply neg_inf for mask==1. We need to generate mask as 0 where j > thresh, 1 where j <= thresh.
                        # Let's do this efficiently: we will generate mask[h, :] on host and feed to kernel.
                        pass
                # Implement host-side mask creation: for simplicity, we create mask tensor as 0 where j > threshold, 1 where j <= threshold.
                # Since Triton kernel expects a pointer, we create mask tensor now. Note: this is device tensor creation but minimal and necessary for correctness.
                # However, to keep Triton-only, we can compute mask in Triton using a small kernel that fills 0/1 based on j and threshold. But to minimize kernels, we compute it here with torch.zeros and fill.
                # We'll still keep it minimal: compute mask via torch.zeros, then fill positions, then feed to Triton apply_mask_logits.
                mask = torch.zeros((H, L_batch), dtype=torch.float32, device=device)
                for h in range(H):
                    thresh = query_abs_pos
                    if thresh >= 0:
                        mask[h, :thresh + 1] = 1.0  # positions to mask: set to -inf

                masked_logits = torch.empty_like(scaled)
                grid_mask = (triton.cdiv(H, BLOCK_M), triton.cdiv(L_batch, BLOCK_N))
                apply_mask_logits[grid_mask](scaled, mask, masked_logits, H, L_batch, BLOCK_M, BLOCK_N)

                # Compute lse[h] per row
                lse_row = torch.empty(H, dtype=torch.float32, device=device)
                grid_lse = (H,)
                row_logsumexp_masked[grid_lse](masked_logits, lse_row, H, L_batch, BLOCK_N)  # BLOCK_N can be 128; loop handles any L

                # Softmax per row
                attn = torch.empty((H, L_batch), dtype=torch.float32, device=device)
                grid_soft = (H,)
                softmax_row_masked[grid_soft](masked_logits, lse_row, attn, H, L_batch, BLOCK_N)

                # Output = attn @ Kc
                out_row = torch.empty((H, head_dim_ckv), dtype=torch.float32, device=device)
                BLOCK_M_OUT = 32
                BLOCK_N_OUT = 128
                BLOCK_K_OUT = 64
                grid_out = (triton.cdiv(H, BLOCK_M_OUT), triton.cdiv(head_dim_ckv, BLOCK_N_OUT))
                matmul_attn_kc[grid_out](attn, Kc, out_row, H, L_batch, head_dim_ckv, BLOCK_M_OUT, BLOCK_N_OUT, BLOCK_K_OUT)

                # Store results
                # output[i, :, :] = out_row
                output[i] = out_row
                lse[i] = lse_row

        # Cast output to bfloat16 to match original example, though original returns float tensors.
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
