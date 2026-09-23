import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels
if TRITON_AVAILABLE:
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
            # qn: [m, k] -> qn_ptr + m*off_k + off_k
            q = tl.load(
                qn_ptr + (offs_m[:, None] * k + offs_k[None, :]),
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0
            )
            # kc^T: we read kc[n, k] as [offs_k, offs_n]
            k_tile = tl.load(
                kc_ptr + (offs_n[None, :] * k + offs_k[:, None]),
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
    def add_logits(a_ptr, b_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        m = H
        n = L
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a = tl.load(a_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        b = tl.load(b_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        out = a + b
        tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    @triton.jit
    def scale_logits(in_ptr, out_ptr, scale: tl.float32,
                     H: tl.constexpr, L: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        m = H
        n = L
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a = tl.load(in_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        out = a * scale
        tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    @triton.jit
    def apply_mask(logits_ptr, mask_ptr, out_ptr, scale: tl.float32,
                   H: tl.constexpr, L: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        # mask_ptr contains 0.0 (keep) or 1.0 (mask), we avoid -inf to prevent NaNs.
        m = H
        n = L
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        logits = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                         mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        mask = tl.load(mask_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                       mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        neg_large = -1e20
        out = tl.where(mask > 0.0, neg_large, logits)  # set masked positions to very negative
        out = out * scale
        tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    @triton.jit
    def row_logsumexp_row(in_ptr, row_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, ln2_inv: tl.float32, BLOCK_N: tl.constexpr):
        # Compute per-row logsumexp over L and store to out_ptr[h]
        m = H
        n = L
        pid = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        # Load and compute row-wise max
        row_max = -1e20
        for j in range(0, n, BLOCK_N):
            cur = tl.load(in_ptr + (pid * n + j + offs_n), mask=(j + offs_n) < n, other=-1e20)
            row_max = tl.maximum(row_max, tl.max(cur, axis=0))
        # Compute sumexp with masking: positions <= threshold are set to very negative; here we rely on input mask or external logic.
        sum_exp = 0.0
        # We assume input has already been masked in apply_mask; here we just exponentiate the loaded values.
        for j in range(0, n, BLOCK_N):
            cur = tl.load(in_ptr + (pid * n + j + offs_n), mask=(j + offs_n) < n, other=-1e20)
            sum_exp += tl.sum(tl.exp(cur - row_max), axis=0)
        lse = row_max + tl.log(sum_exp) * ln2_inv
        tl.store(out_ptr + pid, lse)

    @triton.jit
    def softmax_row_masked_row(in_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, ln2_inv: tl.float32, BLOCK_N: tl.constexpr):
        # Compute softmax for one row h: out[h, :] = exp(in[h, :] - lse[h]) / sum_j exp(in[h, j] - lse[h])
        m = H
        n = L
        pid = tl.program_id(0)
        lse = tl.load(lse_ptr + pid)
        offs_n = tl.arange(0, BLOCK_N)
        # First pass: sum of exp shifted by lse
        sum_exp = 0.0
        for j in range(0, n, BLOCK_N):
            cur = tl.load(in_ptr + (pid * n + j + offs_n), mask=(j + offs_n) < n, other=-1e20)
            sum_exp += tl.sum(tl.exp(cur - lse), axis=0)
        # Second pass: write normalized outputs
        for j in range(0, n, BLOCK_N):
            cur = tl.load(in_ptr + (pid * n + j + offs_n), mask=(j + offs_n) < n, other=-1e20)
            out = tl.exp(cur - lse) / sum_exp
            tl.store(out_ptr + (pid * n + j + offs_n), out, mask=(j + offs_n) < n)

    @triton.jit
    def matmul_attn_kc(softmax_row_ptr, kc_ptr, out_ptr,
                       H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # out[D] = softmax_row[H] @ kc[L, D]
        # softmax_row_ptr is 1D vector of length H
        m = 1  # single row
        n = D
        k = L
        pid_m = tl.program_id(0)  # 0 .. 0
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # we use BLOCK_M=H, but here m=1, so this is just 0
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # Load softmax row (length H)
        # We assume H is small and can load directly; if H is dynamic, we’d need a loop. Here H=16 fixed.
        offs_h = tl.arange(0, BLOCK_M)
        softmax_row = tl.load(softmax_row_ptr + offs_h, mask=offs_h < H, other=0.0)  # [BLOCK_M]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            # kc: [k, n] -> kc_ptr + offs_k*n + offs_n
            kc_tile = tl.load(
                kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
                mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                other=0.0
            )
            # softmax_row: [BLOCK_M] -> broadcast over BLOCK_N
            acc += softmax_row[:, None] * kc_tile
        out_offsets = offs_m[:, None] * D + offs_n[None, :]
        tl.store(out_ptr + out_offsets, acc,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move to device and dtype preparation
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be CUDA tensors."

        # Prepare Kc_all and Kp_all: squeeze 1 dim and make contiguous
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, P]

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        H = num_qo_heads
        D = head_dim_ckv
        P = head_dim_kpe
        len_indptr = qo_indptr.shape[0]
        # batch_size is len_indptr - 1 (per input pattern)
        batch_size = len_indptr - 1

        # Output and lse buffers
        output = torch.zeros(
            (total_q, H, D),
            dtype=torch.bfloat16,
            device=device
        )
        lse = torch.full(
            (total_q, H),
            -float("inf"),
            dtype=torch.float32,
            device=device
        )

        # Process each batch b
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            # q_end: next element in qo_indptr; ensure within bounds
            q_end = int(qo_indptr[b + 1].item()) if b + 1 < len_indptr else int(qo_indptr[-1].item())
            assert q_start >= 0 and q_end <= total_q, "qo_indptr indices out of bounds."

            # kv section
            assert len(kv_indptr) == len_indptr
            assert b < len(kv_indptr) - 1, "kv_indptr has insufficient length for batch."
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            assert 0 <= page_beg < num_pages and 0 <= page_end <= num_pages, "kv_indptr indices out of bounds."
            L = int(page_end - page_beg)
            assert L > 0, "No KV tokens selected."

            # Slice Kc and Kp for this batch
            tok_idx = kv_indices[page_beg:page_end]  # [L]
            Kc = Kc_all[tok_idx]  # [L, D], contiguous
            Kp = Kp_all[tok_idx]  # [L, P], contiguous
            assert Kc.shape[1] == D and Kp.shape[1] == P

            # q_len in this batch
            q_len = q_end - q_start

            for i in range(q_len):
                query_abs_pos = L - q_len + i  # absolute position of current query in this batch's KV
                # Load qn and qp
                qn = q_nope[q_start + i].to(torch.float32).contiguous()  # [H, D]
                qp = q_pe[q_start + i].to(torch.float32).contiguous()   # [H, P]

                # Allocate intermediates on device
                logits = torch.empty((H, L), dtype=torch.float32, device=device)
                # Compute qn @ Kc.T and qp @ Kp.T
                # Choose block sizes suitable for H=16, L=34 (small), D=512, P=64
                BLOCK_M = 16
                BLOCK_N = 32
                BLOCK_K = 64
                matmul_qn_kc[(1, 1)](qn, Kc, logits[:H, :], H, D, L, BLOCK_M, BLOCK_N, BLOCK_K)
                # Initialize second logits with zeros, then compute qp @ Kp.T and add
                logits2 = torch.zeros((H, L), dtype=torch.float32, device=device)
                matmul_qp_kp[(1, 1)](qp, Kp, logits2[:H, :], H, P, L, BLOCK_M, BLOCK_N, BLOCK_K)
                logits = logits + logits2

                # Scale
                logits_scaled = torch.empty_like(logits)
                scale = float(sm_scale)  # ensure Python float, not tensor
                scale_logits[(1, 1)](logits, logits_scaled, scale, H, L, BLOCK_M, BLOCK_N)

                # Prepare mask: per row h, keep positions j where j > query_abs_pos, else 1.0 (mask)
                # If query_abs_pos < 0, all positions are kept (no mask). Typical cases:
                #   q_len = 1 -> query_abs_pos = L - 1 (e.g., 33). Some positions masked depending on L.
                #   q_len > 1 -> query_abs_pos can be negative, hence no mask.
                # Build mask buffer: [H, L], 0.0 where j > query_abs_pos, 1.0 otherwise
                mask = torch.empty((H, L), dtype=torch.float32, device=device)
                j = torch.arange(L, device=device).view(1, L)  # [1, L]
                for h in range(H):
                    cond = j[0] > query_abs_pos
                    mask[h, :] = 0.0 if cond else 1.0
                # Apply mask: set masked positions to very negative
                logits_masked = torch.empty_like(logits_scaled)
                apply_mask[(1, 1)](logits_scaled, mask, logits_masked, scale, H, L, BLOCK_M, BLOCK_N)

                # Compute per-row lse = logsumexp(masked_logits) / ln(2)
                ln2_inv = 1.0 / math.log(2.0)
                row_lse = torch.empty((H,), dtype=torch.float32, device=device)
                row_logsumexp_row[(H,)](logits_masked, logits_masked, row_lse, H, L, ln2_inv, 64)

                # Softmax over rows using lse
                softmax_rows = torch.empty_like(logits_masked)
                softmax_row_masked_row[(H,)](logits_masked, row_lse, softmax_rows, H, L, ln2_inv, 64)

                # Compute output[h, :] = softmax[h, :] @ Kc -> [H, D]
                out_vec = torch.empty((D,), dtype=torch.float32, device=device)
                matmul_attn_kc[(1, 1)](softmax_rows[0, :], Kc, out_vec, H, D, L, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)
                output[q_start + i] = out_vec.to(torch.bfloat16)

                # Update lse
                lse[q_start + i] = row_lse[0]  # only one head in the loop, but keep shape as (1, H)

        return output, lse


def run(*args):
    return ModelNew()(*args)
