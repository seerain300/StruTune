import math
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute out[H, L] = qn[H, D] @ kc[L, D]^T
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
        # kc tile [BLOCK_K, BLOCK_N]
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
    # Compute out[H, L] = qp[H, P] @ kp[L, P]^T
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
    c = a + b
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(logits_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mat = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                  mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                  other=0.0)
    new = mat * scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             new, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def apply_mask_logits(mat_ptr, mask_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # out[H, L] = mat, but set positions where mask[h, j] == 1 to -inf
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mat = tl.load(mat_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                  mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                  other=0.0)
    mask = tl.load(mask_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                   mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                   other=0.0)
    neg_inf = -float("inf")
    new_mat = tl.where(mask != 0, neg_inf, mat)
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             new_mat, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def softmax_row_masked(mat_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr):
    # Given mat[H, L] (masked logits), and lse[H], compute softmax per row and write to out[H, L]
    m = H
    n = L

    pid = tl.program_id(0)
    offs_m = pid * 1 + tl.arange(0, 1)  # one row per program
    row = tl.load(mat_ptr + (offs_m * L + tl.arange(0, L)),
                  mask=(offs_m < m),
                  other=0.0)  # load entire row
    lse = tl.load(lse_ptr + offs_m, mask=(offs_m < m), other=0.0)
    # exp and sum per row
    exp_row = tl.exp(row - lse)
    denom = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / denom
    tl.store(out_ptr + (offs_m * L + tl.arange(0, L)), softmax_row, mask=(offs_m < m))


@triton.jit
def matmul_attn_kc(softmax_ptr, kc_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute out[H, D] = softmax[H, L] @ kc[L, D]
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
        # softmax tile [BLOCK_M, BLOCK_K]
        sm = tl.load(
            softmax_ptr + (offs_m[:, None] * k + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        # kc tile [BLOCK_K, BLOCK_N]
        k_tile = tl.load(
            kc_ptr + (offs_n[None, :] * k + offs_k[:, None]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(sm, k_tile)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


@triton.jit
def row_logsumexp_masked(mat_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr,
                         BLOCK_N: tl.constexpr):
    # lse[h] = logsumexp(masked mat[h, :]) over j in [0, L)
    m = H
    n = L

    pid = tl.program_id(0)
    offs_m = pid * 1 + tl.arange(0, 1)  # one row per program

    # max pass
    m_val = tl.full((), -float("inf"), dtype=tl.float32)
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        row = tl.load(mat_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                      mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                      other=-float("inf"))
        m_tile = tl.max(row, axis=1)[0]  # [1]
        m_val = tl.maximum(m_val, m_tile)

    # sumexp pass
    sumexp = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        row = tl.load(mat_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                      mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                      other=-float("inf"))
        exp_row = tl.exp(row - m_val)
        sumexp += tl.sum(exp_row, axis=1)[0]

    lse = tl.log(sumexp) + m_val
    tl.store(lse_ptr + offs_m, lse, mask=(offs_m < m))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device consistency and extract shapes
        device = q_nope.device
        total_q = int(qo_indptr[1].item()) if qo_indptr.numel() > 1 else int(qo_indptr[0].item())
        # Prepare cached Kc_all, Kp_all as 2-D contiguous float32
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, P]
        H = 16
        D = 512
        P = 64
        batch_size = int(kv_indptr.numel()) - 1
        # Constants for Triton launch parameters
        BLOCK_M = 16  # heads per row
        BLOCK_N = 64  # columns per tile
        BLOCK_K = 32  # reduction tile

        # Output buffers
        output = torch.empty((total_q, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg

            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [L]
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Loop over queries in the batch
            for i in range(q_len):
                # Allocate intermediate tensors (float32 for compute)
                logits = torch.empty((H, L), dtype=torch.float32, device=device)
                scaled = torch.empty((H, L), dtype=torch.float32, device=device)
                masked = torch.empty((H, L), dtype=torch.float32, device=device)

                # qn, qp: [H, D/P] -> pointers
                qn = q_nope[q_start + i].to(torch.float32).contiguous()  # [H, D]
                qp = q_pe[q_start + i].to(torch.float32).contiguous()   # [H, P]

                # GEMMs
                matmul_qn_kc[(1, 1)](qn, Kc, logits, H, D, L,
                                     BLOCK_M, BLOCK_N, BLOCK_K)
                matmul_qp_kp[(1, 1)](qp, Kp, logits, H, P, L,
                                     BLOCK_M, BLOCK_N, BLOCK_K)
                add_logits[(1, 1)](logits, logits, scaled, H, L, BLOCK_M, BLOCK_N)
                scaled = scaled * sm_scale

                # Apply mask: query_abs_pos = L - q_len + i
                query_abs_pos = L - q_len + i
                # Build mask: [H, L], 1 where j <= query_abs_pos else 0
                j = torch.arange(L, device=device).view(1, L)
                mask_int = torch.where(j <= query_abs_pos, torch.tensor(1, dtype=torch.int32, device=device), torch.tensor(0, dtype=torch.int32, device=device))
                apply_mask_logits[(1, 1)](scaled, mask_int, masked, H, L, BLOCK_M, BLOCK_N)

                # Row-wise logsumexp (masked)
                lse_vec = torch.empty((H,), dtype=torch.float32, device=device)
                row_logsumexp = row_logsumexp_masked[(H,)](masked, lse_vec, H, L, BLOCK_N)
                lse[q_start + i] = lse_vec  # store per-query per-head lse

                # Softmax and final GEMM
                attn = torch.empty((H, L), dtype=torch.float32, device=device)
                softmax_row_masked[(H,)](masked, lse[q_start + i], attn, H, L, BLOCK_N)
                out = torch.empty((H, D), dtype=torch.float32, device=device)
                matmul_attn_kc[(1, 1)](attn, Kc, out, H, L, D, BLOCK_M, BLOCK_N, BLOCK_K)
                output[q_start + i] = out.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
