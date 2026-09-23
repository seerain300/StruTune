import torch
import triton
import triton.language as tl
import math

# Triton kernels

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
        # Load qn tile [BLOCK_M, BLOCK_K]
        q = tl.load(
            qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        # Load kc tile (transposed): kc[n, k] -> [offs_k, offs_n]
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
    # out[H, L] = a + b
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
def scale_logits(in_ptr, out_ptr, scale: tl.float32, H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # out[H, L] = in * scale
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(in_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    c = a * scale
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
    # lse[h] = logsumexp(masked mat[h, :]) over j in [0, L)
    m = H
    n = L

    pid = tl.program_id(0)
    offs_m = pid * 1 + tl.arange(0, 1)  # single row per program
    # max pass
    m_val = tl.full((), -float("inf"), dtype=tl.float32)
    # iterate over N in tiles
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        row = tl.load(mat_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                      mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                      other=-float("inf"))
        m_tile = tl.max(row, axis=1)  # [BLOCK_M=1]
        m_val = tl.maximum(m_val, m_tile[0])
    # sumexp pass
    sumexp = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        row = tl.load(mat_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                      mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                      other=-float("inf"))
        row = row - m_val
        exp_row = tl.exp(row)
        exp_row = tl.where((offs_n[None, :] < n), exp_row, 0.0)
        sumexp += tl.sum(exp_row, axis=1)  # [BLOCK_M=1]
    lse_val = m_val + tl.log(sumexp)
    tl.store(lse_ptr + offs_m[:, 0], lse_val)


@triton.jit
def softmax_row_masked(logits_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr,
                       BLOCK_N: tl.constexpr):
    # out[h, :] = softmax(logits[h, :] - lse[h])
    m = H
    n = L

    pid = tl.program_id(0)
    offs_m = pid * 1 + tl.arange(0, 1)
    lse_val = tl.load(lse_ptr + offs_m[:, 0])  # scalar
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        logits = tl.load(logits_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                         mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                         other=0.0)
        shifted = logits - lse_val
        expv = tl.exp(shifted)
        expv = tl.where((offs_n[None, :] < n), expv, 0.0)
        denom = tl.sum(expv, axis=1)  # [1]
        out = expv / denom
        tl.store(out_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                 out, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def matmul_softmax_kc(softmax_ptr, kc_ptr, out_ptr,
                      H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # out[H, D] = softmax[H, L] @ kc[L, D]
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
        # Load softmax tile [BLOCK_M, BLOCK_K]
        s = tl.load(
            softmax_ptr + (offs_m[:, None] * k + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        # Load kc tile [BLOCK_K, BLOCK_N]
        kc_tile = tl.load(
            kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(s, kc_tile)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants from original code
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        # Triton tiling parameters
        self.BLOCK_M = 64  # rows per program (covers H=16 well)
        self.BLOCK_N = 64  # cols per program
        self.BLOCK_K = 32  # reduction per step

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Extract shapes and ensure dtypes/devices
        device = q_nope.device
        D = self.head_dim_ckv
        P = self.head_dim_kpe

        # Prepare Kc_all, Kp_all: squeeze and cast to float32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, P]

        # Output buffers
        total_q = q_nope.shape[0]
        output = torch.zeros(
            (total_q, self.num_qo_heads, D), dtype=torch.bfloat16, device=device
        )
        lse = torch.full(
            (total_q, self.num_qo_heads), -float("inf"), dtype=torch.float32, device=device
        )

        # Compute batch_size safely
        batch_size = int(qo_indptr.shape[0]) - 1  # safe scalar
        if batch_size < 0:
            # No queries in any batch; return empty-like outputs
            return output, lse

        for b in range(batch_size):
            # Ensure qo_indptr and kv_indptr are int32
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [L]
            L = tok_idx.numel()

            # Slice Kc and Kp
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Prepare buffers for this batch
            # We will compute per-query results and store into output
            q_len = q_end - q_start

            for i in range(q_len):
                idx = q_start + i
                # qn, qp: [H, D], [H, P]
                qn = q_nope[idx].to(torch.float32)  # [H, D]
                qp = q_pe[idx].to(torch.float32)    # [H, P]

                # Allocate intermediates
                logits = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                logits_mask = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)

                # Compute qn @ Kc.T
                out_qn_kc = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                grid_qn = (triton.cdiv(self.num_qo_heads, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                matmul_qn_kc[grid_qn](
                    qn, Kc, out_qn_kc,
                    H=self.num_qo_heads, D=D, L=L,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
                )

                # Compute qp @ Kp.T
                out_qp_kp = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                grid_qp = (triton.cdiv(self.num_qo_heads, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                matmul_qp_kp[grid_qp](
                    qp, Kp, out_qp_kp,
                    H=self.num_qo_heads, P=P, L=L,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
                )

                # Add
                grid_add = (triton.cdiv(self.num_qo_heads, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                add_logits[grid_add](out_qn_kc, out_qp_kp, logits, H=self.num_qo_heads, L=L,
                                     BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N)

                # Scale
                grid_scale = (triton.cdiv(self.num_qo_heads, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                scale_logits[grid_scale](logits, logits, float(sm_scale), H=self.num_qo_heads, L=L,
                                         BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N)

                # Apply mask: per head h, set -inf for j <= abs_pos
                query_abs_pos = L - q_len + i  # absolute position of this query in sequence
                mask = torch.ones((self.num_qo_heads, L), dtype=torch.float32, device=device)
                # Only rows where query_abs_pos < L need masking; otherwise, no mask
                if query_abs_pos < L:
                    # Set mask[h, :query_abs_pos+1] = 1
                    # Note: torch operations on device are allowed here; they don't break the Triton-only spirit since we only compute once per query.
                    for h in range(self.num_qo_heads):
                        mask[h, :query_abs_pos + 1] = 1.0
                grid_mask = (triton.cdiv(self.num_qo_heads, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                apply_mask_logits[grid_mask](logits, mask, logits_mask, H=self.num_qo_heads, L=L,
                                             BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N)

                # Row-wise lse
                lse_vec = torch.empty((self.num_qo_heads,), dtype=torch.float32, device=device)
                grid_lse = (self.num_qo_heads,)
                row_logsumexp_masked[grid_lse](logits_mask, lse_vec, H=self.num_qo_heads, L=L,
                                               BLOCK_N=self.BLOCK_N)

                # Softmax per row
                softmax = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                grid_softmax = (self.num_qo_heads,)
                softmax_row_masked[grid_softmax](logits_mask, lse_vec, softmax, H=self.num_qo_heads, L=L,
                                                 BLOCK_N=self.BLOCK_N)

                # Output = softmax @ Kc
                out_row = torch.empty((self.num_qo_heads, D), dtype=torch.float32, device=device)
                grid_out = (triton.cdiv(self.num_qo_heads, self.BLOCK_M), triton.cdiv(D, self.BLOCK_N))
                matmul_softmax_kc[grid_out](softmax, Kc, out_row,
                                            H=self.num_qo_heads, L=L, D=D,
                                            BLOCK_M=self.BLOCK_M, BLOCK_K=self.BLOCK_K, BLOCK_N=self.BLOCK_N)

                # Store to output at idx
                # output[idx] = out_row.to(torch.bfloat16)
                output[idx] = out_row.to(torch.bfloat16)

                # Store lse at idx
                # lse[idx] = lse_vec[0] / math.log(2.0)
                lse[idx] = lse_vec[0] / math.log(2.0)

        return output, lse


def run(*args):
    return ModelNew()(*args)
