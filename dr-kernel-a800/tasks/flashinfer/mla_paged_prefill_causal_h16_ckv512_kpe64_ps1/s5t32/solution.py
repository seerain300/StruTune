import torch
import triton
import triton.language as tl

# GEMM: out[H, L] = qn[H, D] @ Kc[L, D]^T
@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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
        # qn tile: [BLOCK_M, BLOCK_K]
        q = tl.load(qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        # kc tile (as [BLOCK_K, BLOCK_N]): kc[n, k] -> [offs_k, offs_n]
        k_tile = tl.load(kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
                         mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                         other=0.0)
        acc += tl.dot(q, k_tile)

    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


# GEMM: out[H, L] = qp[H, P] @ Kp[L, P]^T
@triton.jit
def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                 H: tl.constexpr, P: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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
                    other=0.0)
        k_tile = tl.load(kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
                         mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                         other=0.0)
        acc += tl.dot(q, k_tile)

    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


# Elementwise add: out = A + B
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


# Elementwise scale: out = A * scale
@triton.jit
def scale_logits(inp_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(inp_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    c = a * scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


# Apply mask: out[h, j] = -inf if j <= threshold else logits[h, j]
@triton.jit
def apply_mask(logits_ptr, out_ptr, threshold, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logits = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                     other=0.0)
    # Build mask: 1 if j <= threshold else 0
    # Note: threshold is scalar; compare with offs_n
    jvec = offs_n[None, :]  # shape [1, BLOCK_N]
    mask = (jvec <= threshold).to(tl.int32)  # 1 where mask, 0 else
    neg_inf = -float("inf")
    new_logits = tl.where(mask != 0, neg_inf, logits)
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             new_logits, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


# Row-wise logsumexp (return per-row lse) using two-pass
@triton.jit
def row_logsumexp_masked(mat_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr):
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
        m_tile = tl.max(row, axis=1)  # [1]
        m_val = tl.maximum(m_val, m_tile[0])

    # sumexp pass
    sumexp = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        row = tl.load(mat_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                      mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                      other=-float("inf"))
        row = row - m_val  # shift
        exp_row = tl.exp(row)
        sumexp += tl.sum(exp_row, axis=1)[0]

    lse = tl.log(sumexp) + m_val
    # Store lse[h]
    tl.store(lse_ptr + offs_m, lse)


# Compute softmax per row using given lse[h]
@triton.jit
def softmax_row_masked(mat_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid = tl.program_id(0)
    offs_m = pid * 1 + tl.arange(0, 1)

    lse_val = tl.load(lse_ptr + offs_m)  # scalar per row
    for n0 in range(0, n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        vals = tl.load(mat_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                       mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                       other=-float("inf"))
        soft = tl.exp(vals - lse_val)  # [1, BLOCK_N]
        total = tl.sum(soft, axis=1)[0]
        soft = soft / total
        tl.store(out_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                 soft, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


# GEMM: out[H, D] = softmax[H, L] @ Kc[L, D]
@triton.jit
def matmul_attn_kc(softmax_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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
        a = tl.load(softmax_ptr + (offs_m[:, None] * k + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        b = tl.load(kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
                    mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                    other=0.0)  # kc^T tile: [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)

    tl.store(out_ptr + (offs_m[:, None] * n + offs_n[None, :]),
             acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # fixed constants from original assertions
        self.H = 16
        self.D = 512
        self.P = 64
        # tuning parameters (can be adjusted if needed)
        self.BLOCK_M = 16
        self.BLOCK_N = 64
        self.BLOCK_K = 32

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on device and types are correct
        assert q_nope.device.type == 'cuda' and q_pe.device.type == 'cuda' and ckv_cache.device.type == 'cuda' and kpe_cache.device.type == 'cuda'
        assert qo_indptr.device.type == 'cuda' and kv_indptr.device.type == 'cuda' and kv_indices.device.type == 'cuda'

        # Convert all to float32 for compute
        q_nope_f = q_nope.to(torch.float32)
        q_pe_f = q_pe.to(torch.float32)
        # Prepare Kc_all and Kp_all as 2-D [num_pages, D] and [num_pages, P]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, P]

        total_q = int(q_nope_f.shape[0])
        num_qo_heads = self.H
        head_dim_ckv = self.D
        head_dim_kpe = self.P

        # Output tensors
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope_f.device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope_f.device)

        # Process each batch element b from 0 to len_indptr - 2
        len_indptr = int(qo_indptr.shape[0])
        batch_size = len_indptr - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            # Ensure q_start < q_end (the original code asserts total_q == qo_indptr[-1], but we keep defensive checks)
            if q_start >= q_end:
                continue

            # Number of KV indices for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue
            L = (page_end - page_beg)

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # 1-D long tensor
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Process each query in the batch
            for i in range(q_start, q_end):
                # Compute absolute position threshold for causal mask
                # query_abs_pos = L - (q_end - q_start) + i
                # Note: q_start, q_end are scalars; q_end - q_start gives Q for this batch
                query_abs_pos = L - (q_end - q_start) + (i - q_start)

                # Load qn and qp
                qn = q_nope_f[i]  # [H, D]
                qp = q_pe_f[i]    # [H, P]

                # Compute logits[H, L] = qn @ Kc.T + qp @ Kp.T
                logits_qn = torch.empty((self.H, L), dtype=torch.float32, device=q_nope_f.device)
                logits_qp = torch.empty((self.H, L), dtype=torch.float32, device=q_nope_f.device)

                # Launch GEMMs
                grid_qn = (triton.cdiv(self.H, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                matmul_qn_kc[grid_qn](
                    qn, Kc, logits_qn,
                    H=self.H, D=self.D, L=L,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
                )

                grid_qp = (triton.cdiv(self.H, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                matmul_qp_kp[grid_qp](
                    qp, Kp, logits_qp,
                    H=self.H, P=self.P, L=L,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
                )

                # Add
                logits_sum = torch.empty((self.H, L), dtype=torch.float32, device=q_nope_f.device)
                grid_add = (triton.cdiv(self.H, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                add_logits[grid_add](
                    logits_qn, logits_qp, logits_sum,
                    H=self.H, L=L, BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
                )

                # Scale
                scaled = torch.empty((self.H, L), dtype=torch.float32, device=q_nope_f.device)
                grid_scale = (triton.cdiv(self.H, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                scale_logits[grid_scale](
                    logits_sum, scaled,
                    sm_scale,
                    H=self.H, L=L, BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
                )

                # Apply mask: positions j <= query_abs_pos -> -inf
                masked = torch.empty((self.H, L), dtype=torch.float32, device=q_nope_f.device)
                grid_mask = (triton.cdiv(self.H, self.BLOCK_M), triton.cdiv(L, self.BLOCK_N))
                apply_mask[grid_mask](
                    scaled, masked,
                    query_abs_pos,
                    H=self.H, L=L, BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
                )

                # Compute per-row lse = logsumexp(masked) with two-pass (max, sumexp)
                lse_row = torch.empty(self.H, dtype=torch.float32, device=q_nope_f.device)
                grid_lse = (self.H,)
                row_logsumexp_masked[grid_lse](
                    masked, lse_row,
                    H=self.H, L=L, BLOCK_N=self.BLOCK_N
                )

                # Output[h, :] = softmax(masked[h, :]) @ Kc[h, :]
                output_row = torch.empty((self.H, self.D), dtype=torch.float32, device=q_nope_f.device)
                grid_out = (triton.cdiv(self.H, self.BLOCK_M), triton.cdiv(self.D, self.BLOCK_N))
                # Launch softmax per row and then GEMM
                # We need to compute softmax first
                softmax_row = torch.empty((self.H, L), dtype=torch.float32, device=q_nope_f.device)
                grid_softmax = (self.H,)
                softmax_row_masked[grid_softmax](
                    masked, lse_row, softmax_row,
                    H=self.H, L=L, BLOCK_N=self.BLOCK_N
                )

                matmul_attn_kc[grid_out](
                    softmax_row, Kc, output_row,
                    H=self.H, L=L, D=self.D,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
                )

                # Store outputs
                # output[i, h, :] = output_row[h, :]
                # lse[i, h] = lse_row[h] * (1.0 / ln(2.0))
                ln2 = 0.6931471805599453
                lse[i, :] = lse_row / ln2

                # Assign to output
                # output is [total_q, H, D]; i is row in q dimension
                # We already computed output_row as [H, D], store it.
                # Here total_q dimension is implicitly handled by i in [q_start, q_end).
                # Ensure correct placement: output[i, :, :] = output_row
                output[i] = output_row

        return output, lse


def run(*args):
    return ModelNew()(*args)
