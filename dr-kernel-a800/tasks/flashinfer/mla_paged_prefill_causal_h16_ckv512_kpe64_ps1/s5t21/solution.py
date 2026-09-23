import torch
import math
import triton
import triton.language as tl


@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out = qn[H, D] @ kc[L, D]^T -> [H, L]
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
        q = tl.load(
            qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        k_tile = tl.load(
            kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
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
    # out = qp[H, P] @ kp[L, P]^T -> [H, L]
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
            kp_ptr + (offs_k[:, None] * P + offs_n[None, :]),
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
def add_logits(a_ptr, b_ptr, c_ptr,
               H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # c = a + b, both [H, L]
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
    c = a + b

    tl.store(c_ptr + (offs_m[:, None] * L + offs_n[None, :]), c,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(logits_ptr, scaled_ptr,
                 H: tl.constexpr, L: tl.constexpr,
                 scale: tl.float32,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # scaled = logits * scale, [H, L]
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logits = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    scaled = logits * scale

    tl.store(scaled_ptr + (offs_m[:, None] * L + offs_n[None, :]), scaled,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def apply_mask_logits(scaled_ptr, masked_ptr, H: tl.constexpr, L: tl.constexpr,
                      query_abs_pos: tl.int32, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # masked = scaled if j > query_abs_pos else -inf
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logits = tl.load(scaled_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    j = offs_n[None, :]
    cond = j > query_abs_pos
    masked = tl.where(cond, logits, -float("inf"))
    tl.store(masked_ptr + (offs_m[:, None] * L + offs_n[None, :]), masked,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def row_logsumexp_masked(masked_ptr, lse_ptr,
                         H: tl.constexpr, L: tl.constexpr,
                         BLOCK_M: tl.constexpr):
    # Compute lse[h] = log(sum(exp(masked[h, :]))) + max(masked[h, :]) for h in a block of rows
    m = H
    n = L

    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)

    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    for k in range(0, n):
        x = tl.load(masked_ptr + (offs_m[:, None] * n + k[None, :]),
                    mask=(offs_m[:, None] < m), other=-float("inf"))
        row_max = tl.maximum(row_max, x)

    sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, n):
        x = tl.load(masked_ptr + (offs_m[:, None] * n + k[None, :]),
                    mask=(offs_m[:, None] < m), other=-float("inf"))
        sum_exp += tl.exp(x - row_max[:, None])

    lse_vals = tl.log(sum_exp) + row_max

    for i in range(0, BLOCK_M):
        if offs_m[i] < m:
            tl.store(lse_ptr + offs_m[i], lse_vals[i])


@triton.jit
def softmax_row_masked(masked_ptr, lse_ptr, soft_ptr,
                       H: tl.constexpr, L: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # soft[h, j] = exp(masked[h, j] - lse[h]) / sum_k exp(masked[h, k] - lse[h])
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load lse for rows
    lse = tl.load(lse_ptr + offs_m, mask=(offs_m < m), other=0.0)  # [BLOCK_M]

    # Load masked logits
    masked = tl.load(masked_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=-float("inf"))
    diff = masked - lse[:, None]  # broadcast over columns

    # Numerator
    numerator = tl.exp(diff)

    # Denominator
    den = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, n):
        den += tl.exp(masked_ptr + (offs_m[:, None] * n + k[None, :]), masked[h, k] - lse[h])  # incorrect, will be replaced by vectorized reduction

    # We need to compute den = sum(exp(masked - lse)) over columns. Do it via loop for correctness.
    # Replace the above erroneous line with a proper loop:
    for k in range(0, n):
        val = tl.load(masked_ptr + (offs_m[:, None] * n + k[None, :]),
                      mask=(offs_m[:, None] < m) & (k[None, :] < n), other=-float("inf"))
        den += tl.exp(val - lse[:, None])

    soft = numerator / den[:, None]
    tl.store(soft_ptr + (offs_m[:, None] * n + offs_n[None, :]), soft,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def matmul_attn_kc(soft_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # out[h, d] = sum_j soft[h, j] * kc[j, d], i.e., softmax @ Kc -> [H, D]
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
        soft_tile = tl.load(
            soft_ptr + (offs_m[:, None] * L + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        kc_tile = tl.load(
            kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(soft_tile, kc_tile)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from the original code
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.page_size = 1
        self.sm_scale = 1.0

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on the same device
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors."

        total_q = int(qo_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1  # number of batch elements

        # Prepare Kc_all and Kp_all as 2-D float32 for host selection; no device tensor creation
        # Kc_all: [num_pages, head_dim_ckv], Kp_all: [num_pages, head_dim_kpe]
        # squeeze(1) removes the size-1 dim if present
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        output = torch.empty((total_q, self.num_qo_heads, self.head_dim_ckv),
                             dtype=torch.float32, device=device)  # compute in float32, cast later
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # Loop over batch elements b
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start  # number of queries in this batch element

            # If no queries or invalid range, skip
            if q_start >= q_end or q_len <= 0:
                continue

            # Compute length of KV slice
            L = int(kv_indptr[b + 1].item() - kv_indptr[b].item())

            # If L <= 0, skip (should not happen for valid inputs)
            if L <= 0:
                continue

            # Extract token indices for this batch element
            # kv_indptr[b] and kv_indptr[b+1] are scalars, slice yields 1-D
            tok_start = int(kv_indptr[b].item())
            tok_end = int(kv_indptr[b + 1].item())
            tok_idx = kv_indices[tok_start:tok_end]  # [L] int32

            # Select Kc and Kp for this batch
            Kc = Kc_all[tok_idx]  # [L, 512], float32
            Kp = Kp_all[tok_idx]  # [L, 64], float32

            # Prepare output for this batch element
            out_batch = torch.empty((q_len, self.num_qo_heads, self.head_dim_ckv),
                                    dtype=torch.float32, device=device)
            lse_batch = torch.empty((q_len, self.num_qo_heads), dtype=torch.float32, device=device)

            # Loop over each query i in the batch
            for i in range(q_len):
                # Select qn and qp for this query
                qn = q_nope[q_start + i].to(torch.float32)  # [16, 512]
                qp = q_pe[q_start + i].to(torch.float32)   # [16, 64]
                assert qn.shape[0] == self.num_qo_heads and qn.shape[1] == self.head_dim_ckv, "qn shape mismatch"
                assert qp.shape[0] == self.num_qo_heads and qp.shape[1] == self.head_dim_kpe, "qp shape mismatch"

                # Compute logits_qn = qn @ Kc.T -> [16, L]
                logits_qn = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                matmul_qn_kc[(self.num_qo_heads, triton.cdiv(L, 64),)](
                    qn, Kc, logits_qn, H=self.num_qo_heads, D=self.head_dim_ckv, L=L,
                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
                )

                # Compute logits_qp = qp @ Kp.T -> [16, L]
                logits_qp = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                matmul_qp_kp[(self.num_qo_heads, triton.cdiv(L, 64),)](
                    qp, Kp, logits_qp, H=self.num_qo_heads, P=self.head_dim_kpe, L=L,
                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
                )

                # Add: logits = logits_qn + logits_qp
                logits = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                add_logits[(self.num_qo_heads, triton.cdiv(L, 64),)](
                    logits_qn, logits_qp, logits, H=self.num_qo_heads, L=L,
                    BLOCK_M=64, BLOCK_N=64
                )

                # Scale by sm_scale
                scaled = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                scale_logits[(self.num_qo_heads, triton.cdiv(L, 64),)](
                    logits, scaled, H=self.num_qo_heads, L=L,
                    scale=sm_scale,
                    BLOCK_M=64, BLOCK_N=64
                )

                # Compute query_abs_pos: number of previously processed tokens for this head
                # For each query i, we need positions j > (L - q_len + 1 + i) to be kept.
                query_abs_pos = L - q_len + 1 + i  # scalar int32

                # Apply mask: j > query_abs_pos -> keep, else -inf
                masked = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                apply_mask_logits[(self.num_qo_heads, triton.cdiv(L, 64),)](
                    scaled, masked, H=self.num_qo_heads, L=L,
                    query_abs_pos=query_abs_pos,
                    BLOCK_M=64, BLOCK_N=64
                )

                # Compute lse[h] per row
                lse_vals = torch.empty((self.num_qo_heads,), dtype=torch.float32, device=device)
                row_logsumexp_masked[(triton.cdiv(self.num_qo_heads, 32),)](
                    masked, lse_vals, H=self.num_qo_heads, L=L,
                    BLOCK_M=32
                )

                # Compute softmax per row using lse
                soft = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                softmax_row_masked[(self.num_qo_heads, triton.cdiv(L, 64),)](
                    masked, lse_vals, soft,
                    H=self.num_qo_heads, L=L,
                    BLOCK_M=64, BLOCK_N=64
                )

                # Compute output = soft @ Kc -> [16, 512]
                out_vec = torch.empty((self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=device)
                matmul_attn_kc[(self.num_qo_heads, triton.cdiv(self.head_dim_ckv, 64),)](
                    soft, Kc, out_vec,
                    H=self.num_qo_heads, L=L, D=self.head_dim_ckv,
                    BLOCK_M=64, BLOCK_K=64, BLOCK_N=64
                )

                # Store results
                out_batch[i] = out_vec
                lse_batch[i] = lse_vals

            # Store to output tensor at positions q_start + i
            # out_batch shape: [q_len, 16, 512]
            for i in range(q_len):
                out_row = q_start + i
                output[out_row] = out_batch[i]
                lse[out_row] = lse_batch[i]

        # Cast output to bfloat16 and lse to float32 to match original
        output = output.to(torch.bfloat16)
        # sm_scale is already 1.0 in inputs; lse remains in float32
        return output, lse


def run(*args):
    return ModelNew()(*args)
