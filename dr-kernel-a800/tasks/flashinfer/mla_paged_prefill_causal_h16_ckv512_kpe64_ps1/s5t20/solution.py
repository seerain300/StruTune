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
        q = tl.load(
            qn_ptr + (offs_m[:, None] * k + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        # kc[n, k] with n = offs_n, k = offs_k -> shape (BLOCK_N, BLOCK_K)
        k_tile = tl.load(
            kc_ptr + (offs_k[:, None] * k + offs_n[None, :]),
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
def add_logits(a_ptr, b_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a = tl.load(a_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    b = tl.load(b_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    out = a + b
    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets, out, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(in_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(in_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    y = x * scale
    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets, y, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def apply_mask(logits_ptr, masked_ptr, query_abs_pos, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    logits = tl.load(logits_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    # mask: keep where j > query_abs_pos, else -inf
    j = offs_n[None, :]
    cond = j > query_abs_pos
    masked = tl.where(cond, logits, -float("inf"))
    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(masked_ptr + out_offsets, masked, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def row_logsumexp_masked(logits_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr):
    m = H
    n = L
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    # First pass: row max
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    for k in range(0, n):
        x = tl.load(logits_ptr + (offs_m[:, None] * n + k[None, :]),
                    mask=(offs_m[:, None] < m), other=-float("inf"))
        row_max = tl.maximum(row_max, x)
    # Second pass: sumexp over masked positions
    sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, n):
        x = tl.load(logits_ptr + (offs_m[:, None] * n + k[None, :]),
                    mask=(offs_m[:, None] < m), other=-float("inf"))
        # mask: keep if k > query_abs_pos; here we use j > query_abs_pos in kernel, so here we rely on masked_ptr
        # Since we compute lse from masked logits, we should load masked buffer. To keep simple, we recompute condition here.
        # However, we'll compute based on logits_ptr with condition; Triton doesn't support dynamic broadcasting here.
        # We will rely on host-side masked_ptr for this kernel. So we load masked buffer directly from host.
        # We pass masked_ptr instead: The host will prepare masked buffer before calling this kernel.
        # But Triton doesn't support passing masked_ptr here; we'll instead use the same buffer by assuming masked is in logits_ptr? No: we need to pass masked.
        # To keep things correct, we implement masked by using masked_ptr as input:
        # We redefine the function signature to accept masked_ptr. Triton will read masked_ptr.
        masked_val = tl.load(logits_ptr + (offs_m[:, None] * n + k[None, :]),
                             mask=(offs_m[:, None] < m), other=0.0)  # placeholder; re-define signature for masked_ptr
    # Note: The above shows we need masked_ptr. Triton doesn't support masked_ptr argument; we need to re-implement correctly.
    # So we define a proper kernel below that uses masked_ptr.
    pass


# We'll define a proper kernel using masked_ptr. Define it properly now.

@triton.jit
def row_logsumexp_masked(logits_ptr, masked_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr):
    m = H
    n = L
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    # First pass: row max on masked
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    for k in range(0, n):
        x = tl.load(masked_ptr + (offs_m[:, None] * n + k[None, :]),
                    mask=(offs_m[:, None] < m), other=-float("inf"))
        row_max = tl.maximum(row_max, x)
    # Second pass: sumexp
    sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, n):
        x = tl.load(masked_ptr + (offs_m[:, None] * n + k[None, :]),
                    mask=(offs_m[:, None] < m), other=-float("inf"))
        sum_exp += tl.exp(x - row_max[:, None])
    # Compute lse
    lse_vals = tl.log(sum_exp) + row_max
    out_offsets = offs_m * 0  # placeholder
    # Store to lse_ptr as a vector
    # Triton stores 1D; we store per row
    for i in range(0, BLOCK_M):
        if offs_m[i] < m:
            tl.store(lse_ptr + offs_m[i], lse_vals[i])


@triton.jit
def softmax_row_masked(masked_ptr, lse_ptr, soft_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Load masked logits
    masked = tl.load(masked_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=-float("inf"))
    # Load lse per row
    lse_row = tl.load(lse_ptr + offs_m, mask=(offs_m < m), other=0.0)[:, None]
    # Softmax: exp(masked - lse) / sum_exp
    shifted = masked - lse_row
    exps = tl.exp(shifted)
    sum_exp = tl.sum(exps, axis=1)  # [BLOCK_M]
    soft = exps / sum_exp[:, None]
    tl.store(soft_ptr + (offs_m[:, None] * n + offs_n[None, :]),
             soft, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


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
        a = tl.load(soft_ptr + (offs_m[:, None] * k + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k), other=0.0)
        b = tl.load(kc_ptr + (offs_k[:, None] * k + offs_n[None, :]),
                    mask=(offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0)
        acc += tl.dot(a, b)

    out_offsets = offs_m[:, None] * n + offs_n[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Extract device and constants
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64, "Shape assertions failed"
        num_pages = ckv_cache.shape[0]
        len_indptr = qo_indptr.numel()
        batch_size = len_indptr - 1  # number of batches

        # Prepare Kc_all and Kp_all on host: squeeze the singleton dim and cast to float32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output and lse
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch b from 0 to len_indptr-2
        for b in range(len_indptr - 1):
            # Compute q_start, q_end and L for this batch
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            # If nothing to process in this batch, skip
            if q_len <= 0:
                continue

            # Token indices for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg
            if L <= 0:
                continue

            # Gather Kc and Kp for these tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # 1-D
            Kc = Kc_all[tok_idx]  # [L, 512]
            Kp = Kp_all[tok_idx]  # [L, 64]

            # Slice q_nope and q_pe for this batch and each query
            # We need per-query qn and qp; loop over i
            for i in range(q_len):
                qn = q_nope[q_start + i].to(torch.float32)  # [16, 512]
                qp = q_pe[q_start + i].to(torch.float32)   # [16, 64]

                # Matmul qn @ Kc.T -> [16, L]
                logits_qn = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                matmul_qn_kc[(num_qo_heads, (L + 31) // 32,)](
                    qn, Kc, logits_qn, num_qo_heads, 512, L, 32, (L + 31) // 32, 64
                )

                # Matmul qp @ Kp.T -> [16, L]
                logits_qp = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                matmul_qp_kp[(num_qo_heads, (L + 31) // 32,)](
                    qp, Kp, logits_qp, num_qo_heads, 64, L, 32, (L + 31) // 32, 64
                )

                # Add logits
                logits = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                add_logits[(num_qo_heads, (L + 31) // 32,)](logits_qn, logits_qp, logits, num_qo_heads, L, 32, (L + 31) // 32)

                # Scale logits by sm_scale
                logits_scaled = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                scale_logits[(num_qo_heads, (L + 31) // 32,)](logits, logits_scaled, sm_scale, num_qo_heads, L, 32, (L + 31) // 32)

                # Apply causal-like mask: keep positions j where j > (L - q_len + 1 - i)
                # query_abs_pos = L - (q_len - 1) + i = L - q_len + 1 + i
                query_abs_pos = L - q_len + 1 + i
                logits_masked = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                apply_mask[(num_qo_heads, (L + 31) // 32,)](logits_scaled, logits_masked, query_abs_pos, num_qo_heads, L, 32, (L + 31) // 32)

                # Compute per-row lse = logsumexp(masked) / ln(2)
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                row_logsumexp_masked[(num_qo_heads,)](logits_masked, lse_row, num_qo_heads, L, 32)

                # Softmax per row
                soft = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                softmax_row_masked[(num_qo_heads, (L + 31) // 32,)](logits_masked, lse_row, soft, num_qo_heads, L, 32, (L + 31) // 32)

                # Output = soft @ Kc -> [16, 512]
                out_vec = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                matmul_attn_kc[(num_qo_heads, (head_dim_ckv + 31) // 32,)](
                    soft, Kc, out_vec, num_qo_heads, L, head_dim_ckv, 32, (head_dim_ckv + 31) // 32, 64
                )

                # Store results
                output[q_start + i] = out_vec.to(torch.bfloat16)
                lse[q_start + i] = lse_row / math.log(2.0)

        return output, lse


def run(*args):
    return ModelNew()(*args)
