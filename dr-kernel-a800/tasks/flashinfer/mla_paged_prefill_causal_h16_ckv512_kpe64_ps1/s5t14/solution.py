import torch
import triton
import triton.language as tl

# Triton kernels: all heavy computation is performed inside @triton.jit

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
        q = tl.load(
            qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        # kc[l, d] for l in offs_n, d in offs_k
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
def add_logits(a_ptr, b_ptr, c_ptr,
               H: tl.constexpr, L: tl.constexpr,
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
def scale_logits(in_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(in_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    y = x * scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             y,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def apply_mask(logits_ptr, mask_ptr, out_ptr, sm_scale, H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # mask_ptr contains -inf (as float) where invalid, otherwise 0.0
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
                   other=0.0)
    # apply mask: logits + mask (mask is -inf or 0)
    out = logits + mask
    out = out * sm_scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             out,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def row_logsumexp(logits_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr,
                  BLOCK_N: tl.constexpr):
    # Compute per-row max and sumexp in Triton
    m = H
    n = L

    row = tl.program_id(0)
    if row >= m:
        return

    offs_n = tl.arange(0, BLOCK_N)
    # Initialize max and sum
    neg_inf = -1.0e20
    max_val = neg_inf
    sumexp = 0.0

    # First pass: max
    for start in range(0, n, BLOCK_N):
        idx = start + offs_n
        vals = tl.load(logits_ptr + (row * L + idx),
                       mask=(idx < n),
                       other=neg_inf)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: sumexp
    for start in range(0, n, BLOCK_N):
        idx = start + offs_n
        vals = tl.load(logits_ptr + (row * L + idx),
                       mask=(idx < n),
                       other=neg_inf)
        vals_shift = vals - max_val
        exp_vals = tl.exp(vals_shift)
        exp_vals = tl.where(idx < n, exp_vals, 0.0)
        sumexp += tl.sum(exp_vals, axis=0)

    lse = tl.log(sumexp) + max_val
    # Write lse as float32
    tl.store(lse_ptr + row, lse)


@triton.jit
def softmax_row_masked(logits_ptr, lse_ptr, soft_ptr, H: tl.constexpr, L: tl.constexpr,
                       BLOCK_N: tl.constexpr):
    # soft[row, col] = exp(logits[row, col] - lse[row]) / sum_j exp(logits[row, j] - lse[row])
    m = H
    n = L

    row = tl.program_id(0)
    if row >= m:
        return

    neg_inf = -1.0e20
    offs_n = tl.arange(0, BLOCK_N)

    # Load lse for row
    lse = tl.load(lse_ptr + row)

    total_sum = 0.0
    # Compute denominator: sum of exp(logits - lse)
    for start in range(0, n, BLOCK_N):
        idx = start + offs_n
        vals = tl.load(logits_ptr + (row * L + idx),
                       mask=(idx < n),
                       other=neg_inf)
        vals_shift = vals - lse
        exp_vals = tl.exp(vals_shift)
        exp_vals = tl.where(idx < n, exp_vals, 0.0)
        total_sum += tl.sum(exp_vals, axis=0)

    # Write softmax values
    for start in range(0, n, BLOCK_N):
        idx = start + offs_n
        vals = tl.load(logits_ptr + (row * L + idx),
                       mask=(idx < n),
                       other=neg_inf)
        vals_shift = vals - lse
        exp_vals = tl.exp(vals_shift)
        exp_vals = tl.where(idx < n, exp_vals, 0.0)
        soft_vals = exp_vals / total_sum
        tl.store(soft_ptr + (row * L + idx),
                 soft_vals,
                 mask=(idx < n))


@triton.jit
def matmul_attn_kc(soft_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[h, d] = sum_l soft[h, l] * kc[l, d]
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
        soft = tl.load(
            soft_ptr + (offs_m[:, None] * L + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        kc_tile = tl.load(
            kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
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
        # Ensure tensors are on the same device
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda \
               and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors."

        # Prepare Kc_all and Kp_all: squeeze 1 dimension and cast to float32 for compute
        num_pages = ckv_cache.shape[0]
        D = q_nope.shape[-1]  # 512
        P = q_pe.shape[-1]    # 64

        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, P]

        total_q = int(qo_indptr[-1].item())
        H = q_nope.shape[1]  # 16
        output = torch.empty((total_q, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        # Loop over batches defined by qo_indptr
        # Note: iterate b from 0 to len(qo_indptr) - 2 to avoid out-of-bounds
        for b in range(0, qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len <= 0:
                continue

            # Slice kv_indices for this batch to get tok_idx
            # tok_idx = kv_indices[page_beg:page_end]; L = length of tok_idx
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            # Select the first q_len tokens from this segment
            tok_idx = kv_indices[page_beg:page_end][:q_len]  # 1-D long tensor on device
            L = tok_idx.shape[0]

            if L <= 0:
                continue

            # Kc and Kp for this batch
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Process each query i in the batch
            for i in range(q_len):
                qn = q_nope[q_start + i].to(torch.float32)  # [H, D]
                qp = q_pe[q_start + i].to(torch.float32)    # [H, P]

                # Compute logits[h, l] = qn[h] @ Kc[l, :]^T + qp[h] @ Kp[l, :]^T
                logits_qn = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_qp = torch.empty((H, L), dtype=torch.float32, device=device)
                logits = torch.empty((H, L), dtype=torch.float32, device=device)

                # Launch GEMMs for qn @ Kc.T and qp @ Kp.T
                # BLOCK sizes: H small (16), L can be large, D=512, P=64
                BLOCK_M = 16
                BLOCK_N = 64
                BLOCK_K_qn = 128
                BLOCK_K_qp = 64

                matmul_qn_kc[(H, L)](
                    qn, Kc, logits_qn,
                    H, D, L,
                    BLOCK_M, BLOCK_N, BLOCK_K_qn
                )
                matmul_qp_kp[(H, L)](
                    qp, Kp, logits_qp,
                    H, P, L,
                    BLOCK_M, BLOCK_N, BLOCK_K_qp
                )

                add_logits[(H, L)](
                    logits_qn, logits_qp, logits,
                    H, L,
                    BLOCK_M, BLOCK_N
                )

                # Scale logits
                logits_scaled = torch.empty((H, L), dtype=torch.float32, device=device)
                scale_logits[(H, L)](
                    logits, logits_scaled, sm_scale,
                    H, L,
                    BLOCK_M, BLOCK_N
                )

                # Apply causal mask: keep j where j > (L - q_len + i), else -inf
                # Compute query_abs_pos for this batch element
                query_abs_pos = (L - q_len) + i  # prefix of processed tokens + current query index
                # Build mask: [H, L] of -inf or 0.0
                mask = torch.empty((H, L), dtype=torch.float32, device=device)
                for h in range(H):
                    cols = torch.arange(L, device=device)
                    keep = cols > query_abs_pos
                    # For positions not kept, set to -inf
                    mask[h, :] = torch.where(keep, torch.zeros(L, dtype=torch.float32, device=device),
                                             torch.full((L,), -float("inf"), dtype=torch.float32, device=device))

                # Apply mask in Triton (out-of-place)
                masked_logits = torch.empty((H, L), dtype=torch.float32, device=device)
                apply_mask[(H, L)](
                    logits_scaled, mask, masked_logits,
                    sm_scale,
                    H, L,
                    BLOCK_M, BLOCK_N
                )

                # Row-wise logsumexp (numerical stability)
                lse_i = torch.empty((H,), dtype=torch.float32, device=device)
                row_logsumexp[(H)](
                    masked_logits,
                    lse_i,
                    H, L,
                    BLOCK_N=128
                )

                # Softmax per row
                softmax_out = torch.empty((H, L), dtype=torch.float32, device=device)
                softmax_row_masked[(H)](
                    masked_logits, lse_i, softmax_out,
                    H, L,
                    BLOCK_N=128
                )

                # Final output = softmax @ Kc
                out_row = torch.empty((H, D), dtype=torch.float32, device=device)
                matmul_attn_kc[(H, D)](
                    softmax_out, Kc, out_row,
                    H, L, D,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=128
                )

                # Store outputs
                output[q_start + i] = out_row.to(torch.bfloat16)
                lse[q_start + i] = (torch.logsumexp(masked_logits, dim=1) / math.log(2.0)).to(torch.float32)

        return output, lse


def run(*args):
    return ModelNew()(*args)
