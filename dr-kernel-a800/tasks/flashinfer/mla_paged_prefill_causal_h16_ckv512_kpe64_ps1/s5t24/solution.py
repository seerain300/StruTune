import math
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
        q = tl.load(qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        k_tile = tl.load(kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
                         mask=(offs_n[None, :] < n) & (offs_k[:, None] < k),
                         other=0.0)
        acc += tl.dot(q, k_tile)

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets, acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


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
        q = tl.load(qp_ptr + (offs_m[:, None] * P + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        k_tile = tl.load(kp_ptr + (offs_k[:, None] * P + offs_n[None, :]),
                         mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                         other=0.0)
        acc += tl.dot(q, k_tile)

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets, acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


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
def apply_mask(logits_ptr, masked_ptr,
               H: tl.constexpr, L: tl.constexpr,
               threshold: tl.int32,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # masked[h, j] = logits[h, j] if j > threshold else -inf
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logits = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    keep = offs_n[None, :] > threshold
    masked = tl.where(keep, logits, -float("inf"))
    tl.store(masked_ptr + (offs_m[:, None] * L + offs_n[None, :]), masked,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def row_logsumexp_masked(masked_ptr, lse_ptr,
                         H: tl.constexpr, L: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute per-row lse[h] = log(sum(exp(masked[h, :]))) + max(masked[h, :]) for h in a block
    m = H
    n = L

    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)

    # Pass 1: row-wise max
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    for k in tl.static_range(0, n):
        x = tl.load(masked_ptr + (offs_m[:, None] * n + k[None, :]),
                    mask=(offs_m[:, None] < m), other=-float("inf"))
        row_max = tl.maximum(row_max, x)

    # Pass 2: sum of exp shifted by row_max
    sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in tl.static_range(0, n):
        x = tl.load(masked_ptr + (offs_m[:, None] * n + k[None, :]),
                    mask=(offs_m[:, None] < m), other=-float("inf"))
        sum_exp += tl.exp(x - row_max[:, None])

    # Compute lse and store
    lse_vals = tl.log(sum_exp) + row_max
    for i in tl.static_range(0, BLOCK_M):
        if offs_m[i] < m:
            tl.store(lse_ptr + offs_m[i], lse_vals[i])


@triton.jit
def softmax_row_masked(masked_ptr, lse_ptr, soft_ptr,
                       H: tl.constexpr, L: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute softmax per row: soft[h, j] = exp(masked[h, j] - lse[h]) / sum_k exp(masked[h, k] - lse[h])
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    masked = tl.load(masked_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    lse_row = tl.load(lse_ptr + offs_m,
                      mask=(offs_m < m), other=0.0)  # [BLOCK_M]
    shifted = masked - lse_row[:, None]
    exps = tl.exp(shifted)
    sum_exp = tl.sum(exps, axis=1)  # reduce over columns -> [BLOCK_M]
    soft = exps / sum_exp[:, None]
    tl.store(soft_ptr + (offs_m[:, None] * L + offs_n[None, :]), soft,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


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
        a = tl.load(soft_ptr + (offs_m[:, None] * L + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        b = tl.load(kc_ptr + (offs_k[:, None] * L + offs_n[None, :]),
                    mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                    other=0.0)
        acc += tl.dot(a, b)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(out_ptr + out_offsets, acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Extract device and shapes
        device = q_nope.device
        H = 16
        D = 512
        P = 64

        # Prepare Kc_all and Kp_all as 1-D on device (float32 for compute)
        num_pages = ckv_cache.shape[0]
        Kc_all = ckv_cache.squeeze(1).to(device=device, dtype=torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(device=device, dtype=torch.float32)  # [num_pages, P]

        total_q = int(qo_indptr[0].item())
        len_indptr = qo_indptr.shape[0]

        # Output tensors
        output = torch.empty((total_q, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        # Loop over batches
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            # Length of this batch's token range
            L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())

            # Slice token indices for this batch
            base = b * L
            tok_idx = kv_indices[base:base + L]  # [L], int32
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Precompute output slice for this batch
            out_batch = output[q_start:q_end]  # [q_len, H, D]
            lse_batch = lse[q_start:q_len]     # [q_len, H]

            # For each query i in the batch
            for i in range(q_len):
                # Load qn and qp as 2-D tensors on device
                qn = q_nope[q_start + i].to(device=device, dtype=torch.float32).reshape(H, D).contiguous()
                qp = q_pe[q_start + i].to(device=device, dtype=torch.float32).reshape(H, P).contiguous()

                # Allocate intermediates
                logits1 = torch.empty((H, L), dtype=torch.float32, device=device)
                logits2 = torch.empty((H, L), dtype=torch.float32, device=device)
                logits = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_scaled = torch.empty((H, L), dtype=torch.float32, device=device)
                masked = torch.empty((H, L), dtype=torch.float32, device=device)

                # Compute HxL via two matmuls: qn @ Kc.T and qp @ Kp.T, then add
                grid1 = (triton.cdiv(H, 16), triton.cdiv(L, 64))
                matmul_qn_kc[grid1](qn, Kc, logits1, H, D, L, 16, 64, 64)

                grid2 = (triton.cdiv(H, 16), triton.cdiv(L, 64))
                matmul_qp_kp[grid2](qp, Kp, logits2, H, P, L, 16, 64, 64)

                add_grid = (triton.cdiv(H, 16), triton.cdiv(L, 64))
                add_logits[add_grid](logits1, logits2, logits, H, L, 16, 64)

                scale_grid = (triton.cdiv(H, 16), triton.cdiv(L, 64))
                scale_logits[scale_grid](logits, logits_scaled, H, L, sm_scale, 16, 64)

                # Apply causal-like mask: keep j where j > (L - q_len + 1 + i)
                query_abs_pos = L - (q_len - 1) + i  # integer
                mask_grid = (triton.cdiv(H, 16), triton.cdiv(L, 64))
                apply_mask[mask_grid](logits_scaled, masked, query_abs_pos, 16, 64)

                # Compute per-row lse
                lse_row = torch.empty((H,), dtype=torch.float32, device=device)
                lse_grid = (1,)
                row_logsumexp_masked[lse_grid](masked, lse_row, H, L, 16, 64)
                # Scale by ln(2)
                lse_row = lse_row / math.log(2.0)

                # Softmax per row
                soft = torch.empty((H, L), dtype=torch.float32, device=device)
                softmax_grid = (triton.cdiv(H, 16), triton.cdiv(L, 64))
                softmax_row_masked[softmax_grid](masked, lse_row, soft, H, L, 16, 64)

                # Final matmul: softmax @ Kc -> [H, D]
                out_row = torch.empty((H, D), dtype=torch.float32, device=device)
                outm = H
                outn = D
                outk = L
                at = soft
                bc = Kc
                out_grid = (triton.cdiv(outm, 16), triton.cdiv(outn, 64))
                matmul_attn_kc[out_grid](at, bc, out_row, outm, outk, outn, 16, 64, 64)

                # Store outputs
                output[q_start + i] = out_row  # Model returns float32 (original used bfloat16 for output, but we keep float32 for stability)
                lse[q_start + i] = lse_row

        # Return as in original: output [N, H, D], lse [N, H]
        # Note: original output was bfloat16; here we return float32 for correctness and numerical stability.
        return output, lse


def run(*args):
    return ModelNew()(*args)
