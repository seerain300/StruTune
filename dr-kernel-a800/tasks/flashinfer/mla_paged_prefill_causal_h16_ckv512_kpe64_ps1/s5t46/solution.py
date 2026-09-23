import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: all heavy computations
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
            q = tl.load(
                qn_ptr + (offs_m[:, None] * k + offs_k[None, :]),
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0
            )
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
    def add_logits(a_ptr, b_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
        m = H
        n = L

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * 1 + tl.arange(0, 1)
        offs_n = pid_n * 1 + tl.arange(0, 1)
        # simple 1D reduction for this specific op; more blocks than necessary but safe
        for m0 in range(0, m):
            row_a = tl.load(a_ptr + m0 * L + offs_n, mask=(offs_n < n), other=0.0)
            row_b = tl.load(b_ptr + m0 * L + offs_n, mask=(offs_n < n), other=0.0)
            tl.store(out_ptr + m0 * L + offs_n, row_a + row_b, mask=(offs_n < n))
        # Note: For simplicity we parallelize over H and L via grid (H, L), but the above is minimal. Alternatively, we can use 2D tiled add too.
        # Here, we launch with grid=(H, L), and each program handles a single element add.


    @triton.jit
    def scale_logits(inp_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr):
        m = H
        n = L

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * 1 + tl.arange(0, 1)
        offs_n = pid_n * 1 + tl.arange(0, 1)
        for m0 in range(0, m):
            row = tl.load(inp_ptr + m0 * L + offs_n, mask=(offs_n < n), other=0.0)
            tl.store(out_ptr + m0 * L + offs_n, row * scale, mask=(offs_n < n))


    @triton.jit
    def apply_mask(logits_ptr, mask_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
        m = H
        n = L

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * 1 + tl.arange(0, 1)
        offs_n = pid_n * 1 + tl.arange(0, 1)
        for m0 in range(0, m):
            row = tl.load(logits_ptr + m0 * L + offs_n, mask=(offs_n < n), other=0.0)
            # mask: 1 means keep, 0 means set to -inf
            is_keep = tl.load(mask_ptr + m0 * L + offs_n, mask=(offs_n < n), other=1)
            # -inf constant
            neg_inf = -float("inf")
            row = tl.where(is_keep != 0, row, neg_inf)
            tl.store(out_ptr + m0 * L + offs_n, row, mask=(offs_n < n))


    @triton.jit
    def row_logsumexp(inp_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
        # out[h] = logsumexp(inp[h, :]) / ln(2)
        m = H
        n = L
        inv_ln2 = 1.4426950408889634  # 1 / ln(2)

        for h in range(0, m):
            # first pass: max
            maxv = -float("inf")
            for j in range(0, n):
                v = tl.load(inp_ptr + h * n + j)
                maxv = tl.maximum(maxv, v)
            # second pass: sumexp
            sumexp = 0.0
            for j in range(0, n):
                v = tl.load(inp_ptr + h * n + j)
                sumexp += tl.exp(v - maxv)
            lse = maxv + tl.log(sumexp) * inv_ln2
            tl.store(out_ptr + h, lse)


    @triton.jit
    def softmax_row_masked(inp_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
        m = H
        n = L
        inv_ln2 = 1.4426950408889634  # Not used here; lse_ptr is already computed in host kernel.

        for h in range(0, m):
            # Read lse[h] and compute denominator
            lse = tl.load(lse_ptr + h)
            # We need sum of exp(inp - lse) across j
            denom = 0.0
            for j in range(0, n):
                v = tl.load(inp_ptr + h * n + j)
                denom += tl.exp(v - lse)
            # Store softmax values
            for j in range(0, n):
                v = tl.load(inp_ptr + h * n + j)
                soft = tl.exp(v - lse) / denom
                tl.store(out_ptr + h * n + j, soft)


    @triton.jit
    def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr):
        # out[H, D] = attn[H, L] @ kc[L, D]
        m = H
        n = L
        k = D

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * 1 + tl.arange(0, 1)
        offs_k = pid_n * 1 + tl.arange(0, 1)
        # We'll implement a simple tiled matmul
        acc = tl.zeros((1, 1), dtype=tl.float32)
        for k0 in range(0, k, 64):
            offs_k_tile = k0 + tl.arange(0, 64)
            # attn row: load row h across L in chunks
            row_attn = tl.zeros((1,), dtype=tl.float32)
            for j0 in range(0, n, 64):
                offs_n_tile = j0 + tl.arange(0, 64)
                # load attn[h, j:j+64]
                attn_row_chunk = tl.load(
                    attn_ptr + h * n + offs_n_tile,
                    mask=(offs_n_tile < n),
                    other=0.0
                )
                # load kc[j:j+64, k:k+64] as [64, 64]
                kc_chunk = tl.load(
                    kc_ptr + (offs_n_tile[:, None] * k + offs_k_tile[None, :]),
                    mask=(offs_n_tile[:, None] < n) & (offs_k_tile[None, :] < k),
                    other=0.0
                )
                # attn_row_chunk is [64], kc_chunk is [64,64], dot over 64 to get new attn row values
                # We need to update acc with attn_row_chunk @ kc_chunk per j0 step? Instead, accumulate each column update.
                # To keep it simple, we loop over columns and update acc:
                # For each col in offs_k_tile:
                for col in range(64):
                    attn_val = attn_row_chunk[col]  # scalar
                    kc_col = kc_chunk[col, :]       # [64]
                    acc += attn_val * kc_col        # broadcast to [D]
            # After loop, acc is [1,1]? Actually, we need to store acc into out[h, :].
            # But above approach is too cumbersome. Let's do a 2D tiled matmul properly.
            # We'll re-implement with 2D tiling over D and L.

        # Instead of the above nested loops, implement a proper 2D tiled matmul:
        # Each program handles a tile [BLOCK_M, BLOCK_N] with BLOCK_K inner loop.
        # But since m=H and n=L are small per batch (e.g., up to a few tens), we can keep it simple and do:
        # We'll use grid (H, D) and loop over L in the kernel. This is fine for the given dimensions.

        # Re-implement proper matmul for this case:
        # We need to compute out[h, d] = sum_{l} attn[h, l] * kc[l, d]
        # We can do this by:
        # For each h: acc[D] = 0, then for l in range(0, L): load attn[h, l], then for d in range(0, D): acc[d] += attn[h,l] * kc[l,d].
        # Store acc to out[h, :].
        # Implement with loop-based approach here (H and D are constexpr).
        # Note: This kernel is simple and acceptable for the provided dimensions.

        # We'll iterate over l in chunks of 64 and over d in chunks of 64:
        # Initialize out vector for each h
        for h in range(0, m):
            out_vec = tl.zeros((D,), dtype=tl.float32)
            for j0 in range(0, n, 64):
                offs_n_tile = j0 + tl.arange(0, 64)
                attn_row = tl.load(
                    attn_ptr + h * n + offs_n_tile,
                    mask=(offs_n_tile < n),
                    other=0.0
                )
                # For each element in attn_row, multiply with kc chunk and accumulate
                # Loop over the 64 elements:
                for jj in range(64):
                    l_idx = offs_n_tile[jj]
                    a_val = attn_row[jj]
                    # load kc[l_idx, d] vector across d in chunks
                    for d0 in range(0, k, 64):
                        offs_d_tile = d0 + tl.arange(0, 64)
                        kc_vec = tl.load(
                            kc_ptr + (l_idx * k + offs_d_tile),
                            mask=(offs_d_tile < k),
                            other=0.0
                        )
                        # out_vec += a_val * kc_vec
                        out_vec += a_val * kc_vec
            # Store out_vec
            for d in range(0, k):
                tl.store(out_ptr + h * k + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # We must not use torch.* device math. All math will be in Triton kernels.
        # Prepare constants
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        # From original assertions
        num_qo_heads = num_qo_heads  # 16
        head_dim_ckv = head_dim_ckv   # 512
        head_dim_kpe = head_dim_kpe   # 64
        # Prepare Kc_all and Kp_all (host-side, no torch on device math)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Ensure indptrs are 1-D int32 tensors
        # qo_indptr and kv_indptr should already be provided as 1-D, but guard anyway
        if qo_indptr.dim() != 1 or kv_indptr.dim() != 1:
            # Convert to 1-D if needed
            qo_indptr = qo_indptr.reshape(-1)
            kv_indptr = kv_indptr.reshape(-1)

        len_indptr = qo_indptr.numel()

        for b in range(len_indptr - 1):
            # Compute q_start and q_end safely using .item()
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item()) if b + 1 < len_indptr else int(qo_indptr[-1].item())
            # Compute L for kv_indices
            if b + 1 < len_indptr:
                page_beg = int(kv_indptr[b].item())
                page_end = int(kv_indptr[b + 1].item())
            else:
                # If len_indptr is not strictly b+1, fallback to the last element
                page_beg = int(kv_indptr[b].item())
                page_end = int(kv_indptr[-1].item())
            L = max(page_end - page_beg, 0)
            # If no queries or no kv, continue
            if q_start >= q_end or L == 0:
                continue

            # Gather Kc and Kp for this batch's tokens
            tok_idx = kv_indices[page_beg:page_end]  # 1-D long
            Kc = Kc_all[tok_idx]  # [L, 512]
            Kp = Kp_all[tok_idx]  # [L, 64]

            # For each query i
            for i in range(q_end - q_start):
                h = num_qo_heads
                D = head_dim_ckv
                P = head_dim_kpe

                # Prepare qn and qp as float32 contiguous (device tensors, no torch math)
                qn = q_nope[q_start + i].contiguous().to(torch.float32)  # [H, D]
                qp = q_pe[q_start + i].contiguous().to(torch.float32)   # [H, P]

                # Allocate intermediates
                logits_qn = torch.empty((h, L), dtype=torch.float32, device=device)
                logits_qp = torch.empty((h, L), dtype=torch.float32, device=device)
                logits = torch.empty((h, L), dtype=torch.float32, device=device)
                logits_scaled = torch.empty((h, L), dtype=torch.float32, device=device)
                attn = torch.empty((h, L), dtype=torch.float32, device=device)
                out_row = torch.empty((D,), dtype=torch.float32, device=device)

                # Launch matmul kernels
                # For qn @ Kc.T -> [H, L]
                # We'll implement a simple 1D launch (grid) for clarity. Triton requires grid tuples, so we use (H, L) programs via 1 per element loops in kernels isn't practical here. Instead, we compute using Triton matmul with tiling over H and L.
                # However, Triton kernels expect proper grid; since H and L are small in this batch, we can launch with grid size (H, L).
                # But to avoid confusion, we implement full matmul loops in Triton kernels above. Here, we simply call the matmul kernels by reshaping pointers logically. Since Triton kernels operate on flattened contiguous memory, we can pass pointers and sizes and rely on the kernels to handle tiling internally via BLOCK sizes.
                # For simplicity, we assume BLOCK sizes are chosen; Triton will handle tiling.

                # Invoke matmul_qn_kc
                # We need to call Triton with proper pointers; Triton will cover H x L results by tiling over H and L.
                # The above kernels are defined to handle H, D, L as constexpr-like dims; here, we pass actual sizes.
                # Launch grid: (H, L) as (h, L)
                matmul_qn_kc[(h, L)](qn, Kc, logits_qn, h, D, L, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)
                # Launch matmul_qp_kp
                matmul_qp_kp[(h, L)](qp, Kp, logits_qp, h, P, L, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)
                # Add
                add_logits[(h, L)](logits_qn, logits_qp, logits, h, L)
                # Scale
                scale_logits[(h, L)](logits, logits_scaled, sm_scale, h, L)

                # Compute query_abs_pos for mask
                # prefix_len = L - (q_end - q_start) + i
                prefix_len = L - (q_end - q_start) + i
                # Mask: keep if j > prefix_len else -inf
                mask = torch.ones((h, L), dtype=torch.uint8, device=device)
                for h0 in range(h):
                    for j in range(L):
                        if j <= prefix_len:
                            mask[h0, j] = 0  # means set to -inf
                # Apply mask
                apply_mask[(h, L)](logits_scaled, mask, attn, h, L)

                # Compute lse per head
                # We need to compute lse[h] = logsumexp(attn[h, :]) / ln(2)
                # We implement a Triton kernel that reads attn row by row and computes max and sumexp.
                # Note: Triton kernels are defined above; we call row_logsumexp
                # We pass attn and out buffer for lse (length H)
                row_logsumexp[(h,)](attn, lse[q_start + i], h, L)

                # Softmax per row using lse
                # We need to compute softmax[h, j] = exp(attn[h, j] - lse[h]) / sum_j exp(attn - lse[h])
                softmax_row_masked[(h, L)](attn, lse[q_start + i], attn, h, L)
                # attn is modified in-place to softmax values

                # Final output: attn @ Kc -> [H, D]
                matmul_attn_kc[(h, D)](attn, Kc, out_row, h, L, D)

                # Store output row
                # output[q_start + i, :, :] = out_row.to(torch.bfloat16)
                output[q_start + i, :, :] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
