import torch
import math
import triton
import triton.language as tl


@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 qn_stride0, qn_stride1,
                 kc_stride0, kc_stride1,
                 out_stride0, out_stride1,
                 num_warps: tl.constexpr):
    # Each program computes one output row (head h) across tiles of L
    h = tl.program_id(0)  # [0..H)
    tile_l = tl.program_id(1)
    offs_l = tile_l * 128 + tl.arange(0, 128)
    mask_l = offs_l < L

    acc = tl.zeros((128,), dtype=tl.float32)

    # Reduction over K = D
    for k in range(0, D, 128):
        offs_k = k + tl.arange(0, 128)
        # qn[h, k] -> shape (128,)
        q_sub = tl.load(qn_ptr + h * qn_stride0 + offs_k * qn_stride1, mask=offs_k < D, other=0.0)
        # kc[offs_l, k] -> shape (128, 128)
        kc_sub = tl.load(kc_ptr + offs_l[:, None] * kc_stride0 + offs_k[None, :] * kc_stride1,
                         mask=mask_l[:, None] & (offs_k[None, :] < D),
                         other=0.0)
        # acc += q_sub^T @ kc_sub
        acc += tl.sum(q_sub[None, :] * kc_sub, axis=1)

    tl.store(out_ptr + h * out_stride0 + offs_l * out_stride1, acc, mask=mask_l)


@triton.jit
def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                 H: tl.constexpr, P: tl.constexpr, L: tl.constexpr,
                 qp_stride0, qp_stride1,
                 kp_stride0, kp_stride1,
                 out_stride0, out_stride1,
                 num_warps: tl.constexpr):
    h = tl.program_id(0)
    tile_l = tl.program_id(1)
    offs_l = tile_l * 128 + tl.arange(0, 128)
    mask_l = offs_l < L

    acc = tl.zeros((128,), dtype=tl.float32)

    for k in range(0, P, 128):
        offs_k = k + tl.arange(0, 128)
        q_sub = tl.load(qp_ptr + h * qp_stride0 + offs_k * qp_stride1, mask=offs_k < P, other=0.0)
        kp_sub = tl.load(kp_ptr + offs_l[:, None] * kp_stride0 + offs_k[None, :] * kp_stride1,
                         mask=mask_l[:, None] & (offs_k[None, :] < P),
                         other=0.0)
        acc += tl.sum(q_sub[None, :] * kp_sub, axis=1)

    tl.store(out_ptr + h * out_stride0 + offs_l * out_stride1, acc, mask=mask_l)


@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr,
               H: tl.constexpr, L: tl.constexpr,
               a_stride0, a_stride1,
               b_stride0, b_stride1,
               out_stride0, out_stride1,
               num_warps: tl.constexpr):
    h = tl.program_id(0)
    tile_l = tl.program_id(1)
    offs_l = tile_l * 128 + tl.arange(0, 128)
    mask_l = offs_l < L
    a = tl.load(a_ptr + h * a_stride0 + offs_l * a_stride1, mask=mask_l, other=0.0)
    b = tl.load(b_ptr + h * b_stride0 + offs_l * b_stride1, mask=mask_l, other=0.0)
    tl.store(out_ptr + h * out_stride0 + offs_l * out_stride1, a + b, mask=mask_l)


@triton.jit
def scale_logits(inp_ptr, out_ptr,
                 H: tl.constexpr, L: tl.constexpr,
                 inp_stride0, inp_stride1,
                 out_stride0, out_stride1,
                 scale: tl.constexpr,
                 num_warps: tl.constexpr):
    h = tl.program_id(0)
    tile_l = tl.program_id(1)
    offs_l = tile_l * 128 + tl.arange(0, 128)
    mask_l = offs_l < L
    x = tl.load(inp_ptr + h * inp_stride0 + offs_l * inp_stride1, mask=mask_l, other=0.0)
    y = x * scale
    tl.store(out_ptr + h * out_stride0 + offs_l * out_stride1, y, mask=mask_l)


@triton.jit
def apply_mask(inp_ptr, out_ptr,
               H: tl.constexpr, L: tl.constexpr,
               inp_stride0, inp_stride1,
               out_stride0, out_stride1,
               query_abs_pos: tl.constexpr,
               num_warps: tl.constexpr):
    # Each program handles one row h; tile over L
    h = tl.program_id(0)
    tile_l = tl.program_id(1)
    offs_l = tile_l * 128 + tl.arange(0, 128)
    mask_l = offs_l < L
    x = tl.load(inp_ptr + h * inp_stride0 + offs_l * inp_stride1, mask=mask_l, other=0.0)
    keep = offs_l > query_abs_pos
    x = tl.where(keep, x, -float('inf'))
    tl.store(out_ptr + h * out_stride0 + offs_l * out_stride1, x, mask=mask_l)


@triton.jit
def row_logsumexp(inp_ptr, lse_ptr,
                  H: tl.constexpr, L: tl.constexpr,
                  inp_stride0, inp_stride1,
                  ln2: tl.constexpr,
                  num_warps: tl.constexpr):
    # Assumes inp_ptr points to a single row [L] for each head; grid is (H,). We implement per-row.
    h = tl.program_id(0)
    # Pass 1: compute max
    max_val = -float('inf')
    for j in range(0, L):
        x = tl.load(inp_ptr + j * inp_stride1)
        if x > max_val:
            max_val = x
    # Pass 2: compute sumexp
    sumexp = 0.0
    for j in range(0, L):
        x = tl.load(inp_ptr + j * inp_stride1)
        sumexp += tl.exp(x - max_val)
    lse_val = max_val + tl.log(sumexp)
    # Divide by ln(2)
    lse_val = lse_val / ln2
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def softmax_row_triton(inp_ptr, lse_row: tl.constexpr, out_ptr, L: tl.constexpr, num_warps: tl.constexpr):
    # Compute softmax for a single row of length L using lse_row (logsumexp of that row).
    # Phase 1: compute denominator = sum_j exp(x_j - lse_row)
    denom = 0.0
    for j in range(0, L):
        x = tl.load(inp_ptr + j)
        denom += tl.exp(x - lse_row)
    # Phase 2: write normalized outputs
    inv_denom = 1.0 / denom
    for j in range(0, L):
        x = tl.load(inp_ptr + j)
        y = tl.exp(x - lse_row) * inv_denom
        tl.store(out_ptr + j, y)


@triton.jit
def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                   attn_stride0, attn_stride1,
                   kc_stride0, kc_stride1,
                   out_stride0, out_stride1,
                   num_warps: tl.constexpr):
    # Each program computes one output row (head h) across tiles of D
    h = tl.program_id(0)  # [0..H)
    tile_d = tl.program_id(1)
    offs_d = tile_d * 128 + tl.arange(0, 128)
    mask_d = offs_d < D

    acc = tl.zeros((128,), dtype=tl.float32)

    # Reduction over K = L
    for k in range(0, L, 128):
        offs_k = k + tl.arange(0, 128)
        mask_k = offs_k < L
        attn_sub = tl.load(attn_ptr + h * attn_stride0 + offs_k * attn_stride1, mask=mask_k, other=0.0)  # (128,)
        kc_sub = tl.load(kc_ptr + offs_d[:, None] * kc_stride0 + offs_k[None, :] * kc_stride1,
                         mask=mask_d[:, None] & mask_k[None, :],
                         other=0.0)  # (128, 128)
        acc += tl.sum(attn_sub[None, :] * kc_sub, axis=1)  # (128,)

    tl.store(out_ptr + h * out_stride0 + offs_d * out_stride1, acc, mask=mask_d)


def ModelNew():
    class _ModelNew(torch.nn.Module):
        def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
            device = q_nope.device
            total_q, H, D = q_nope.shape
            _, _, P = q_pe.shape
            num_pages = ckv_cache.shape[0]

            # Prepare Kc_all and Kp_all (squeeze batch dim, cast to float32 for compute)
            Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
            Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, P]

            # Output and lse
            output = torch.empty((total_q, H, D), dtype=torch.float32, device=device)
            lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

            # Process each batch element
            for b in range(qo_indptr.shape[0] - 1):
                q_start = int(qo_indptr[b].item())
                q_end = int(qo_indptr[b + 1].item())
                if q_start >= q_end:
                    continue

                # Get K tokens for this batch
                page_beg = int(kv_indptr[b].item())
                page_end = int(kv_indptr[b + 1].item())
                if page_beg >= page_end:
                    continue

                tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [L]
                L = tok_idx.numel()
                Kc = Kc_all[tok_idx]  # [L, D]
                Kp = Kp_all[tok_idx]  # [L, P]

                # Slice batched queries
                q_nope_batch = q_nope[q_start:q_end].to(torch.float32).contiguous()  # [Q, H, D]
                q_pe_batch = q_pe[q_start:q_end].to(torch.float32).contiguous()     # [Q, H, P]
                Q = q_nope_batch.shape[0]

                for i in range(Q):
                    qn = q_nope_batch[i]  # [H, D]
                    qp = q_pe_batch[i]    # [H, P]

                    # Allocate intermediates
                    A = torch.empty((H, L), dtype=torch.float32, device=device)
                    B = torch.empty((H, L), dtype=torch.float32, device=device)
                    Logits = torch.empty((H, L), dtype=torch.float32, device=device)
                    Scaled = torch.empty((H, L), dtype=torch.float32, device=device)
                    Masked = torch.empty((H, L), dtype=torch.float32, device=device)
                    SoftmaxOut = torch.empty((H, L), dtype=torch.float32, device=device)

                    # Compute A = qn @ Kc.T
                    grid_A = (H, (L + 128 - 1) // 128)
                    matmul_qn_kc[grid_A](
                        qn, Kc, A,
                        H, D, L,
                        qn.stride(0), qn.stride(1),
                        Kc.stride(0), Kc.stride(1),
                        A.stride(0), A.stride(1),
                        num_warps=4
                    )

                    # Compute B = qp @ Kp.T
                    grid_B = (H, (L + 128 - 1) // 128)
                    matmul_qp_kp[grid_B](
                        qp, Kp, B,
                        H, P, L,
                        qp.stride(0), qp.stride(1),
                        Kp.stride(0), Kp.stride(1),
                        B.stride(0), B.stride(1),
                        num_warps=4
                    )

                    # Add
                    grid_add = (H, (L + 128 - 1) // 128)
                    add_logits[grid_add](
                        A, B, Logits,
                        H, L,
                        A.stride(0), A.stride(1),
                        B.stride(0), B.stride(1),
                        Logits.stride(0), Logits.stride(1),
                        num_warps=4
                    )

                    # Scale
                    grid_scale = (H, (L + 128 - 1) // 128)
                    scale_logits[grid_scale](
                        Logits, Scaled,
                        H, L,
                        Logits.stride(0), Logits.stride(1),
                        Scaled.stride(0), Scaled.stride(1),
                        sm_scale,
                        num_warps=4
                    )

                    # Apply causal mask: j > (L - Q + i)
                    prefix_len = L - Q
                    query_abs_pos = prefix_len + i
                    grid_mask = (H, (L + 128 - 1) // 128)
                    apply_mask[grid_mask](
                        Scaled, Masked,
                        H, L,
                        Scaled.stride(0), Scaled.stride(1),
                        Masked.stride(0), Masked.stride(1),
                        query_abs_pos,
                        num_warps=4
                    )

                    # Row-wise logsumexp and write to lse[q_start + i]
                    ln2 = math.log(2.0)
                    grid_lse = (H,)
                    row_logsumexp[grid_lse](
                        Masked, lse[q_start + i],
                        H, L,
                        Masked.stride(0), Masked.stride(1),
                        ln2,
                        num_warps=1
                    )

                    # Softmax per row using lse
                    softmax_row_triton[(H,)](Masked, float(lse[q_start + i].item()), SoftmaxOut, L, num_warps=4)

                    # Final matmul: SoftmaxOut[H, L] @ Kc[L, D] -> [H, D]
                    OutRow = torch.empty((H, D), dtype=torch.float32, device=device)
                    grid_final = (H, (D + 128 - 1) // 128)
                    matmul_attn_kc[grid_final](
                        SoftmaxOut, Kc, OutRow,
                        H, D, L,
                        SoftmaxOut.stride(0), SoftmaxOut.stride(1),
                        Kc.stride(0), Kc.stride(1),
                        OutRow.stride(0), OutRow.stride(1),
                        num_warps=4
                    )

                    # Store to output
                    output[q_start + i] = OutRow  # overwrite row

            # Cast output to bfloat16 to match original
            output = output.to(torch.bfloat16)
            return output, lse

    return _ModelNew


def run(*args):
    return ModelNew()(*args)
