import math
import torch
import triton
import triton.language as tl


@triton.jit
def fused_qk_two_src_kernel(
    q_nope_ptr,   # *fp32, [H, D_ckv]
    q_pe_ptr,     # *fp32, [H, D_kpe]
    Kc_ptr,       # *fp32, [L, D_ckv]
    Kp_ptr,       # *fp32, [L, D_kpe]
    out_ptr,      # *fp32, [H, L]
    H,            # int32, number of heads
    L,            # int32, number of KV rows
    D_ckv,        # int32
    D_kpe,        # int32
    h_stride_qn,  # int32, stride between heads in q_nope (row stride)
    h_stride_qp,  # int32, stride between heads in q_pe (row stride)
    Kc_stride0,   # int32, stride0 of Kc
    Kc_stride1,   # int32, stride1 of Kc
    Kp_stride0,   # int32, stride0 of Kp
    Kp_stride1,   # int32, stride1 of Kp
    BLOCK_H: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Grid: (pid_h, pid_l) where pid_h tiles heads, pid_l tiles L
    pid_h = tl.program_id(0)
    pid_l = tl.program_id(1)

    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)  # [BLOCK_L]
    mask_h = hs < H
    mask_l = ls < L

    # Load q_nope rows for this head tile: shape [BLOCK_H, D_ckv]
    qn_rows = tl.load(
        q_nope_ptr + hs[:, None] * h_stride_qn + tl.arange(0, D_ckv)[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )  # [BLOCK_H, D_ckv]

    # Load q_pe rows for this head tile: shape [BLOCK_H, D_kpe]
    qp_rows = tl.load(
        q_pe_ptr + hs[:, None] * h_stride_qp + tl.arange(0, D_kpe)[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )  # [BLOCK_H, D_kpe]

    # Load Kc tile: shape [BLOCK_L, D_ckv]
    Kc_tile = tl.load(
        Kc_ptr + ls[:, None] * Kc_stride0 + tl.arange(0, D_ckv)[None, :] * Kc_stride1,
        mask=mask_l[:, None],
        other=0.0,
    )  # [BLOCK_L, D_ckv]

    # Load Kp tile: shape [BLOCK_L, D_kpe]
    Kp_tile = tl.load(
        Kp_ptr + ls[:, None] * Kp_stride0 + tl.arange(0, D_kpe)[None, :] * Kp_stride1,
        mask=mask_l[:, None],
        other=0.0,
    )  # [BLOCK_L, D_kpe]

    # Compute matmuls:
    # qn_rows: [BLOCK_H, D_ckv] @ Kc_tile.T: [D_ckv, BLOCK_L] -> [BLOCK_H, BLOCK_L]
    acc_qn = tl.dot(qn_rows, tl.trans(Kc_tile))
    # qp_rows: [BLOCK_H, D_kpe] @ Kp_tile.T: [D_kpe, BLOCK_L] -> [BLOCK_H, BLOCK_L]
    acc_qp = tl.dot(qp_rows, tl.trans(Kp_tile))
    logits = acc_qn + acc_qp  # [BLOCK_H, BLOCK_L]

    # Store results
    out_ptrs = out_ptr + hs[:, None] * L + ls[None, :]  # [BLOCK_H, BLOCK_L]
    store_mask = mask_h[:, None] & mask_l[None, :]
    tl.store(out_ptrs, logits, mask=store_mask)


def _fused_qk_two_src_triton(q_nope_row, q_pe_row, Kc, Kp):
    """
    Fused q @ Kc.T + qpe @ Kp.T using Triton.
    q_nope_row: [H, D_ckv] contiguous float32
    q_pe_row:   [H, D_kpe] contiguous float32
    Kc: [L, D_ckv] contiguous float32
    Kp: [L, D_kpe] contiguous float32
    Returns logits [H, L] float32
    """
    assert q_nope_row.is_cuda and q_pe_row.is_cuda and Kc.is_cuda and Kp.is_cuda, "All tensors must be on CUDA"
    H = q_nope_row.shape[0]
    L = Kc.shape[0]
    D_ckv = q_nope_row.shape[1]
    D_kpe = q_pe_row.shape[1]
    out = torch.empty((H, L), dtype=torch.float32, device=q_nope_row.device)

    # Launch grid over H and L tiles
    grid = (triton.cdiv(H, 16), triton.cdiv(L, 64))  # tile sizes tuned for H=16, L up to a few thousand
    fused_qk_two_src_kernel[grid](
        q_nope_row, q_pe_row, Kc, Kp, out,
        H=H, L=L, D_ckv=D_ckv, D_kpe=D_kpe,
        h_stride_qn=q_nope_row.stride(0),
        h_stride_qp=q_pe_row.stride(0),
        Kc_stride0=Kc.stride(0), Kc_stride1=Kc.stride(1),
        Kp_stride0=Kp.stride(0), Kp_stride1=Kp.stride(1),
        BLOCK_H=16, BLOCK_L=64,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA device"
        assert qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Indptr and indices must be on CUDA device"

        # Shapes (assumptions per original code)
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        assert ckv_cache.shape[-1] == 1 and kpe_cache.shape[-1] == 1, "Caches must have [num_pages, 1, dim]"
        assert ckv_cache.shape[2] == head_dim_ckv
        assert kpe_cache.shape[2] == head_dim_kpe
        assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64, "These dimensions are fixed per original code"

        # Prepare Kc_all and Kp_all: [num_pages, dim], float32, contiguous
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        # Output buffers
        output = torch.zeros((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Compute batch_size and len_indptr (as in original)
        batch_size = int(kv_indptr.numel() - 1)
        len_indptr = qo_indptr.numel()

        # For each batch element, process queries and KVs
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            # Token indices within this batch's KV block
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # [kv_len]
            # Select Kc and Kp rows for this batch
            Kc_batch = Kc_all[tok_idx]  # [kv_len, 512]
            Kp_batch = Kp_all[tok_idx]  # [kv_len, 64]

            # Process each query in this batch element
            q_len = q_end - q_start
            for i in range(q_len):
                q_row_i = q_nope[q_start + i].contiguous().to(torch.float32)  # [16, 512]
                qpe_row_i = q_pe[q_start + i].contiguous().to(torch.float32)  # [16, 64]

                # Fused Triton computation of logits = q_row_i @ Kc_batch.T + qpe_row_i @ Kp_batch.T
                logits = _fused_qk_two_src_triton(q_row_i, qpe_row_i, Kc_batch, Kp_batch)  # [16, kv_len]
                logits_scaled = logits * sm_scale  # scale by scalar

                # Causal mask: positions with idx > (kv_len - q_len + i) are -inf
                query_abs_pos = kv_len - q_len + i
                causal_mask = torch.arange(kv_len, device=logits_scaled.device) > query_abs_pos
                logits_scaled.masked_fill_(causal_mask.unsqueeze(0), -float("inf"))

                # Compute LSE and softmax along L (dim=-1) per head
                lse_i = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)  # [16]
                lse[q_start + i] = lse_i  # [16], keep shape (16,) for that query position

                attn = torch.softmax(logits_scaled, dim=-1)  # [16, kv_len]
                out_i = attn @ Kc_batch  # [16, 512]
                output[q_start + i] = out_i.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
