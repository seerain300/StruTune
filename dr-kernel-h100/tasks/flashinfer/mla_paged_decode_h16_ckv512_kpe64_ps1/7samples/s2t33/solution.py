import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute output and lse for a single (b, h). It takes q_nope[b, h], q_pe[b, h],
# and Kc_all/Kp_all indexed by tok_idx. It loops over tokens to compute logits, then lse,
# attention vector, and final output vector for that head. All math is in fp32; output is bfloat16.
@triton.jit
def _single_head_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    tok_idx_ptr,
    out_ptr, lse_ptr,
    # sizes
    Dc: tl.constexpr, Dp: tl.constexpr, L_tokens: tl.constexpr,
    sm_scale: tl.float32
):
    # One program per (b, h); we assume grid=(B,H) when launching.
    # Triton doesn't allow reading b/h from program_id directly; we must pass them as args.
    # However, since we launch with grid=(B,H), we can infer b=pid0, h=pid1 via arguments passed as scalars.
    # To keep it simple, we'll rely on the caller passing q_nope_ptr and q_pe_ptr pointing to the right b,h.
    # Triton will receive out_ptr pointing to output[b,h,:] and lse_ptr pointing to lse[b,h].

    # We need b and h to compute base offsets for q_nope and q_pe. We'll get them by decomposing the pointer offsets
    # by using stride_b and stride_h of the original tensors; but since we already set q_nope_ptr/q_pe_ptr to be
    # the correct per-(b,h) slice, we can simply treat the pointers as starting at (b,h) row.

    # Prepare vectors for qn and qp (fp32 loads from bfloat16 pointers)
    # q_nope_ptr points to [Dc], q_pe_ptr points to [Dp]
    # We'll create qn and qp vectors as fp32
    qn = tl.zeros([Dc], dtype=tl.float32)
    # Load qn row: q_nope[b, h, :] in bfloat16, upcast to fp32
    # Triton pointer arithmetic: we assume q_nope_ptr points to the start of this row (already sliced on host).
    # However, Triton pointers are opaque; we can't load qn without host slicing. So we must pass qn separately.
    # To avoid torch ops in host, we won't pass qn here; we will read directly from q_nope_ptr. Triton supports
    # loading with tl.load, but we need to know the strides and shape. Since we've already sliced q_nope to per-(b,h),
    # we can just treat q_nope_ptr as pointing to the row [Dc]. Same for q_pe_ptr -> [Dp].

    # Simulate qn and qp loads:
    # We'll load qn from q_nope_ptr via tl.load with offsets 0..Dc-1
    # Create indices for load
    idx_qn = tl.arange(0, Dc)
    qn = tl.load(q_nope_ptr + idx_qn, mask=idx_qn < Dc, other=0.0).to(tl.float32)

    idx_qp = tl.arange(0, Dp)
    qp = tl.load(q_pe_ptr + idx_qp, mask=idx_qp < Dp, other=0.0).to(tl.float32)

    # Accumulate logits vector (fp32)
    logits = tl.zeros([L_tokens], dtype=tl.float32)

    # Loop over tokens t
    for t in tl.static_range(L_tokens):
        # Load Kc[t, :] and Kp[t, :]
        kc_row = tl.load(Kc_all_ptr + t * Dc + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        kp_row = tl.load(Kp_all_ptr + t * Dp + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        # Compute dot products: sum_i qn[i] * kc_row[i] and sum_j qp[j] * kp_row[j]
        dot_qn = 0.0
        for i in tl.static_range(Dc):
            dot_qn += qn[i] * kc_row[i]

        dot_qp = 0.0
        for j in tl.static_range(Dp):
            dot_qp += qp[j] * kp_row[j]

        logits[t] = dot_qn + dot_qp

    # Scale logits and compute lse (base-2)
    logits_scaled = logits * sm_scale
    # Compute max for numerical stability
    max_scaled = logits_scaled[0]
    for t in tl.static_range(L_tokens):
        if logits_scaled[t] > max_scaled:
            max_scaled = logits_scaled[t]

    sum_exp = 0.0
    for t in tl.static_range(L_tokens):
        sum_exp += tl.exp(logits_scaled[t] - max_scaled)

    lse_val = max_scaled + tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)

    # Compute attention vector
    attn = tl.zeros([L_tokens], dtype=tl.float32)
    for t in tl.static_range(L_tokens):
        attn[t] = tl.exp(logits_scaled[t] - lse_val) / 1.4426950408889634

    # Compute final output vector: out = sum_t attn[t] * Kc[t, :]
    out_vec = tl.zeros([Dc], dtype=tl.float32)
    for t in tl.static_range(L_tokens):
        kc_row = tl.load(Kc_all_ptr + t * Dc + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        out_vec += attn[t] * kc_row

    # Store output as bfloat16 (Triton will cast on store if needed)
    tl.store(out_ptr + tl.arange(0, Dc), out_vec.to(tl.bfloat16))

    # Store lse as float32
    tl.store(lse_ptr, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # q_nope: [B, H, Dc], q_pe: [B, H, Dp], ckv_cache: [num_pages, Dc], kpe_cache: [num_pages, Dp]
        # kv_indptr: [len_indptr], kv_indices: [num_kv_indices]
        # We must compute output [B, H, Dc] and lse [B, H]

        # Assumptions consistent with original code
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        assert q_nope.shape[2] == 512, "head_dim_ckv must be 512"
        assert q_pe.shape[2] == 64, "head_dim_kpe must be 64"
        assert ckv_cache.shape[1] == 1, "ckv_cache has been squeezed in original; use [num_pages, Dc]"
        assert kpe_cache.shape[1] == 1, "kpe_cache has been squeezed in original; use [num_pages, Dp]"

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]

        device = q_nope.device
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # Ensure all inputs are contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()

        # For each batch element, process tokens according to kv_indptr
        for b in range(B):
            # Compute token range for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(page_end - page_beg, 0)
            if L_tokens == 0:
                # No tokens for this batch; output zeros and lse = -inf
                lse[b] = -float("inf")
                # output[b, :, :] = 0
                # We don't need to call the kernel in this case
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.long).contiguous()

            # Prepare per-(b,h) slices for q_nope and q_pe pointers
            # Triton kernel expects pointers to qn and qp of length Dc and Dp respectively.
            # We'll call the kernel with grid=(1,), and pass q_nope[b,h,:] and q_pe[b,h,:] by slicing them.
            # But Triton launch grid needs (B,H); so we launch per (b,h).
            for h in range(H):
                # Slice q_nope and q_pe to per-head rows
                q_nope_row_ptr = q_nope[b, h, :].contiguous()  # [Dc]
                q_pe_row_ptr = q_pe[b, h, :].contiguous()     # [Dp]

                # Output and lse pointers for this (b,h)
                out_ptr = output[b, h, :].contiguous()  # [Dc] bfloat16
                lse_ptr = lse[b, h]                     # scalar float32

                # Launch Triton kernel: one program per (b,h)
                _single_head_kernel[(1,)](
                    q_nope_row_ptr, q_pe_row_ptr,
                    ckv_cache, kpe_cache,
                    tok_idx,
                    out_ptr, lse_ptr,
                    Dc=512, Dp=64, L_tokens=L_tokens,
                    sm_scale=sm_scale
                )

        return output, lse


def run(*args):
    return ModelNew()(*args)
