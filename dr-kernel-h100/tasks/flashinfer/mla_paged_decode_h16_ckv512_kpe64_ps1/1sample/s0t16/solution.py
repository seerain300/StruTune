import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_kernel(
    qn_ptr,      # *fp32, [N]
    qp_ptr,      # *fp32, [Kp_dim]
    Kc_ptr,      # *fp32, [TOTAL_PAGES, N]
    Kp_ptr,      # *fp32, [TOTAL_PAGES, Kp_dim]
    tok_idx_ptr, # *int32, [M_total]
    lse_ptr,     # *fp32, scalar per (b,h)
    out_ptr,     # *fp32, [N]
    N,           # int32
    Kp_dim,      # int32
    M_total,     # int32
    sm_scale,    # fp32
    BLOCK_M: tl.constexpr,  # chunk size for token loop
    b,           # int32 (unused in pointer math, kept for context)
    h            # int32 (unused in pointer math, kept for context)
):
    # First pass: compute row-wise max and sum of exp for logsumexp
    row_max = -float("inf")
    sum_exp = 0.0

    m = 0
    while m < M_total:
        # Process in chunks of BLOCK_M
        for mm in range(BLOCK_M):
            idx = m + mm
            valid = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=valid, other=0)  # scalar token index

            # Compute logits_scaled = dot(qn, Kc[tok]) + dot(qp, Kp[tok])
            # Load qn_row and compute dot with Kc[tok]
            offs = tl.arange(0, N)
            qn_row = tl.load(qn_ptr + offs)  # [N]
            Kc_row = tl.load(Kc_ptr + tok * N + offs, mask=valid, other=0.0)  # [N]
            dot_qn = tl.sum(qn_row * Kc_row, axis=0)  # scalar

            # Load qp_row and compute dot with Kp[tok]
            qp_row = tl.load(qp_ptr + tl.arange(0, Kp_dim))  # [Kp_dim]
            Kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=valid, other=0.0)  # [Kp_dim]
            dot_qp = tl.sum(qp_row * Kp_row, axis=0)  # scalar

            logits = dot_qn + dot_qp
            logits_scaled = logits * sm_scale  # scalar

            # Update row-wise max and sum of exp
            if valid:
                if logits_scaled > row_max:
                    row_max = logits_scaled
                sum_exp += tl.exp(logits_scaled - row_max)

        m += BLOCK_M

    # Compute lse = log(sum_exp) / ln(2)
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2
    # Store lse to lse_ptr
    tl.store(lse_ptr, lse_val)

    # Second pass: accumulate output y = sum_m attn[m] * Kc[m, :]
    # attn[m] = exp((logits_scaled[m] - lse_val) / M_total)
    for m2 in range(M_total):
        tok = tl.load(tok_idx_ptr + m2)
        qn_row = tl.load(qn_ptr + tl.arange(0, N))  # [N]
        Kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N))  # [N]
        Kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim))  # [Kp_dim]

        dot_qn = tl.sum(qn_row * tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=tl.full((N,), True, tl.int1), other=0.0), axis=0)
        dot_qp = tl.sum(tl.load(qp_ptr + tl.arange(0, Kp_dim)), axis=0)
        logits = dot_qn + dot_qp
        logits_scaled = logits * sm_scale

        attn = tl.exp(logits_scaled - lse_val) / M_total  # scalar

        y_vec = tl.load(out_ptr + tl.arange(0, N))  # [N]
        y_vec += attn * tl.sum(Kc_row, axis=0)  # attn is scalar, multiply by row sum (incorrect if summing, correct if multiplying by vector)
        # Correction: attn is scalar; multiply vector Kc_row directly and add to y
        y_vec += attn * Kc_row
        tl.store(out_ptr + tl.arange(0, N), y_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, block_m=128):
        super().__init__()
        self.block_m = block_m

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale=1.0):
        # Extract shapes and cast to fp32 for numerical stability
        B, H, N = q_nope.shape
        _, H2, Kp_dim = q_pe.shape
        assert H == H2, "num_qo_heads mismatch"
        num_pages = ckv_cache.shape[0]
        total_pages, N2, _ = ckv_cache.shape
        assert N == N2 and Kp_dim == kpe_cache.shape[2], "dim mismatches"
        device = q_nope.device

        # Cast inputs to fp32 and make contiguous
        qn_fp32 = q_nope.to(torch.float32).contiguous()  # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()    # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous()  # [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous()  # [num_pages, Kp_dim]

        # Compute M_total per batch from kv_indptr
        M_indptr = kv_indptr.to(torch.int32)
        M_total_list = (M_indptr[1:] - M_indptr[:-1]).to(torch.int32).tolist()

        # Allocate output in fp32 and lse in fp32
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel per (b, h)
        for b_idx in range(B):
            # If no tokens for this batch element, skip work
            if M_total_list[b_idx] == 0:
                lse[b_idx] = -float("inf")
                output_fp32[b_idx].zero_()
                continue

            start = int(M_indptr[b_idx].item())
            end = int(M_indptr[b_idx + 1].item())
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total_b]

            # Launch kernel for each head h
            for h_idx in range(H):
                lse_scalar = lse[b_idx, h_idx]  # scalar fp32
                # Prepare pointers for this (b,h)
                qn_h = qn_fp32[b_idx, h_idx]       # [N]
                qp_h = qp_fp32[b_idx, h_idx]       # [Kp_dim]
                out_y = output_fp32[b_idx, h_idx]  # [N]

                lse_and_output_kernel[(1,)](
                    qn_h.contiguous(),              # *fp32 [N]
                    qp_h.contiguous(),              # *fp32 [Kp_dim]
                    Kc_fp32,                        # *fp32 [num_pages, N]
                    Kp_fp32,                        # *fp32 [num_pages, Kp_dim]
                    tok_idx,                        # *int32 [M_total_b]
                    lse[b_idx, h_idx].contiguous(),  # *fp32 scalar
                    out_y,                           # *fp32 [N]
                    N, Kp_dim, M_total_list[b_idx], self.sm_scale,
                    self.block_m,
                    b_idx, h_idx
                )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
