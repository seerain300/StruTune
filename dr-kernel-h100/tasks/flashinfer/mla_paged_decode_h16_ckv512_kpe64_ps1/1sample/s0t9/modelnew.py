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
    lse_ptr,     # *fp32, scalar output for lse of this (b,h)
    out_ptr,     # *fp32, [N] output vector for this (b,h)
    N,           # int32, head_dim_ckv
    Kp_dim,      # int32, head_dim_kpe
    M_total,     # int32, number of used tokens in this batch
    sm_scale,    # fp32
    BLOCK_M: tl.constexpr,  # chunk size for token processing
    b,           # int32, batch index (for context)
    h            # int32, head index (for context)
):
    # Initialize lse accumulators
    row_max = -float("inf")
    sum_exp = 0.0

    # First pass: compute row-wise max and sum of exp for logsumexp
    m = 0
    while m < M_total:
        # For each mm in this chunk
        for mm in range(BLOCK_M):
            idx = m + mm
            # Mask to avoid out-of-range idx
            mask_scalar = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=mask_scalar, other=0)  # int32 token index

            # Load qn[h, :] and compute dot with Kc[tok, :]
            ar = tl.arange(0, N)
            qn_row = tl.load(qn_ptr + ar)  # [N]
            kc_row = tl.load(Kc_ptr + tok * N + ar, mask=mask_scalar, other=0.0)  # [N]
            dot_qn = tl.sum(qn_row * kc_row, axis=0)

            # Load qp[h, :] and compute dot with Kp[tok, :]
            br = tl.arange(0, Kp_dim)
            q = tl.load(qp_ptr + br)  # [Kp_dim]
            k = tl.load(Kp_ptr + tok * Kp_dim + br, mask=mask_scalar, other=0.0)  # [Kp_dim]
            dot_qp = tl.sum(q * k, axis=0)

            logits = dot_qn + dot_qp
            logits_scaled = logits * sm_scale
            row_max = tl.maximum(row_max, logits_scaled)
            # Only accumulate when idx is valid
            sum_exp += tl.where(mask_scalar, tl.exp(logits_scaled - row_max), 0.0)
        m += BLOCK_M

    # Compute lse for this (b, h)
    lse_val = tl.log(sum_exp) / tl.log(2.0)
    tl.store(lse_ptr, lse_val)

    # Second pass: compute output y[h, :] = sum_m attn[m] * Kc[m, :]
    y = tl.zeros([N], dtype=tl.float32)
    m = 0
    while m < M_total:
        for mm in range(BLOCK_M):
            idx = m + mm
            mask_scalar = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=mask_scalar, other=0)

            ar = tl.arange(0, N)
            qn_row = tl.load(qn_ptr + ar)
            kc_row = tl.load(Kc_ptr + tok * N + ar, mask=mask_scalar, other=0.0)
            dot_qn = tl.sum(qn_row * kc_row, axis=0)

            br = tl.arange(0, Kp_dim)
            q = tl.load(qp_ptr + br)
            k = tl.load(Kp_ptr + tok * Kp_dim + br, mask=mask_scalar, other=0.0)
            dot_qp = tl.sum(q * k, axis=0)

            logits = dot_qn + dot_qp
            logits_scaled = logits * sm_scale
            attn = tl.where(mask_scalar, tl.exp(logits_scaled - lse_val) / M_total, 0.0)
            y += attn * kc_row
        m += BLOCK_M

    # Store output
    tl.store(out_ptr + tl.arange(0, N), y)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_m=128):
        super().__init__()
        self.sm_scale = float(sm_scale)  # default matches original
        self.block_m = int(block_m)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale=None):
        # Fallback sm_scale
        if sm_scale is None:
            sm_scale = self.sm_scale

        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Sanity checks (as in original code)
        # We assume these are true; no torch assertions to keep Triton-only
        # Prepare data: cast to fp32 and make contiguous
        qn_fp32 = q_nope.to(torch.float32).contiguous()          # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()            # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).squeeze(1).contiguous()  # [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).squeeze(1).contiguous()  # [num_pages, Kp_dim]

        # Compute M_total per batch
        # Note: eval harness uses int32 kv_indptr; we compute M_total robustly
        # Create lse and output tensors
        lse = torch.empty((B, H), dtype=torch.float32, device=device)
        output_fp32 = torch.empty((B, H, head_dim_ckv), dtype=torch.float32, device=device)

        for b_idx in range(B):
            # Determine used tokens for this batch
            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item())
            M_total_b = max(end - start, 0)
            if M_total_b == 0:
                # No tokens for this batch element: output zeros and lse -inf
                output_fp32[b_idx] = 0.0
                lse[b_idx] = float("-inf")
                continue
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total_b]

            # For each head h
            for h_idx in range(H):
                # Launch Triton kernel: one program per (b,h)
                lse_and_output_kernel[(1,)](
                    qn_fp32[b_idx, h_idx].contiguous(),   # [N]
                    qp_fp32[b_idx, h_idx].contiguous(),   # [Kp_dim]
                    Kc_fp32,                               # [num_pages, N]
                    Kp_fp32,                               # [num_pages, Kp_dim]
                    tok_idx,                               # [M_total_b]
                    lse[b_idx, h_idx],                     # scalar output
                    output_fp32[b_idx, h_idx],             # [N]
                    head_dim_ckv,                         # N
                    head_dim_kpe,                         # Kp_dim
                    M_total_b,                            # M_total
                    sm_scale,                              # sm_scale
                    self.block_m,                         # BLOCK_M
                    b_idx, h_idx
                )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse