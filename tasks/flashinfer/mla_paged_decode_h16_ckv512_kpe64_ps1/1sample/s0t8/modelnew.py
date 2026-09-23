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
    lse_ptr,     # *fp32, scalar (per (b,h))
    out_ptr,     # *fp32, [N]
    N,           # int32
    Kp_dim,      # int32
    M_total,     # int32
    sm_scale,    # fp32
    BLOCK_M: tl.constexpr,  # chunk size for token loop
    b,           # int32, batch index (for context, not used in pointer math)
    h            # int32, head index (for context, not used in pointer math)
):
    # Compute row-wise max and sum of exp for logsumexp
    row_max = -float("inf")
    sum_exp = 0.0

    # First pass: compute lse
    m = 0
    while m < M_total:
        # Process in chunks of BLOCK_M
        for mm in range(BLOCK_M):
            idx = m + mm
            if idx >= M_total:
                break
            tok = tl.load(tok_idx_ptr + idx)
            # Compute dot(qn[h], Kc[tok, :])
            qn_row = tl.load(qn_ptr + tl.arange(0, N))  # [N]
            kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N))  # [N]
            dot_qn = tl.sum(qn_row * kc_row, axis=0)

            # Compute dot(qp[h], Kp[tok, :])
            q = tl.load(qp_ptr + tl.arange(0, Kp_dim))  # [Kp_dim]
            k = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim))  # [Kp_dim]
            dot_qp = tl.sum(q * k, axis=0)

            logits = dot_qn + dot_qp
            logits_scaled = logits * sm_scale
            row_max = tl.maximum(row_max, logits_scaled)
            sum_exp += tl.exp(logits_scaled - row_max)
        m += BLOCK_M

    lse_val = tl.log(sum_exp) / tl.log(2.0)
    tl.store(lse_ptr, lse_val)

    # Second pass: compute output y[h, :] = sum_m attn[m] * Kc[m, :]
    # attn[m] = exp((logits_scaled[m] - lse) / M_total)
    m = 0
    while m < M_total:
        for mm in range(BLOCK_M):
            idx = m + mm
            if idx >= M_total:
                break
            tok = tl.load(tok_idx_ptr + idx)
            qn_row = tl.load(qn_ptr + tl.arange(0, N))  # [N]
            kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N))  # [N]
            dot_qn = tl.sum(qn_row * kc_row, axis=0)

            q = tl.load(qp_ptr + tl.arange(0, Kp_dim))  # [Kp_dim]
            k = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim))  # [Kp_dim]
            dot_qp = tl.sum(q * k, axis=0)

            logits = dot_qn + dot_qp
            logits_scaled = logits * sm_scale
            attn = tl.exp(logits_scaled - lse_val) / (M_total + 0.0)
            # y += attn * Kc[tok, :]
            kc_row2 = tl.load(Kc_ptr + tok * N + tl.arange(0, N))  # [N]
            y_chunk = attn * kc_row2  # elementwise
            out_row = tl.load(out_ptr + tl.arange(0, N), mask=True, other=0.0)  # [N]
            out_row = out_row + y_chunk
            tl.store(out_ptr + tl.arange(0, N), out_row)
        m += BLOCK_M


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_m=128):
        super().__init__()
        self.sm_scale = float(sm_scale)
        self.block_m = int(block_m)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale=None):
        # Handle sm_scale: if None, use self.sm_scale; otherwise use provided
        sm_scale = self.sm_scale if (sm_scale is None) else float(sm_scale)

        device = q_nope.device
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]

        # Ensure types and contiguity
        qn_fp32 = q_nope.to(torch.float32).contiguous()      # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()        # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous()   # [num_pages, 1, N] -> [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous()   # [num_pages, 1, Kp_dim] -> [num_pages, Kp_dim]

        # Compute M_total per batch from kv_indptr (shape [B+1])
        M_totals = (kv_indptr[1:] - kv_indptr[:batch_size]).to(torch.int32).contiguous()  # [B]
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Prepare output tensor in fp32
        output_fp32 = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

        for b_idx in range(batch_size):
            if b_idx >= batch_size:
                break
            M_total_b = int(M_totals[b_idx].item())
            if M_total_b <= 0:
                # No tokens used for this batch element
                lse[b_idx] = -float("inf")
                output_fp32[b_idx] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                continue

            # Gather token indices for this batch
            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item())
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total_b]

            # For each head h
            for h_idx in range(num_qo_heads):
                # Launch Triton kernel once per (b, h)
                # Pass qn[h], qp[h], Kc, Kp, tok_idx, and out pointer for this head
                y = output_fp32[b_idx, h_idx]  # [N]
                lse_scalar = lse[b_idx, h_idx]  # scalar

                lse_and_output_kernel[(1,)](
                    qn_fp32[b_idx, h_idx].contiguous(),   # [N] fp32
                    qp_fp32[b_idx, h_idx].contiguous(),   # [Kp_dim] fp32
                    Kc_fp32,                               # [num_pages, N] fp32
                    Kp_fp32,                               # [num_pages, Kp_dim] fp32
                    tok_idx,                               # [M_total_b] int32
                    lse_scalar,                            # *fp32 scalar
                    y,                                     # *fp32 [N]
                    head_dim_ckv,                         # N
                    head_dim_kpe,                         # Kp_dim
                    M_total_b,                            # M_total
                    self.sm_scale,                        # sm_scale
                    self.block_m,                         # BLOCK_M
                    b_idx, h_idx
                )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse