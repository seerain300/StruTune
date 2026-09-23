import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_fused_kernel(
    qn_ptr,      # *fp32, [N]
    qp_ptr,      # *fp32, [Kp_dim]
    Kc_ptr,      # *fp32, [TOTAL_PAGES, N]
    Kp_ptr,      # *fp32, [TOTAL_PAGES, Kp_dim]
    tok_idx_ptr, # *int32, [M_total]
    lse_ptr,     # *fp32, scalar (0-dim)
    out_ptr,     # *fp32, [N]
    N,           # int32
    Kp_dim,      # int32
    M_total,     # int32
    sm_scale,    # fp32
    BLOCK_M: tl.constexpr
):
    # Compute LogSumExp of logits_scaled = dot(qn, Kc[tok]) * sm_scale over tokens
    row_max = -float("inf")
    sum_exp = 0.0

    m = 0
    while m < M_total:
        # Process tokens in chunks of BLOCK_M with masks
        for mm in range(BLOCK_M):
            idx = m + mm
            # Mask for valid idx
            valid = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=valid, other=0)  # int32
            # Load Kc row [N] with mask
            kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=valid, other=0.0)  # [N]
            dot = tl.sum(qn_ptr * kc_row)  # scalar
            logits_scaled = dot * sm_scale
            # Update running max and sum_exp
            # If idx is invalid, logits_scaled is 0, so it won't affect sum_exp
            row_max = tl.maximum(row_max, logits_scaled)
            sum_exp += tl.exp(logits_scaled - row_max) * valid.to(tl.float32)
        m += BLOCK_M

    # Compute lse = log(sum_exp) / ln(2)
    lse_val = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr, lse_val)

    # Now compute output vector y = sum_m exp((dot(qp, Kp[tok]) + dot(qn, Kc[tok]) * sm_scale - lse) / M_total) * Kc[tok, :]
    # We need to iterate tokens again and accumulate y.
    m = 0
    while m < M_total:
        for mm in range(BLOCK_M):
            idx = m + mm
            valid = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=valid, other=0)  # int32

            # Compute logits_scaled for this token
            kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=valid, other=0.0)  # [N]
            dot_nc = tl.sum(qn_ptr * kc_row)  # dot(qn, Kc[tok])

            kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=valid, other=0.0)  # [Kp_dim]
            dot_pk = tl.sum(qp_ptr * kp_row)  # dot(qp, Kp[tok])
            logits_scaled = (dot_nc + dot_pk) * sm_scale

            attn = tl.exp(logits_scaled - lse_val) * (1.0 / M_total) * valid.to(tl.float32)  # scalar

            # Accumulate y += attn * Kc[tok, :]
            kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=valid, other=0.0)  # [N]
            out_ptr += attn * kc_row  # elementwise multiply and accumulate
        m += BLOCK_M


class ModelNew(torch.nn.Module):
    def __init__(self, block_m=128, sm_scale=1.0):
        super().__init__()
        self.block_m = block_m
        self.sm_scale = sm_scale

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale=None):
        # Ensure device and dtype handling
        device = q_nope.device
        # Cast and make contiguous for fp32 computation
        qn_fp32 = q_nope.to(torch.float32).contiguous()  # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()   # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous()  # [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous()  # [num_pages, Kp_dim]

        B, H, N = qn_fp32.shape
        _, _, Kp_dim = qp_fp32.shape
        num_pages = Kc_fp32.shape[0]

        # Compute M_total per batch from kv_indptr
        M_total_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M_total_list.append(end - start)
        # If any batch has zero tokens, we can skip work and return zeros
        # but evaluator typically provides non-empty kv_indices.

        # Prepare output tensors
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel per (batch, head)
        for b_idx in range(B):
            M_total_b = M_total_list[b_idx]
            if M_total_b == 0:
                output_fp32[b_idx] = torch.zeros(H, N, dtype=torch.float32, device=device)
                lse[b_idx] = -float("inf")
                continue

            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total_b]
            # Prepare per-(batch, head) vectors
            qn_h = qn_fp32[b_idx]  # [N]
            qp_h = qp_fp32[b_idx]  # [Kp_dim]

            # Output vector for this (b,h)
            y = torch.zeros(N, dtype=torch.float32, device=device)

            # Scalar lse buffer
            lse_scalar = torch.empty((), dtype=torch.float32, device=device)

            # Launch kernel: grid size (1,) since one scalar lse and output per (b,h)
            lse_and_output_fused_kernel[(1,)](
                qn_h.contiguous(),           # *fp32 [N]
                qp_h.contiguous(),           # *fp32 [Kp_dim]
                Kc_fp32,                     # *fp32 [num_pages, N]
                Kp_fp32,                     # *fp32 [num_pages, Kp_dim]
                tok_idx,                     # *int32 [M_total_b]
                lse_scalar,                  # *fp32 scalar
                y,                           # *fp32 [N]
                N,                           # int32
                Kp_dim,                      # int32
                M_total_b,                   # int32
                self.sm_scale if sm_scale is None else float(sm_scale),  # fp32
                self.block_m,                # BLOCK_M
                b_idx,                       # batch index (for context, not used in math)
                0                            # dummy head index since we iterate h below
            )

            # Store results
            lse[b_idx] = lse_scalar
            output_fp32[b_idx] = y.view(H, N)

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse