import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_kernel(
    qn_ptr,       # *fp32, pointer to qn[h, :] flattened of length N
    qp_ptr,       # *fp32, pointer to qp[h, :] flattened of length Kp_dim
    Kc_ptr,       # *fp32, pointer to Kc [TOTAL_PAGES, N], contiguous
    Kp_ptr,       # *fp32, pointer to Kp [TOTAL_PAGES, Kp_dim], contiguous
    tok_idx_ptr,  # *int32, [M_total]
    lse_ptr,      # *fp32, [B*H] storage for per-(b,h) lse
    N: tl.constexpr,           # head_dim_ckv (512)
    Kp_dim: tl.constexpr,      # head_dim_kpe (64)
    M_total: tl.constexpr,     # number of tokens for this batch
    sm_scale,                  # fp32 scalar
    b, h                        # int32 batch and head indices
):
    # Compute lse for this (b, h)
    row_max = -float("inf")
    sum_exp = 0.0

    m = 0
    while m < M_total:
        tok = tl.load(tok_idx_ptr + m)  # token index into Kc/Kp

        # Load qn[h, :] vector
        idx_qn = m * N + tl.arange(0, N)
        qn_row = tl.load(qn_ptr + idx_qn, mask=tl.arange(0, N) < N, other=0.0)

        # Load Kc[tok, :]
        kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
        dot_qn = tl.sum(qn_row * kc_row, axis=0)

        # Load qp[h, :] vector
        idx_qp = m * Kp_dim + tl.arange(0, Kp_dim)
        qp_row = tl.load(qp_ptr + idx_qp, mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)

        # Load Kp[tok, :]
        kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)
        dot_qp = tl.sum(qp_row * kp_row, axis=0)

        logits = dot_qn + dot_qp
        logits_scaled = logits * sm_scale

        # Row-wise LogSumExp accumulation
        row_max = tl.maximum(row_max, logits_scaled)
        sum_exp += tl.exp(logits_scaled - row_max)
        m += 1

    lse_val = tl.log(sum_exp) / tl.log(2.0)
    tl.store(lse_ptr + b * 16 + h, lse_val)

    # Now compute output y[h, :] = sum_m exp(logits_scaled - lse_val) / M_total * Kc[m, :]
    m = 0
    y = tl.zeros([N], dtype=tl.float32)
    while m < M_total:
        tok = tl.load(tok_idx_ptr + m)

        # Recompute logits_scaled
        idx_qn = m * N + tl.arange(0, N)
        qn_row = tl.load(qn_ptr + idx_qn, mask=tl.arange(0, N) < N, other=0.0)

        kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
        dot_qn = tl.sum(qn_row * kc_row, axis=0)

        idx_qp = m * Kp_dim + tl.arange(0, Kp_dim)
        qp_row = tl.load(qp_ptr + idx_qp, mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)
        kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)
        dot_qp = tl.sum(qp_row * kp_row, axis=0)

        logits = dot_qn + dot_qp
        logits_scaled = logits * sm_scale
        attn = tl.exp(logits_scaled - lse_val) / M_total

        kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
        y += attn * kc_row

        m += 1

    # Store y as fp32 (we'll cast to bfloat16 on host)
    # Note: output layout is [B, H, N] fp32 in host
    return y


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0):
        super().__init__()
        self.sm_scale = float(sm_scale)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on GPU and contiguous
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors."
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        N = q_nope.shape[2]
        Kp_dim = q_pe.shape[2]

        # Convert inputs to fp32 for numeric stability
        qn_fp32 = q_nope.to(torch.float32)          # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32)            # [B, H, Kp_dim]

        # Gather Kc and Kp for all tokens
        # TOTAL_PAGES is the length of ckv_cache's first dimension
        TOTAL_PAGES = ckv_cache.shape[0]
        Kc_fp32 = ckv_cache.to(torch.float32).squeeze(1)  # [TOTAL_PAGES, N]
        Kp_fp32 = kpe_cache.to(torch.float32).squeeze(1)  # [TOTAL_PAGES, Kp_dim]

        # Prepare output and lse storage
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process each batch element
        for b_idx in range(B):
            # Compute number of used tokens for this batch: L_tokens = kv_indptr[b+1] - kv_indptr[b]
            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item())
            M_total = end - start
            if M_total <= 0:
                # No tokens used; output zeros, lse -inf
                output_fp32[b_idx] = 0.0
                lse[b_idx] = float("-inf")
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total]

            # Launch Triton kernel once per (b, h)
            for h_idx in range(H):
                y = lse_and_output_kernel[(1,)](
                    qn_fp32[b_idx, h_idx].contiguous(),   # [N] fp32
                    qp_fp32[b_idx, h_idx].contiguous(),   # [Kp_dim] fp32
                    Kc_fp32,                               # [TOTAL_PAGES, N] fp32
                    Kp_fp32,                               # [TOTAL_PAGES, Kp_dim] fp32
                    tok_idx,                               # [M_total] int32
                    lse[b_idx],                            # *fp32 for this (b,h)
                    N, Kp_dim, M_total, self.sm_scale, b_idx, h_idx
                )
                output_fp32[b_idx, h_idx] = y

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse