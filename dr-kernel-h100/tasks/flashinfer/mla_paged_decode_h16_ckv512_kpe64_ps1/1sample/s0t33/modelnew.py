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
    out_y_ptr,   # *fp32, [N]
    out_lse_ptr, # *fp32, scalar
    N: tl.constexpr,           # head_dim_ckv (512)
    Kp_dim: tl.constexpr,      # head_dim_kpe (64)
    M_total,                   # number of tokens for this batch element (runtime int)
    sm_scale,                  # fp32 scaling factor
    BLOCK_M: tl.constexpr      # chunk size over tokens (not used in simple loop, but kept for compatibility)
):
    # Compute LogSumExp (scaled) over tokens for this (qn, qp)
    row_max = -float("inf")
    sum_exp = 0.0  # scalar fp32

    # First pass: compute row_max and sum_exp
    m = 0
    while m < M_total:
        tok = tl.load(tok_idx_ptr + m)  # int32
        # Load qn, Kc, Kp row for this token
        qn_vec = tl.load(qn_ptr + tl.arange(0, N))
        Kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N))
        Kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim))
        # Compute dot products
        logits = tl.sum(qn_vec * Kc_row) + tl.sum(qp_ptr * Kp_row)
        logits_scaled = logits * sm_scale
        # Update row_max and sum_exp (for scalar row, this reduces to single element)
        row_max = tl.maximum(row_max, logits_scaled)
        sum_exp += tl.exp(logits_scaled - row_max)
        m += 1

    lse = tl.log(sum_exp) / 0.6931471805599453  # ln(2)

    # Store lse
    tl.store(out_lse_ptr, lse)

    # Second pass: accumulate output vector
    acc = tl.zeros((N,), dtype=tl.float32)
    m = 0
    while m < M_total:
        tok = tl.load(tok_idx_ptr + m)  # int32
        qn_vec = tl.load(qn_ptr + tl.arange(0, N))
        Kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N))
        Kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim))
        logits = tl.sum(qn_vec * Kc_row) + tl.sum(qp_ptr * Kp_row)
        logits_scaled = logits * sm_scale
        attn = tl.exp(logits_scaled - lse) / M_total
        acc += attn * Kc_row
        m += 1

    # Store output vector
    tl.store(out_y_ptr + tl.arange(0, N), acc)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_m=1):
        super().__init__()
        self.sm_scale = float(sm_scale)
        self.block_m = int(block_m)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale=None):
        # Match evaluator's call: no extra defaults, 7 args expected
        B, H, N = q_nope.shape
        Kp_dim = q_pe.shape[-1]
        num_pages, _, N_ckv = ckv_cache.shape
        num_pages2, _, Kp_ckv = kpe_cache.shape
        assert N_ckv == N and Kp_ckv == Kp_dim, "Cache dimensions must match head dims."
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA."

        # Cast to fp32 and make contiguous
        qn_fp32 = q_nope.to(torch.float32).contiguous()  # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()    # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous().view(-1, N)  # [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous().view(-1, Kp_dim)  # [num_pages, Kp_dim]

        # Compute tok_idx per batch
        # Note: For Triton, we can't directly index Kc/Kp with a runtime vector token.
        # So we precompute tok_idx and pass to kernel. This matches original run semantics.
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M_total = end - start
            if M_total <= 0:
                # No tokens for this batch element; output zeros and lse -inf
                lse[b] = -float("inf")
                output_fp32[b] = 0.0
                continue

            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total] int32

            # Launch Triton kernel per (b, h)
            for h in range(H):
                # Pointers to qn, qp for this (b, h)
                qn_ptr = qn_fp32[b, h]  # [N]
                qp_ptr = qp_fp32[b, h]  # [Kp_dim]
                # Output and lse scalars
                out_y = torch.empty((N,), dtype=torch.float32, device=q_nope.device)
                out_lse = torch.empty((), dtype=torch.float32, device=q_nope.device)
                # Grid is 1D over N (vectorized store), but we do scalar work inside kernel
                lse_and_output_kernel[(1,)](
                    qn_ptr, qp_ptr, Kc_fp32, Kp_fp32, tok_idx,
                    out_y, out_lse,
                    N, Kp_dim, M_total, self.sm_scale if sm_scale is None else float(sm_scale),
                    self.block_m
                )
                # Store result
                lse[b, h] = out_lse[0]
                output_fp32[b, h] = out_y

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse