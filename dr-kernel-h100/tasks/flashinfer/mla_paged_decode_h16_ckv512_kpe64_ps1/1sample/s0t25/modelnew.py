import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_fused_kernel(
    qn_ptr,      # *fp32, [N]
    qp_ptr,      # *fp32, [Kp_dim]
    Kc_ptr,      # *fp32, [M_total, N] (we pass Kc rows via tok_idx)
    Kp_ptr,      # *fp32, [M_total, Kp_dim]
    tok_idx_ptr, # *int32, [M_total]
    lse_out_ptr, # *fp32, scalar output per (b,h)
    out_ptr,     # *fp32, [N]
    N: tl.constexpr,          # head_dim_ckv (512)
    Kp_dim: tl.constexpr,     # head_dim_kpe (64)
    M_total,                  # number of tokens for this batch
    sm_scale,                 # scale factor (float)
    BLOCK_M: tl.constexpr     # chunk size for token loop
):
    # Compute LogSumExp across tokens in chunks
    row_max = -float("inf")
    sum_exp = 0.0  # scalar fp32

    # First pass: compute lse
    m = 0
    while m < M_total:
        offs = m + tl.arange(0, BLOCK_M)
        mask = offs < M_total

        # Load qn (vector of size N)
        qn_vec = tl.load(qn_ptr + tl.arange(0, N), mask=mask, other=0.0)  # vector [N]

        # Load Kc rows for these tokens (mask ensures safety)
        kc_chunk = tl.load(Kc_ptr + offs * N + tl.arange(0, N), mask=mask, other=0.0)  # [BLOCK_M, N]

        # Load Kp rows for these tokens
        kp_chunk = tl.load(Kp_ptr + offs * Kp_dim + tl.arange(0, Kp_dim), mask=mask, other=0.0)  # [BLOCK_M, Kp_dim]

        # Compute logits = qn · Kc^T + qp · Kp^T per token, vectorized over N
        # qn_vec is [N], kc_chunk is [BLOCK_M, N], so we need per-token dot over N
        # We can do it by summing over N: dot_i = sum_j qn_vec[j] * kc_chunk[i, j]
        dot_qn = tl.sum(qn_vec[None, :] * kc_chunk, axis=1)  # [BLOCK_M]
        dot_qp = tl.sum(qp_ptr[None, :] * kp_chunk, axis=1)  # [BLOCK_M]

        logits = dot_qn + dot_qp  # [BLOCK_M]
        logits_scaled = logits * sm_scale  # [BLOCK_M]

        # Update row_max and sum_exp
        local_max = tl.max(tl.where(mask, logits_scaled, -float("inf")))
        row_max = tl.maximum(row_max, local_max)

        exp_vals = tl.exp(logits_scaled - row_max)
        # Mask out invalid entries to avoid contributing to sum_exp
        exp_vals = tl.where(mask, exp_vals, 0.0)
        sum_exp += tl.sum(exp_vals)

        m += BLOCK_M

    # Compute lse (base 2): lse = log(sum_exp) / ln(2)
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2
    # Store lse scalar
    tl.store(lse_out_ptr, lse_val)

    # Second pass: compute output vector y = sum_m attn[m] * Kc[m, :]
    m = 0
    y = tl.zeros([N], dtype=tl.float32)
    while m < M_total:
        offs = m + tl.arange(0, BLOCK_M)
        mask = offs < M_total

        qn_vec = tl.load(qn_ptr + tl.arange(0, N), mask=mask, other=0.0)
        kc_chunk = tl.load(Kc_ptr + offs * N + tl.arange(0, N), mask=mask, other=0.0)
        kp_chunk = tl.load(Kp_ptr + offs * Kp_dim + tl.arange(0, Kp_dim), mask=mask, other=0.0)

        dot_qn = tl.sum(qn_vec[None, :] * kc_chunk, axis=1)  # [BLOCK_M]
        dot_qp = tl.sum(qp_ptr[None, :] * kp_chunk, axis=1)  # [BLOCK_M]
        logits = dot_qn + dot_qp
        logits_scaled = logits * sm_scale

        attn = tl.exp(logits_scaled - lse_val)  # [BLOCK_M]
        attn = tl.where(mask, attn, 0.0)

        # Accumulate y += sum_m attn[m] * Kc[m, :]
        # For each i in chunk, add attn[i] * Kc[i, :]
        for i in range(BLOCK_M):
            if (m + i) < M_total:
                # attn_i scalar
                attn_i = attn[i]
                kc_row = kc_chunk[i, :]  # vector [N]
                y += attn_i * kc_row

        m += BLOCK_M

    # Store output vector y
    tl.store(out_ptr + tl.arange(0, N), y, mask=True)  # y already computed as [N]


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_m=128):
        super().__init__()
        self.sm_scale = sm_scale
        self.block_m = block_m

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices):
        # Expect 7 positional arguments as provided by the evaluator
        # Device and dtype setup
        device = q_nope.device
        N = q_nope.shape[-1]  # head_dim_ckv, fixed 512
        Kp_dim = q_pe.shape[-1]  # head_dim_kpe, fixed 64

        B = q_nope.shape[0]
        H = q_nope.shape[1]

        # Cast to fp32 for compute
        qn_fp32 = q_nope.to(torch.float32).contiguous()  # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()    # [B, H, Kp_dim]
        # Flatten ckv/kpe caches to [num_pages, N] / [num_pages, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous()  # [num_pages, 1, N] -> [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous()  # [num_pages, 1, Kp_dim] -> [num_pages, Kp_dim]

        # Prepare output and lse
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Compute tok_idx per batch
        # kv_indptr: [len_indptr], num_tokens = kv_indptr[-1] - kv_indptr[0]
        num_tokens = int(kv_indptr[-1].item())
        # For each batch b, tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b+1]]
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M_total = max(end - start, 0)
            if M_total == 0:
                # No tokens for this batch: output zeros, lse = -inf
                lse[b, :] = -float("inf")
                output_fp32[b] = torch.zeros((H, N), dtype=torch.float32, device=device)
                continue

            tok_idx = kv_indices[start:end].contiguous().to(torch.int32)  # [M_total]

            # Gather Kc and Kp rows for these tokens. We'll pass a temporary contiguous tensors [M_total, N] / [M_total, Kp_dim]
            # Build Kc_chunk and Kp_chunk by indexing Kc_fp32 and Kp_fp32 using tok_idx
            # Note: Triton expects pointers; we'll pass the chunked rows via tok_idx
            Kc_chunk = Kc_fp32[tok_idx]  # [M_total, N], fp32
            Kp_chunk = Kp_fp32[tok_idx]  # [M_total, Kp_dim], fp32

            # Select one head h for demonstration; actually we need to loop over all heads
            # Since output shape is [B, H, N], we launch the kernel for each (b, h)
            for h in range(H):
                # We need qn and qp for this head: reshape [N] and [Kp_dim]
                qn_row = qn_fp32[b, h, :].contiguous()  # [N]
                qp_row = qp_fp32[b, h, :].contiguous()  # [Kp_dim]

                # Output vector for this (b, h)
                out_vec = output_fp32[b, h, :].contiguous()  # [N]

                # Allocate scalar lse for this (b, h)
                lse_scalar = torch.empty((), dtype=torch.float32, device=device)

                # Launch kernel: fuse lse computation and output accumulation
                grid = (1,)  # single program instance; handle all tokens via loops
                lse_and_output_fused_kernel[grid](
                    qn_row,             # *fp32 [N]
                    qp_row,             # *fp32 [Kp_dim]
                    Kc_chunk,           # *fp32 [M_total, N]
                    Kp_chunk,           # *fp32 [M_total, Kp_dim]
                    tok_idx,            # *int32 [M_total]
                    lse_scalar,         # *fp32 scalar
                    out_vec,            # *fp32 [N]
                    N=N, Kp_dim=Kp_dim, M_total=M_total, sm_scale=self.sm_scale, BLOCK_M=self.block_m
                )

                # Store lse
                lse[b, h] = lse_scalar.item()

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse