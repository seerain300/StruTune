import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_lse_kernel(
    qn_ptr,      # *fp32, [N]
    Kc_ptr,      # *fp32, [TOTAL_PAGES, N]
    tok_idx_ptr, # *int32, [M_total]
    lse_ptr,     # *fp32, scalar (per (b,h))
    N: tl.constexpr,            # head_dim_ckv (512)
    M_total,                    # int32 runtime
    sm_scale,                   # fp32
    BLOCK_M: tl.constexpr       # chunk size for tokens
):
    # Row-wise max and sum of exp for LogSumExp
    row_max = -float("inf")
    sum_exp = 0.0

    # First pass: compute row_max and sum_exp of exp(logits_scaled - row_max)
    m = 0
    while m < M_total:
        # Process tokens in chunks
        for mm in tl.static_range(BLOCK_M):
            idx = m + mm
            valid = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=valid, other=0)
            offs = tl.arange(0, N)
            qn_row = tl.load(qn_ptr + offs)
            kc_ptrs = Kc_ptr + tok * N + offs
            kc_row = tl.load(kc_ptrs, mask=valid, other=0.0)
            dot_ck = tl.sum(qn_row * kc_row, axis=0)
            logits = dot_ck * sm_scale
            # Update row_max and sum_exp
            if valid:
                row_max = tl.maximum(row_max, logits)
                sum_exp += tl.exp(logits - row_max)
    # Store lse = log(sum_exp) / ln(2)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_ptr, lse_val)


@triton.jit
def compute_output_kernel(
    qn_ptr,      # *fp32, [N]
    qp_ptr,      # *fp32, [Kp_dim]
    Kc_ptr,      # *fp32, [TOTAL_PAGES, N]
    Kp_ptr,      # *fp32, [TOTAL_PAGES, Kp_dim]
    tok_idx_ptr, # *int32, [M_total]
    lse_ptr,     # *fp32, scalar (per (b,h))
    out_ptr,     # *fp32, [N]
    N: tl.constexpr,            # head_dim_ckv (512)
    Kp_dim: tl.constexpr,       # head_dim_kpe (64)
    M_total,                    # int32 runtime
    sm_scale,                   # fp32
    BLOCK_M: tl.constexpr       # chunk size for tokens
):
    # Initialize output vector to zero
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, 0.0)

    # Compute LogSumExp lse value (same as above) and reuse
    # We recompute here to have consistent control flow; alternatively, we can pass lse_ptr as read-only and compute once.
    row_max = -float("inf")
    sum_exp = 0.0

    # First pass: compute row_max and sum_exp (to know lse for normalization)
    m = 0
    while m < M_total:
        for mm in tl.static_range(BLOCK_M):
            idx = m + mm
            valid = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=valid, other=0)
            qn_row = tl.load(qn_ptr + tl.arange(0, N))
            kc_ptrs = Kc_ptr + tok * N + tl.arange(0, N)
            kc_row = tl.load(kc_ptrs, mask=valid, other=0.0)
            dot_ck = tl.sum(qn_row * kc_row, axis=0)
            logits = dot_ck * sm_scale
            if valid:
                row_max = tl.maximum(row_max, logits)
                sum_exp += tl.exp(logits - row_max)
    lse_val = tl.log(sum_exp) / 1.4426950408889634

    # Second pass: accumulate output y = sum_m attn[m] * Kc[tok, :]
    for m in tl.static_range(BLOCK_M, 1024):  # placeholder; actual loop handled by runtime m
        pass
    # The above placeholder is to satisfy Triton; we will implement the loop via dynamic while as Triton doesn't support
    # tl.static_range over runtime M_total; hence we implement dynamic while for output accumulation.

    # Implement dynamic while loop for output accumulation
    m = 0
    while m < M_total:
        for mm in tl.static_range(BLOCK_M):
            idx = m + mm
            valid = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=valid, other=0)
            qn_row = tl.load(qn_ptr + tl.arange(0, N))
            kc_ptrs = Kc_ptr + tok * N + tl.arange(0, N)
            kc_row = tl.load(kc_ptrs, mask=valid, other=0.0)
            dot_ck = tl.sum(qn_row * kc_row, axis=0)
            logits = dot_ck * sm_scale
            attn = tl.exp(logits - lse_val) / M_total
            out_ptrs = out_ptr + tl.arange(0, N)
            # out += attn * Kc[tok, :]
            tl.store(out_ptrs, tl.load(out_ptrs) + attn * kc_row, mask=valid)
        m += BLOCK_M

    # Note: The above dynamic loop is necessary. Triton requires explicit runtime loops. The previous attempt
    # used tl.static_range with a fixed 1024; this is not correct. We replace it with actual dynamic while loops.


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_m=128, block_out=128):
        super().__init__()
        self.sm_scale = sm_scale
        self.block_m = block_m
        self.block_out = block_out

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices):
        """
        q_nope: [B, H, N], bfloat16
        q_pe: [B, H, Kp_dim], bfloat16
        ckv_cache: [num_pages, 1, N], bfloat16
        kpe_cache: [num_pages, 1, Kp_dim], bfloat16
        kv_indptr: [len_indptr], int32
        kv_indices: [num_tokens], int32
        Returns: output [B, H, N], bfloat16; lse [B, H], fp32
        """
        device = q_nope.device
        B, H, N = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        num_pages = ckv_cache.shape[0]
        assert num_pages == kpe_cache.shape[0], "ckv_cache and kpe_cache must have same num_pages"

        # Cast inputs to fp32 for computation
        qn_fp32 = q_nope.to(torch.float32).contiguous()   # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()     # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous()  # [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous()  # [num_pages, Kp_dim]

        # Prepare outputs
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Compute per-batch token counts and indices
        # len_indptr = kv_indptr.shape[0]
        start = 0
        for b_idx in range(B):
            end = int(kv_indptr[b_idx + 1].item()) if (b_idx + 1) < kv_indptr.shape[0] else 0
            M_total = max(end - int(kv_indptr[b_idx].item()), 0)
            if M_total == 0:
                # No tokens for this batch, match original behavior: output zeros, lse = -inf
                lse[b_idx] = -float("inf")
                continue
            tok_idx = kv_indices[int(kv_indptr[b_idx].item()):end].to(torch.int32).contiguous()  # [M_total]

            # Launch Triton kernel to compute lse per (b, h)
            for h_idx in range(H):
                lse_ptr = lse[b_idx, h_idx].view(1)  # scalar pointer
                compute_lse_kernel[(1,)](
                    qn_fp32[b_idx, h_idx],            # [N]
                    Kc_fp32,                           # [num_pages, N]
                    tok_idx,                           # [M_total]
                    lse_ptr,                           # *fp32 scalar
                    N=N, M_total=M_total, sm_scale=self.sm_scale, BLOCK_M=self.block_m
                )

            # Launch Triton kernel to compute output per (b, h)
            for h_idx in range(H):
                out_ptr = output_fp32[b_idx, h_idx]  # [N]
                compute_output_kernel[(1,)](
                    qn_fp32[b_idx, h_idx],            # [N]
                    qp_fp32[b_idx, h_idx],            # [Kp_dim]
                    Kc_fp32,                           # [num_pages, N]
                    Kp_fp32,                           # [num_pages, Kp_dim]
                    tok_idx,                           # [M_total]
                    lse[b_idx, h_idx].view(1),        # *fp32 scalar
                    out_ptr,                           # *fp32 [N]
                    N=N, Kp_dim=Kp_dim, M_total=M_total, sm_scale=self.sm_scale, BLOCK_M=self.block_out
                )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse