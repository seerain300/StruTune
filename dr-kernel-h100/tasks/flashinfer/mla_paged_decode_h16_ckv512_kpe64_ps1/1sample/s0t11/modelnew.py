import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_kernel(
    qn_ptr,      # *fp32, [N]
    qp_ptr,      # *fp32, [Kp_dim]
    Kc_ptr,      # *fp32, [num_pages, N]
    Kp_ptr,      # *fp32, [num_pages, Kp_dim]
    tok_idx_ptr, # *int32, [ROWS, BLOCK_M] 2D grid of token indices
    lse_ptr,     # *fp32, scalar pointer for this (b,h)
    out_ptr,     # *fp32, [N]
    N: tl.constexpr,            # head_dim_ckv (512)
    Kp_dim: tl.constexpr,       # head_dim_kpe (64)
    M_total_b: tl.constexpr,    # actual number of tokens used (<= ROWS * BLOCK_M)
    sm_scale,                   # fp32 scaling factor
    ROWS: tl.constexpr,         # number of chunks in the token grid
    BLOCK_M: tl.constexpr        # chunk size along tokens
):
    # Cast N and Kp_dim to fp32 scalars
    N_f = tl.full((), N, tl.float32)
    Kp_dim_f = tl.full((), Kp_dim, tl.float32)

    # Row-wise max and sum_exp for logsumexp
    row_max = -float("inf")
    sum_exp = 0.0

    # First pass: compute logsumexp over tokens
    for r in range(ROWS):
        # Compute indices for this chunk
        mm = tl.arange(0, BLOCK_M)
        idx = r * BLOCK_M + mm  # [BLOCK_M]
        mask = idx < M_total_b  # mask to ignore padding
        tok = tl.load(tok_idx_ptr + r * BLOCK_M + mm, mask=mask, other=-1)  # [BLOCK_M] int32

        # Load qn[h] row vector
        qn_row = tl.load(qn_ptr + mm, mask=mm < N, other=0.0)  # [BLOCK_M] fp32
        # Load Kc rows [BLOCK_M, N]
        kc_ptrs = Kc_ptr + tok[:, None] * N + tl.arange(0, N)[None, :]  # [BLOCK_M, N]
        kc_vals = tl.load(kc_ptrs, mask=(mask[:, None] & (tl.arange(0, N)[None, :] < N)), other=0.0)  # [BLOCK_M, N] fp32
        kc_sum = tl.sum(kc_vals * qn_row[:, None], axis=0)  # [N] reduce over BLOCK_M

        # Load Kp rows [BLOCK_M, Kp_dim]
        kp_ptrs = Kp_ptr + tok[:, None] * Kp_dim + tl.arange(0, Kp_dim)[None, :]  # [BLOCK_M, Kp_dim]
        kp_vals = tl.load(kp_ptrs, mask=(mask[:, None] & (tl.arange(0, Kp_dim)[None, :] < Kp_dim)), other=0.0)  # [BLOCK_M, Kp_dim] fp32
        qp_row = tl.load(qp_ptr + tl.arange(0, Kp_dim), mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)  # [Kp_dim]
        kp_sum = tl.sum(kp_vals * qp_row[:, None], axis=0)  # [Kp_dim]

        # logits = kc_sum + kp_sum, scaled
        logits = kc_sum + kp_sum  # [N]
        logits_scaled = logits * sm_scale  # [N]
        # For masked entries (idx >= M_total_b), set logits_scaled to -inf so they don't affect max/sum
        # We compute per element: if mask[i] is False, set logits_scaled[i] = -inf
        # Build per-element mask
        # Note: We want to apply mask per token i in this chunk. Triton doesn't support vectorized if on scalar,
        # but we can use masked loads; here we compute logits_scaled and then set masked positions to -inf via tl.where.
        # However, since kc_vals/kp_vals for masked positions are zero, kc_sum/kp_sum for those positions are zero,
        # and kc_sum + kp_sum = 0 for masked. We must set 0 -> -inf to exclude from max/sum. Use tl.where.
        # We can create a zeros_like mask via comparing idx to M_total_b, but we need per-element mask.
        # Instead, we set -inf directly:
        for i in range(BLOCK_M):
            if (r * BLOCK_M + i) >= M_total_b:
                # For this token, set kc_sum[i] and kp_sum[i] to -inf
                kc_sum = kc_sum.replace(kc_sum[i], -float("inf"))
                kp_sum = kp_sum.replace(kp_sum[i], -float("inf"))
        # Now compute chunk logsumexp
        chunk_max = tl.max(kc_sum + kp_sum, axis=0)  # scalar
        chunk_sum = tl.sum(tl.exp((kc_sum + kp_sum) - chunk_max), axis=0)  # scalar
        # Update running max/sum
        row_max = tl.maximum(row_max, chunk_max)
        sum_exp += chunk_sum

    # Compute lse per (b,h): logsumexp scaled by log2
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2 + row_max

    # Store lse for this (b,h)
    tl.store(lse_ptr, lse_val)

    # Second pass: compute output vector y = sum_m attn[m] * Kc[tok, :]
    for r in range(ROWS):
        mm = tl.arange(0, BLOCK_M)
        idx = r * BLOCK_M + mm  # [BLOCK_M]
        mask = idx < M_total_b
        tok = tl.load(tok_idx_ptr + r * BLOCK_M + mm, mask=mask, other=-1)  # [BLOCK_M] int32

        # Load qn[h] row vector
        qn_row = tl.load(qn_ptr + mm, mask=mm < N, other=0.0)  # [BLOCK_M] fp32
        # Kc rows [BLOCK_M, N]
        kc_ptrs = Kc_ptr + tok[:, None] * N + tl.arange(0, N)[None, :]  # [BLOCK_M, N]
        kc_vals = tl.load(kc_ptrs, mask=(mask[:, None] & (tl.arange(0, N)[None, :] < N)), other=0.0)  # [BLOCK_M, N] fp32
        kc_sum = tl.sum(kc_vals * qn_row[:, None], axis=0)  # [N]

        # Kp rows [BLOCK_M, Kp_dim]
        kp_ptrs = Kp_ptr + tok[:, None] * Kp_dim + tl.arange(0, Kp_dim)[None, :]  # [BLOCK_M, Kp_dim]
        kp_vals = tl.load(kp_ptrs, mask=(mask[:, None] & (tl.arange(0, Kp_dim)[None, :] < Kp_dim)), other=0.0)  # [BLOCK_M, Kp_dim] fp32
        qp_row = tl.load(qp_ptr + tl.arange(0, Kp_dim), mask=tl.arange(0, Kp_dim) < Kp_dim, other=0.0)  # [Kp_dim]
        kp_sum = tl.sum(kp_vals * qp_row[:, None], axis=0)  # [Kp_dim]
        logits = kc_sum + kp_sum  # [N]
        logits_scaled = logits * sm_scale  # [N]

        # attn per token: exp((logits_scaled - lse) / M_total_b)
        # For masked positions (idx >= M_total_b), skip contribution
        for i in range(BLOCK_M):
            pos = r * BLOCK_M + i
            if pos < M_total_b:
                attn_i = tl.exp((logits_scaled[i] - lse_val) / tl.full((), M_total_b, tl.float32))
                # Accumulate y = sum_m attn[m] * Kc[tok[m], :]
                kvec_ptrs = Kc_ptr + tok[i] * N + tl.arange(0, N)
                kvec = tl.load(kvec_ptrs, mask=(tl.arange(0, N) < N), other=0.0)  # [N]
                out_vec = out_ptr + tl.arange(0, N)
                # Atomic add: y += attn_i * kvec
                # Triton does not support direct per-element vector atomics here; do scalar loop
                for j in range(N):
                    tl.store(out_vec + j, tl.load(out_vec + j) + attn_i * kvec[j])

    # We used atomic adds to accumulate y. No need to store anything else.


class ModelNew(torch.nn.Module):
    def __init__(self, block_m: int = 128):
        super().__init__()
        self.block_m = block_m  # token chunk size

    def forward(self, q_nope: torch.Tensor, q_pe: torch.Tensor, ckv_cache: torch.Tensor, kpe_cache: torch.Tensor,
                kv_indptr: torch.Tensor, kv_indices: torch.Tensor, sm_scale: float):
        """
        Triton implementation of the original run function, computing output and lse entirely inside Triton kernels.
        Arguments:
            q_nope: [B, H, N] bfloat16
            q_pe:   [B, H, Kp_dim] bfloat16
            ckv_cache: [num_pages, 1, N] bfloat16
            kpe_cache: [num_pages, 1, Kp_dim] bfloat16
            kv_indptr: [B+1] int32
            kv_indices: [L] int32
            sm_scale: float
        Returns:
            output: [B, H, N] bfloat16
            lse: [B, H] fp32
        """
        assert q_nope.ndim == 3, "q_nope must be [B, H, N]"
        assert q_pe.ndim == 3, "q_pe must be [B, H, Kp_dim]"
        B, H, N = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        num_pages = ckv_cache.shape[0]
        assert ckv_cache.shape == (num_pages, 1, N)
        assert kpe_cache.shape == (num_pages, 1, Kp_dim)

        device = q_nope.device

        # Extract batch sizes from kv_indptr
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr length must be B+1"
        M_total_all = kv_indptr[1:] - kv_indptr[:-1]  # [B]
        assert torch.all(M_total_all >= 0), "Invalid kv_indptr: negative token counts"

        # Cast q_nope and q_pe to fp32
        qn_fp32 = q_nope.to(torch.float32).contiguous()  # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()    # [B, H, Kp_dim]

        # Cast ckv_cache and kpe_cache to fp32
        Kc_fp32 = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, N]
        Kp_fp32 = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Kp_dim]

        # Allocate output and lse
        output_fp32 = torch.zeros((B, H, N), dtype=torch.float32, device=device)
        lse_fp32 = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # Process each batch b
        for b_idx in range(B):
            M_total_b = int(M_total_all[b_idx])
            if M_total_b == 0:
                # No KV for this batch, output zeros and lse -inf
                lse_fp32[b_idx] = -float("inf")
                continue

            # Gather token indices for this batch
            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item())
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total_b]

            # Build 2D token grid [ROWS, BLOCK_M]
            BLOCK_M = self.block_m
            ROWS = (M_total_b + BLOCK_M - 1) // BLOCK_M
            tok_idx_grid = torch.empty((ROWS, BLOCK_M), dtype=torch.int32, device=device)
            # Fill tok_idx_grid with tok_idx repeated across rows
            for r in range(ROWS):
                idxs = torch.arange(BLOCK_M, device=device)
                tok_row = tok_idx[r * BLOCK_M : r * BLOCK_M + BLOCK_M]  # length <= BLOCK_M
                # Pad with -1 if needed
                tok_row = torch.cat([tok_row, torch.full((BLOCK_M - len(tok_row),), -1, dtype=torch.int32, device=device)])
                tok_idx_grid[r, :] = tok_row

            # Launch Triton kernel for each (b, h)
            for h_idx in range(H):
                # Allocate out vector for this head
                y = torch.zeros((N,), dtype=torch.float32, device=device)

                # Run kernel
                lse_and_output_kernel[(1,)](
                    qn_fp32[b_idx, h_idx],                # [N] fp32
                    qp_fp32[b_idx, h_idx],                # [Kp_dim] fp32
                    Kc_fp32,                               # [num_pages, N] fp32
                    Kp_fp32,                               # [num_pages, Kp_dim] fp32
                    tok_idx_grid,                          # [ROWS, BLOCK_M] int32
                    lse_fp32[b_idx, h_idx],               # *fp32 scalar
                    y,                                     # [N] fp32
                    N, Kp_dim, M_total_b, sm_scale,
                    ROWS, BLOCK_M
                )

                # Store accumulated output
                output_fp32[b_idx, h_idx] = y

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse_fp32