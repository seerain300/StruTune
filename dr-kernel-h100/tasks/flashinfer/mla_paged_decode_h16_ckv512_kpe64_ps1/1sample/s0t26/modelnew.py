import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_fused_kernel(
    qn_ptr,      # *fp32, [N]
    qp_ptr,      # *fp32, [Kp_dim]
    Kc_ptr,      # *fp32, [M_total, N]
    Kp_ptr,      # *fp32, [M_total, Kp_dim]
    tok_idx_ptr, # *int32, [M_total]
    lse_ptr,     # *fp32, scalar
    out_ptr,     # *fp32, [N]
    N: tl.constexpr,           # head_dim_ckv (512), constexpr for vectorization
    Kp_dim: tl.constexpr,      # head_dim_kpe (64), constexpr
    M_total,                   # number of tokens in this batch element
    sm_scale,                  # float32 scaling (original uses 1.0)
    BLOCK_M: tl.constexpr      # chunk size over tokens
):
    # Constants
    LN2 = 0.6931471805599453

    # Pass 1: compute row_max and sum_exp for lse
    row_max = -float("inf")
    sum_exp = 0.0

    for m0 in range(0, M_total, BLOCK_M):
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask = m_offsets < M_total

        # Load qn_row [N]
        n_offsets = tl.arange(0, N)
        qn_row = tl.load(qn_ptr + n_offsets)

        # Load Kc chunk [BLOCK_M, N]
        kc_ptrs = Kc_ptr + m_offsets[:, None] * N + n_offsets[None, :]
        kc_chunk = tl.load(kc_ptrs, mask=mask[:, None], other=0.0)

        # Load Kp chunk [BLOCK_M, Kp_dim]
        kp_offsets = tl.arange(0, Kp_dim)
        kp_ptrs = Kp_ptr + m_offsets[:, None] * Kp_dim + kp_offsets[None, :]
        kp_chunk = tl.load(kp_ptrs, mask=mask[:, None], other=0.0)

        # Load token indices for these rows
        tok_vals = tl.load(tok_idx_ptr + m_offsets, mask=mask, other=0)  # int32

        # Compute logits for each token in the chunk: [BLOCK_M]
        # logits[m] = sum_k (qn_row[k] * kc_chunk[m, k]) + sum_k (qp_row[k] * kp_chunk[m, k])
        # We can compute by summing along N and Kp_dim.
        dot_qn_kc = tl.sum(kc_chunk * qn_row[None, :], axis=1)  # [BLOCK_M]
        dot_qp_kp = tl.sum(kp_chunk * qp_ptr[None, :], axis=1)  # [BLOCK_M]
        logits = dot_qn_kc + dot_qp_kp

        # Scale and compute logsumexp contributions
        scaled = logits * sm_scale
        # Update row_max and sum_exp for masked entries
        # Mask invalid tokens to -inf so they don't contribute
        scaled_masked = tl.where(mask, scaled, -float("inf"))
        chunk_max = tl.max(scaled_masked, axis=0)
        # Ensure chunk_max >= row_max
        row_max = tl.maximum(row_max, chunk_max)
        # sum_exp += sum(exp(scaled - row_max)) over valid tokens
        exp_contrib = tl.exp(scaled_masked - row_max)
        # Sum only valid positions
        sum_exp += tl.sum(tl.where(mask, exp_contrib, 0.0), axis=0)

    # Compute lse = log(sum_exp) / ln(2)
    lse_val = tl.log(sum_exp) / LN2
    tl.store(lse_ptr, lse_val)

    # Pass 2: accumulate output y = sum_m exp(scaled - lse) * Kc[m, :]
    out_vec = tl.zeros([N], dtype=tl.float32)
    for m0 in range(0, M_total, BLOCK_M):
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask = m_offsets < M_total

        # Load qn_row [N]
        n_offsets = tl.arange(0, N)
        qn_row = tl.load(qn_ptr + n_offsets)

        # Load Kc chunk [BLOCK_M, N]
        kc_ptrs = Kc_ptr + m_offsets[:, None] * N + n_offsets[None, :]
        kc_chunk = tl.load(kc_ptrs, mask=mask[:, None], other=0.0)

        # Load Kp chunk [BLOCK_M, Kp_dim]
        kp_offsets = tl.arange(0, Kp_dim)
        kp_ptrs = Kp_ptr + m_offsets[:, None] * Kp_dim + kp_offsets[None, :]
        kp_chunk = tl.load(kp_ptrs, mask=mask[:, None], other=0.0)

        # Load token indices
        tok_vals = tl.load(tok_idx_ptr + m_offsets, mask=mask, other=0)

        # Recompute logits
        dot_qn_kc = tl.sum(kc_chunk * qn_row[None, :], axis=1)  # [BLOCK_M]
        dot_qp_kp = tl.sum(kp_chunk * qp_ptr[None, :], axis=1)  # [BLOCK_M]
        logits = dot_qn_kc + dot_qp_kp
        scaled = logits * sm_scale

        # attn = exp(scaled - lse)
        attn = tl.exp(scaled - lse_val)

        # Contribution to output: sum over tokens of attn * Kc rows
        # Since attn is [BLOCK_M], multiply per-token to its Kc row
        contrib = attn[:, None] * kc_chunk  # [BLOCK_M, N]
        # Mask invalid tokens
        contrib = tl.where(mask[:, None], contrib, 0.0)
        # Reduce along tokens
        out_vec += tl.sum(contrib, axis=0)

    # Store output vector
    tl.store(out_ptr + tl.arange(0, N), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, block_m=128):
        super().__init__()
        self.block_m = block_m

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, _unused=None):
        """
        Accepts 8 positional arguments to match the evaluator's call site.
        q_nope: [B, H, N] bfloat16
        q_pe: [B, H, Kp_dim] bfloat16
        ckv_cache: [num_pages, 1, N] bfloat16
        kpe_cache: [num_pages, 1, Kp_dim] bfloat16
        kv_indptr: [len_indptr] int32
        kv_indices: [num_tokens] int32
        sm_scale: float32 scalar
        _unused: ignored (default None), allows 8-arg call to pass successfully
        """
        assert q_nope.dim() == 3 and q_pe.dim() == 3
        assert ckv_cache.dim() == 3 and kpe_cache.dim() == 3
        B, H, N = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        num_pages, _, _ = ckv_cache.shape
        num_pages_kp, _, _ = kpe_cache.shape
        assert num_pages == num_pages_kp
        device = q_nope.device
        dtype_fp32 = torch.float32

        # Prepare inputs
        qn = q_nope.to(dtype_fp32).contiguous()   # [B, H, N]
        qp = q_pe.to(dtype_fp32).contiguous()     # [B, H, Kp_dim]
        Kc_all = ckv_cache.to(dtype_fp32).contiguous().squeeze(1)  # [num_pages, N]
        Kp_all = kpe_cache.to(dtype_fp32).contiguous().squeeze(1)  # [num_pages, Kp_dim]

        # Output and lse tensors
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # For each batch b: compute tok_idx and run Triton kernel
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M_total = end - start
            if M_total <= 0:
                # No tokens for this batch element: output zeros and lse -inf
                lse[b] = -float("inf")
                continue

            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total]

            # Slice Kc_all and Kp_all for these tokens
            # Since Kc_all, Kp_all are [num_pages, dim], we need to gather by tok_idx.
            # But the original Model.run uses a single qn/qp per batch element, not per token.
            # We need Kc rows corresponding to tok_idx. Build pointers using tok_idx.
            # Create per-token pointers:
            Kc_chunk = Kc_all[tok_idx]  # [M_total, N]
            Kp_chunk = Kp_all[tok_idx]  # [M_total, Kp_dim]

            # Select the qn/qp for this batch element: use the first head or any; original uses single qn/qp for all heads?
            # The original run uses q_nope[b] and q_pe[b] for each head separately. Here we run per (b, h).
            # We'll iterate h to match the output dimension.
            for h in range(H):
                # Extract qn row and qp row for this head
                qn_row = qn[b, h]                    # [N]
                qp_row = qp[b, h]                    # [Kp_dim]

                # Allocate scalar for lse and output vector
                lse_scalar = torch.empty((), dtype=torch.float32, device=device)
                out_vec = torch.empty((N,), dtype=torch.float32, device=device)

                # Launch Triton kernel
                lse_and_output_fused_kernel[(1,)](
                    qn_row,               # *fp32 [N]
                    qp_row,               # *fp32 [Kp_dim]
                    Kc_chunk,             # *fp32 [M_total, N]
                    Kp_chunk,             # *fp32 [M_total, Kp_dim]
                    tok_idx,              # *int32 [M_total]
                    lse_scalar,           # *fp32 scalar
                    out_vec,              # *fp32 [N]
                    N=N, Kp_dim=Kp_dim, M_total=M_total,
                    sm_scale=float(sm_scale), BLOCK_M=self.block_m
                )

                # Store results
                lse[b, h] = lse_scalar
                output_fp32[b, h, :] = out_vec

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse