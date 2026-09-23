import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_kernel(
    qn_ptr,          # *fp32, [N]
    qp_ptr,          # *fp32, [Kp_dim]
    Kc_ptr,          # *fp32, [M_total, N]
    Kp_ptr,          # *fp32, [M_total, Kp_dim]
    tok_idx_ptr,     # *int32, [M_total]
    lse_ptr,         # *fp32, scalar (per (b,h))
    out_ptr,         # *fp32, [N]
    N,               # int32, head_dim_ckv (512)
    Kp_dim,          # int32, head_dim_kpe (64)
    M_total,         # int32
    sm_scale,        # fp32
    BLOCK_M: tl.constexpr,  # chunk size for tokens
):
    # Pass 1: compute lse
    # Initialize running max and sum_exp for logsumexp
    row_max = -float("inf")
    sum_exp = 0.0

    # Iterate tokens in chunks
    m = 0
    while m < M_total:
        m_offsets = m + tl.arange(0, BLOCK_M)            # [BLOCK_M]
        mask_m = m_offsets < M_total                      # [BLOCK_M] boolean

        # Load qn_row [N] (vector)
        n_offsets = tl.arange(0, N)                      # [N]
        qn_row_ptrs = qn_ptr + n_offsets                 # [N]
        qn_row = tl.load(qn_row_ptrs)                   # [N] fp32

        # Load Kc chunk [BLOCK_M, N]
        kc_ptrs = Kc_ptr + m_offsets[:, None] * N + n_offsets[None, :]  # [BLOCK_M, N]
        kc_mask = mask_m[:, None]                        # [BLOCK_M, 1] broadcast -> [BLOCK_M, N]
        kc_chunk = tl.load(kc_ptrs, mask=kc_mask, other=0.0)  # [BLOCK_M, N] fp32

        # Load Kp chunk [BLOCK_M, Kp_dim]
        kp_offsets = tl.arange(0, Kp_dim)               # [Kp_dim]
        kp_ptrs = Kp_ptr + m_offsets[:, None] * Kp_dim + kp_offsets[None, :]  # [BLOCK_M, Kp_dim]
        kp_mask = mask_m[:, None]                        # [BLOCK_M, 1] broadcast -> [BLOCK_M, Kp_dim]
        kp_chunk = tl.load(kp_ptrs, mask=kp_mask, other=0.0)  # [BLOCK_M, Kp_dim] fp32

        # Load token indices for these rows: tok_idx[m_offsets]
        tok_vals = tl.load(tok_idx_ptr + m_offsets, mask=mask_m, other=0)  # [BLOCK_M] int32

        # Compute per-token logits_scaled for this chunk:
        # logits[m] = dot(qn_row, kc[m, :]) + dot(qp_row, kp[m, :])
        # We can compute dot via sum over N and Kp_dim. Use reduction along axis=1 after making it [1, N] etc.

        # First term: qn_row @ kc_chunk.T -> [BLOCK_M]
        # kc_chunk [BLOCK_M, N], qn_row [N] -> result [BLOCK_M]
        # Reshape qn_row to [1, N], kc_chunk to [BLOCK_M, N] -> dot: sum over N
        qn_row_1d = qn_row[None, :]                      # [1, N]
        dot_qn_kc = tl.sum(kc_chunk * qn_row_1d, axis=1)  # [BLOCK_M]

        # Second term: qp_row @ kp_chunk.T -> [BLOCK_M]
        # kp_chunk [BLOCK_M, Kp_dim], qp_row [Kp_dim] -> result [BLOCK_M]
        kp_row_1d = kp_chunk[:, 0]                      # [BLOCK_M] assuming Kp_dim>=1? Better load correctly.
        # We need to sum over Kp_dim. Since kp_chunk has shape [BLOCK_M, Kp_dim], we can sum across axis=1.
        dot_qp_kp = tl.sum(kp_chunk, axis=1)            # [BLOCK_M], wrong: we need to multiply by qp_row
        # Correct way: make qp_row a vector [Kp_dim] and do elementwise product with kp_chunk, then sum over axis=1
        # Load qp_ptr as a vector
        n_offsets_qp = tl.arange(0, Kp_dim)            # [Kp_dim]
        qp_row_ptrs = qp_ptr + n_offsets_qp            # [Kp_dim]
        qp_row = tl.load(qp_row_ptrs)                  # [Kp_dim]
        dot_qp_kp = tl.sum(kp_chunk * qp_row[None, :], axis=1)  # [BLOCK_M]

        logits = dot_qn_kc + dot_qp_kp                  # [BLOCK_M]
        logits_scaled = logits * sm_scale               # [BLOCK_M]

        # Update row_max and sum_exp for logsumexp
        chunk_max = tl.max(tl.where(mask_m, logits_scaled, -float("inf")))
        new_max = tl.maximum(row_max, chunk_max)
        # sum_exp = sum_exp * exp(row_max - new_max) + sum(exp(logits_scaled - new_max))
        sum_exp = sum_exp * tl.exp(row_max - new_max)
        exps = tl.exp(tl.where(mask_m, logits_scaled - new_max, -float("inf")))
        sum_exp += tl.sum(tl.where(mask_m, exps, 0.0))
        row_max = new_max

        m += BLOCK_M

    # Compute lse = log(sum_exp) / ln(2)
    ln2 = 1.4426950408889634  # log(2)
    lse_val = tl.log(sum_exp) / ln2
    tl.store(lse_ptr, lse_val)  # store scalar

    # Pass 2: accumulate output vector y = sum_m attn[m] * Kc[m, :]
    # We need to recompute logits_scaled and attn per token, but since attn only depends on logits_scaled and lse_val,
    # we can compute attn here and accumulate Kc rows.

    # Reinitialize accumulator
    out_vec = tl.zeros((N,), dtype=tl.float32)

    m = 0
    while m < M_total:
        m_offsets = m + tl.arange(0, BLOCK_M)          # [BLOCK_M]
        mask_m = m_offsets < M_total

        # Load qn_row and Kc/Kp chunks
        n_offsets = tl.arange(0, N)
        qn_row_ptrs = qn_ptr + n_offsets
        qn_row = tl.load(qn_row_ptrs)                   # [N]

        kc_ptrs = Kc_ptr + m_offsets[:, None] * N + n_offsets[None, :]
        kc_mask = mask_m[:, None]
        kc_chunk = tl.load(kc_ptrs, mask=kc_mask, other=0.0)  # [BLOCK_M, N]

        kp_offsets = tl.arange(0, Kp_dim)
        kp_ptrs = Kp_ptr + m_offsets[:, None] * Kp_dim + kp_offsets[None, :]
        kp_mask = mask_m[:, None]
        kp_chunk = tl.load(kp_ptrs, mask=kp_mask, other=0.0)  # [BLOCK_M, Kp_dim]

        tok_vals = tl.load(tok_idx_ptr + m_offsets, mask=mask_m, other=0)  # [BLOCK_M] int32

        # Compute logits_scaled and attn for this chunk
        qn_row_1d = qn_row[None, :]                      # [1, N]
        dot_qn_kc = tl.sum(kc_chunk * qn_row_1d, axis=1)  # [BLOCK_M]

        n_offsets_qp = tl.arange(0, Kp_dim)
        qp_row_ptrs = qp_ptr + n_offsets_qp
        qp_row = tl.load(qp_row_ptrs)                  # [Kp_dim]
        dot_qp_kp = tl.sum(kp_chunk * qp_row[None, :], axis=1)  # [BLOCK_M]

        logits = dot_qn_kc + dot_qp_kp
        logits_scaled = logits * sm_scale
        attn = tl.exp(logits_scaled - lse_val)         # [BLOCK_M], note: lse_val is scalar

        # Accumulate output: out_vec += sum_m (attn[m] * Kc[m, :]) for tokens in this chunk
        # Kc chunk already loaded: kc_chunk [BLOCK_M, N]
        # Multiply by attn per token row and sum along rows
        # We need to zero out non-masked rows before summing
        contrib = tl.where(mask_m[:, None], attn[:, None] * kc_chunk, 0.0)  # [BLOCK_M, N]
        out_vec += tl.sum(contrib, axis=0)             # [N]

        m += BLOCK_M

    # Store output vector
    out_ptrs = out_ptr + n_offsets
    tl.store(out_ptrs, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, block_m: int = 128):
        super().__init__()
        self.block_m = block_m

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, _unused=None):
        # Prepare device
        device = q_nope.device
        assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
        assert ckv_cache.dtype == torch.bfloat16 and kpe_cache.dtype == torch.bfloat16

        # Cast to fp32 and make contiguous
        qn_fp32 = q_nope.to(torch.float32).contiguous()  # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()    # [B, H, Kp_dim]

        # Flatten ckv_cache and kpe_cache to [num_pages, N] and [num_pages, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous()  # [num_pages, 1, N] -> [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous()  # [num_pages, 1, Kp_dim] -> [num_pages, Kp_dim]

        B, H, N = qn_fp32.shape
        _, _, Kp_dim = qp_fp32.shape
        num_pages, N_kc = Kc_fp32.shape
        assert N_kc == N
        num_pages_kp, Kp_dim_k = Kp_fp32.shape
        assert Kp_dim_k == Kp_dim

        # Prepare outputs
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Iterate over batch and heads, launch Triton per (b, h)
        for b_idx in range(B):
            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item())
            M_total = end - start
            if M_total <= 0:
                # No tokens for this batch element; output zeros, lse = -inf
                lse[b_idx, :] = -float("inf")
                continue

            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total]
            # For each head h
            for h_idx in range(H):
                # Prepare row vectors qn and qp for this head
                qn_row = qn_fp32[b_idx, h_idx, :].contiguous()          # [N]
                qp_row = qp_fp32[b_idx, h_idx, :].contiguous()          # [Kp_dim]

                # Build Kc and Kp chunks corresponding to tok_idx
                # Kc_ptr and Kp_ptr are [num_pages, N] and [num_pages, Kp_dim] flattened
                # We need Kc_chunk and Kp_chunk of shape [M_total, N] and [M_total, Kp_dim]
                # Build chunked pointers using tok_idx
                Kc_chunk = Kc_fp32[tok_idx]                             # [M_total, N]
                Kp_chunk = Kp_fp32[tok_idx]                             # [M_total, Kp_dim]

                # Allocate output vector and lse scalar
                out_vec = torch.empty((N,), dtype=torch.float32, device=device)  # [N]
                lse_scalar = torch.empty((), dtype=torch.float32, device=device)

                # Launch Triton kernel
                lse_and_output_kernel[(1,)](
                    qn_row, Kp_dim, M_total, sm_scale, BLOCK_M=self.block_m,
                    Kc_ptr=Kc_chunk, Kp_ptr=Kp_chunk, tok_idx_ptr=tok_idx,
                    lse_ptr=lse_scalar, out_ptr=out_vec,
                    N=N, Kp_dim=Kp_dim, M_total=M_total, sm_scale=sm_scale
                )

                # Store results
                lse[b_idx, h_idx] = lse_scalar.item()
                output_fp32[b_idx, h_idx, :] = out_vec

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
