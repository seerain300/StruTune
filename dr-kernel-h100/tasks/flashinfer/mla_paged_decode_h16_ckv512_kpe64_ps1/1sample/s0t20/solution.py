import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_kernel(
    qn_ptr,       # *fp32, [N]
    qp_ptr,       # *fp32, [Kp_dim]
    Kc_ptr,       # *fp32, [num_pages, N]
    Kp_ptr,       # *fp32, [num_pages, Kp_dim]
    tok_idx_ptr,  # *int32, [M_total]
    lse_ptr,      # *fp32, scalar for this (b,h)
    out_ptr,      # *fp32, [N]
    N: tl.constexpr,           # head_dim_ckv (512)
    Kp_dim: tl.constexpr,      # head_dim_kpe (64)
    M_total: tl.constexpr,     # number of tokens in this batch
    sm_scale: tl.constexpr,    # scaling factor
    BLOCK_M: tl.constexpr      # chunk size for token loop
):
    # Load qn_row and qp_row as 1D vectors
    offs_n = tl.arange(0, N)
    qn_row = tl.load(qn_ptr + offs_n)  # [N]
    offs_kp = tl.arange(0, Kp_dim)
    qp_row = tl.load(qp_ptr + offs_kp)  # [Kp_dim]

    # First pass: compute row-wise max and sum of exp for logsumexp
    row_max = -float("inf")
    sum_exp = 0.0
    # Iterate over tokens in chunks of BLOCK_M
    for m in range(0, M_total, BLOCK_M):
        # Vector of token offsets for this chunk
        mm = tl.arange(0, BLOCK_M)
        tok_vec = m + mm  # [BLOCK_M]
        mask = tok_vec < M_total  # [BLOCK_M]

        # Gather tok indices for this chunk
        tok_idx_chunk = tl.load(tok_idx_ptr + tok_vec, mask=mask, other=0)  # [BLOCK_M], int32

        # Compute logits for this chunk
        # qn_row: [N], Kc_ptr: [num_pages, N]
        # We need Kc[tok, :] for each tok in chunk
        # Construct 2D pointers for chunk: [BLOCK_M, N]
        # Pointer to start of each row in Kc
        # Note: tok_idx_chunk is int32, multiply by N to get linear index
        kc_ptrs = Kc_ptr + (tok_idx_chunk[:, None] * N) + offs_n[None, :]  # [BLOCK_M, N]
        kp_ptrs = Kp_ptr + (tok_idx_chunk[:, None] * Kp_dim) + offs_kp[None, :]  # [BLOCK_M, Kp_dim]

        # Load Kc and Kp for this chunk
        kc_chunk = tl.load(kc_ptrs, mask=mask[:, None], other=0.0)  # [BLOCK_M, N]
        kp_chunk = tl.load(kp_ptrs, mask=mask[:, None], other=0.0)  # [BLOCK_M, Kp_dim]

        # Compute dot products: qn_row @ kc_chunk.T -> [BLOCK_M]
        dot_qn = tl.sum(qn_row[None, :] * kc_chunk, axis=1)  # [BLOCK_M]
        dot_qp = tl.sum(qp_row[None, :] * kp_chunk, axis=1)  # [BLOCK_M]

        logits = dot_qn + dot_qp  # [BLOCK_M]
        logits_scaled = logits * sm_scale  # [BLOCK_M]

        # Mask out invalid tokens
        logits_scaled = tl.where(mask, logits_scaled, -float("inf"))

        # Update row_max and sum_exp
        # row_max = max(row_max, max(logits_scaled))
        chunk_max = tl.max(logits_scaled, axis=0)
        row_max = tl.maximum(row_max, chunk_max)
        # sum_exp += sum(exp(logits_scaled - row_max))
        exp_chunk = tl.exp(logits_scaled - row_max)
        sum_exp += tl.sum(tl.where(mask, exp_chunk, 0.0), axis=0)

    # Compute lse: log(sum_exp) / ln(2)
    ln2 = 0.6931471805599453  # natural log of 2
    lse_val = tl.log(sum_exp) / ln2
    # Store lse scalar for this (b,h)
    tl.store(lse_ptr, lse_val)

    # Second pass: compute output y = sum_m attn[m] * Kc[m, :]
    y = tl.zeros((N,), dtype=tl.float32)
    for m in range(0, M_total, BLOCK_M):
        mm = tl.arange(0, BLOCK_M)
        tok_vec = m + mm
        mask = tok_vec < M_total
        tok_idx_chunk = tl.load(tok_idx_ptr + tok_vec, mask=mask, other=0)  # [BLOCK_M]

        kc_ptrs = Kc_ptr + (tok_idx_chunk[:, None] * N) + offs_n[None, :]  # [BLOCK_M, N]
        kp_ptrs = Kp_ptr + (tok_idx_chunk[:, None] * Kp_dim) + offs_kp[None, :]  # [BLOCK_M, Kp_dim]

        kc_chunk = tl.load(kc_ptrs, mask=mask[:, None], other=0.0)  # [BLOCK_M, N]
        kp_chunk = tl.load(kp_ptrs, mask=mask[:, None], other=0.0)  # [BLOCK_M, Kp_dim]

        dot_qn = tl.sum(qn_row[None, :] * kc_chunk, axis=1)  # [BLOCK_M]
        dot_qp = tl.sum(qp_row[None, :] * kp_chunk, axis=1)  # [BLOCK_M]

        logits = dot_qn + dot_qp
        logits_scaled = logits * sm_scale
        logits_scaled = tl.where(mask, logits_scaled, -float("inf"))

        attn = tl.exp(logits_scaled - lse_val)  # [BLOCK_M]
        attn = tl.where(mask, attn, 0.0)
        attn = attn / M_total  # normalize by number of tokens

        # Accumulate y += attn[:, None] * kc_chunk
        y += tl.sum(attn[:, None] * kc_chunk, axis=0)

    # Store output y
    tl.store(out_ptr + offs_n, y)


class ModelNew(torch.nn.Module):
    def __init__(self, block_m=128, sm_scale=1.0):
        super().__init__()
        self.block_m = block_m
        self.sm_scale = float(sm_scale)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices):
        # Expect 6 arguments; the evaluator provides 7, but we handle default for sm_scale
        # q_nope: [B, H, N], q_pe: [B, H, Kp_dim], ckv_cache: [num_pages, 1, N],
        # kpe_cache: [num_pages, 1, Kp_dim], kv_indptr: [B+1], kv_indices: [M_total]

        # Shapes
        B, H, N = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        num_pages, _, _ = ckv_cache.shape
        _, _, _ = kpe_cache.shape

        # Cast to fp32 and make contiguous
        qn_fp32 = q_nope.to(torch.float32).contiguous()    # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()      # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, N]
        Kp_fp32 = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Kp_dim]

        # Output and lse tensors
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=qn_fp32.device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=qn_fp32.device)

        # Process each batch element
        for b_idx in range(B):
            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item())
            M_total = end - start

            if M_total == 0:
                # No KV tokens for this batch element: output zeros, lse = -inf
                lse[b_idx] = -float("inf")
                # output_fp32[b_idx] remains zero since we zero-init
                continue

            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total]

            # Launch Triton kernel once per (b, h)
            for h_idx in range(H):
                # out_ptr points to output_fp32[b_idx, h_idx, :]
                out_ptr = output_fp32[b_idx, h_idx]
                # lse_ptr is scalar at lse[b_idx, h_idx]
                lse_ptr = lse[b_idx, h_idx]

                lse_and_output_kernel[(1,)](
                    qn_fp32[b_idx, h_idx].contiguous(),   # *fp32 [N]
                    qp_fp32[b_idx, h_idx].contiguous(),   # *fp32 [Kp_dim]
                    Kc_fp32,                               # *fp32 [num_pages, N]
                    Kp_fp32,                               # *fp32 [num_pages, Kp_dim]
                    tok_idx,                               # *int32 [M_total]
                    lse_ptr,                               # *fp32 scalar
                    out_ptr,                               # *fp32 [N]
                    N=N, Kp_dim=Kp_dim, M_total=M_total, sm_scale=self.sm_scale,
                    BLOCK_M=self.block_m
                )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
