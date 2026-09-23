import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_kernel(
    qn_ptr,      # *fp32, pointer to q_nope[b, h, :] contiguous row of length N
    qp_ptr,      # *fp32, pointer to q_pe[b, h, :] contiguous row of length Kp_dim
    Kc_ptr,      # *fp32, [num_pages, N] gathered into tok_idx
    Kp_ptr,      # *fp32, [num_pages, Kp_dim] gathered into tok_idx
    tok_idx_ptr, # *int32, [M_total]
    lse_ptr,     # *fp32, [B, H]
    N,           # head_dim_ckv (512)
    Kp_dim,      # head_dim_kpe (64)
    M_total,     # number of tokens for this batch element
    sm_scale,    # float32 scaling
    b, h         # batch and head indices
):
    # Accumulate row-wise max and sumexp for logsumexp
    row_max = -float("inf")
    sum_exp = 0.0

    # Loop over tokens
    m = 0
    while m < M_total:
        tok = tl.load(tok_idx_ptr + m)  # int32 index into cached Kc/Kp

        # Load qn[h, :] and Kc[tok, :]
        qn_row = tl.load(qn_ptr + tl.arange(0, N), mask=False, other=0.0)      # [N]
        kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=False, other=0.0)  # [N]
        dot_qn = tl.sum(qn_row * kc_row, axis=0)  # scalar

        # Load qp[h, :] and Kp[tok, :]
        q_row = tl.load(qp_ptr + tl.arange(0, Kp_dim), mask=False, other=0.0)   # [Kp_dim]
        kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=False, other=0.0)  # [Kp_dim]
        dot_qp = tl.sum(q_row * kp_row, axis=0)  # scalar

        logits = dot_qn + dot_qp
        logits_scaled = logits * sm_scale
        row_max = tl.maximum(row_max, logits_scaled)
        sum_exp += tl.exp(logits_scaled - row_max)

        m += 1

    lse_val = tl.log(sum_exp) / tl.log(2.0)  # convert to base-2 logsumexp
    # Store lse[b, h]
    # lse_ptr is flattened [B, H], so index is b * H + h
    tl.store(lse_ptr + b * 16 + h, lse_val)


@triton.jit
def output_gemv_kernel(
    qn_ptr,      # *fp32, pointer to q_nope[b, h, :] contiguous row of length N
    qp_ptr,      # *fp32, pointer to q_pe[b, h, :] contiguous row of length Kp_dim
    Kc_ptr,      # *fp32, [num_pages, N] gathered into tok_idx
    Kp_ptr,      # *fp32, [num_pages, Kp_dim] gathered into tok_idx
    tok_idx_ptr, # *int32, [M_total]
    output_ptr,  # *fp32, [B, H, N] output buffer (fp32)
    N,           # head_dim_ckv (512)
    Kp_dim,      # head_dim_kpe (64)
    M_total,     # number of tokens for this batch element
    sm_scale,    # float32 scaling
    b, h         # batch and head indices
):
    # Compute output[h, :] = softmax(logits_scaled) @ Kc[:, :]
    # We do not store attention; compute attn[m] on the fly and accumulate y.
    y = tl.zeros([N], dtype=tl.float32)

    m = 0
    while m < M_total:
        tok = tl.load(tok_idx_ptr + m)

        qn_row = tl.load(qn_ptr + tl.arange(0, N), mask=False, other=0.0)      # [N]
        kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=False, other=0.0)  # [N]
        dot_qn = tl.sum(qn_row * kc_row, axis=0)

        q_row = tl.load(qp_ptr + tl.arange(0, Kp_dim), mask=False, other=0.0)   # [Kp_dim]
        kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=False, other=0.0)  # [Kp_dim]
        dot_qp = tl.sum(q_row * kp_row, axis=0)

        logits = dot_qn + dot_qp
        logits_scaled = logits * sm_scale
        # For softmax we need row max to normalize
        # We don't have accumulated max here; recompute max across remaining tokens including this one.
        # To do proper softmax, we need the full set; but since we don't store attention, we recompute
        # an approximate max by adding this logits_scaled if larger, but that's incorrect. Instead,
        # recompute sum_exp and max here. However, since this is per-token, we can't get correct
        # normalization without storing attention or recompute full lse for each m.
        # To ensure correctness, this kernel should be avoided and lse computed in kernel 1.
        # We therefore recompute lse here too to get proper attention. This is acceptable for correctness.

        # Recompute row-wise max and sum_exp from scratch (per-token softmax requires full max).
        row_max = -float("inf")
        sum_exp = 0.0
        i = 0
        while i < M_total:
            t = tl.load(tok_idx_ptr + i)
            qn_row2 = tl.load(qn_ptr + tl.arange(0, N), mask=False, other=0.0)
            kc_row2 = tl.load(Kc_ptr + t * N + tl.arange(0, N), mask=False, other=0.0)
            dot_qn2 = tl.sum(qn_row2 * kc_row2, axis=0)
            q_row2 = tl.load(qp_ptr + tl.arange(0, Kp_dim), mask=False, other=0.0)
            kp_row2 = tl.load(Kp_ptr + t * Kp_dim + tl.arange(0, Kp_dim), mask=False, other=0.0)
            dot_qp2 = tl.sum(q_row2 * kp_row2, axis=0)
            log2 = dot_qn2 + dot_qp2
            log2 = log2 * sm_scale
            row_max = tl.maximum(row_max, log2)
            sum_exp += tl.exp(log2 - row_max)
            i += 1
        lse_val = tl.log(sum_exp) / tl.log(2.0)

        attn = tl.exp(logits_scaled - lse_val)  # scalar
        y += attn * kc_row

        m += 1

    # Store y into output[b, h, :]
    base = b * 16 * N + h * N
    # Write y over N
    offs = tl.arange(0, N)
    tl.store(output_ptr + base + offs, y)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0):
        super().__init__()
        self.sm_scale = float(sm_scale)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale=None):
        """
        q_nope: [B, 16, 512] bfloat16
        q_pe: [B, 16, 64] bfloat16
        ckv_cache: [num_pages, 1, 512] bfloat16
        kpe_cache: [num_pages, 1, 64] bfloat16
        kv_indptr: [B+1] int32
        kv_indices: [M_total] int32
        sm_scale: float
        Returns: (output [B, 16, 512] bfloat16, lse [B, 16] float32)
        """
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        N = q_nope.shape[2]
        Kp_dim = q_pe.shape[2]
        assert q_pe.shape[1] == H and q_pe.shape[0] == B
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
        # Ensure tensors are contiguous and on CUDA
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Inputs must be on CUDA device"
        assert kv_indptr.is_cuda and kv_indices.is_cuda, "kv_indptr and kv_indices must be on CUDA"
        # Prepare cached Kc/Kp in fp32 for compute stability
        Kc_fp32 = ckv_cache.squeeze(1).to(torch.float32)
        Kp_fp32 = kpe_cache.squeeze(1).to(torch.float32)
        qn_fp32 = q_nope.to(torch.float32)  # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32)    # [B, H, Kp_dim]

        # Output buffers
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            # Read start/end and compute M_total
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            M_total = max(page_end - page_beg, 0)
            if M_total == 0:
                # No tokens for this batch element: output zeros, lse -inf
                lse[b] = -float("inf")
                continue

            # Gather token indices
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)

            # Launch Triton kernels per head
            for h in range(H):
                # Kernel 1: compute lse for this (b, h)
                lse_kernel[(1,)](
                    qn_fp32[b, h],                # pointer to qn row (length N)
                    qp_fp32[b, h],                # pointer to qp row (length Kp_dim)
                    Kc_fp32,                      # pointer to Kc
                    Kp_fp32,                      # pointer to Kp
                    tok_idx,                      # pointer to tok_idx
                    lse,                          # pointer to lse[b, h]
                    N, Kp_dim, M_total, self.sm_scale, b, h,
                    num_warps=4, num_stages=2
                )

                # Kernel 2: compute output[h] = attn @ Kc
                output_gemv_kernel[(1,)](
                    qn_fp32[b, h],
                    qp_fp32[b, h],
                    Kc_fp32,
                    Kp_fp32,
                    tok_idx,
                    output_fp32[b, h],
                    N, Kp_dim, M_total, self.sm_scale, b, h,
                    num_warps=4, num_stages=2
                )

        # Cast output to bfloat16 to match original
        output_bf16 = output_fp32.to(torch.bfloat16)
        # Return output [B, 16, 512] bfloat16 and lse [B, 16] float32
        return output_bf16, lse