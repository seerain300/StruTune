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
    tok_idx_grid_ptr,  # *int32, [ROWS, BLOCK_M]
    out_ptr,     # *fp32, [N] output vector for this (b,h)
    N,           # int32
    Kp_dim,      # int32
    M_total,     # int32
    sm_scale,    # fp32
    lse_out_ptr, # *fp32, scalar output for this (b,h)
    BLOCK_M: tl.constexpr,  # chunk width
    ROWS: tl.constexpr       # number of rows (tiles) in tok_idx_grid
):
    # Accumulators for row-wise logsumexp
    row_max = -float("inf")
    sum_exp = 0.0

    # Loop over tiles; ROWS is compile-time constant
    for t in range(ROWS):
        # Load token index for this row (first column only; others are -1 after padding)
        idx = tl.load(tok_idx_grid_ptr + t * BLOCK_M + 0)
        # Validity mask
        valid = idx >= 0
        if not valid:
            # Skip invalid (padding) rows
            continue

        # Compute logits contributions for this token
        # Load qn[h, :] vector
        offs = tl.arange(0, N)
        qn_row = tl.load(qn_ptr + offs)  # [N], vectorized across N
        # Load Kc[idx, :] vector
        kc_row = tl.load(Kc_ptr + idx * N + offs)
        # Load Kp[idx, :] vector
        kp_row = tl.load(Kp_ptr + idx * Kp_dim + tl.arange(0, Kp_dim))
        # Dot products
        dot_qn = tl.sum(qn_row * kc_row, axis=0)  # scalar
        dot_qp = tl.sum(qp_ptr * kp_row, axis=0)  # scalar
        # Logits and scaled logits
        logits = dot_qn + dot_qp
        scaled = logits * sm_scale

        # Update row-wise max and sum_exp
        if scaled > row_max:
            # Rescale sum_exp when new max is found
            sum_exp = sum_exp * tl.exp(row_max - scaled) + 1.0
            row_max = scaled
        else:
            sum_exp += tl.exp(scaled - row_max)

    # Compute logsumexp and lse (base 2)
    lse_val = tl.log(sum_exp) / math.log(2.0)
    # Write lse to output scalar
    tl.store(lse_out_ptr, lse_val)

    # Second pass: accumulate output y = sum_m attn[m] * Kc[m, :]
    for t in range(ROWS):
        idx = tl.load(tok_idx_grid_ptr + t * BLOCK_M + 0)
        valid = idx >= 0
        if not valid:
            continue

        kc_row = tl.load(Kc_ptr + idx * N + tl.arange(0, N))
        # Compute scaled logits again
        qn_row = tl.load(qn_ptr + tl.arange(0, N))
        kp_row = tl.load(Kp_ptr + idx * Kp_dim + tl.arange(0, Kp_dim))
        dot_qn = tl.sum(qn_row * kc_row, axis=0)
        dot_qp = tl.sum(qp_ptr * kp_row, axis=0)
        logits = dot_qn + dot_qp
        scaled = logits * sm_scale

        attn = tl.exp(scaled - lse_val) / M_total
        # out is a vector [N]
        offs = tl.arange(0, N)
        out_vec = tl.load(out_ptr + offs)
        out_vec += attn * kc_row
        tl.store(out_ptr + offs, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, block_m=128):
        super().__init__()
        self.block_m = block_m

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices):
        # Ensure device is CUDA; Triton requires CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda \
            and kv_indptr.is_cuda and kv_indices.is_cuda, "Tensors must be on CUDA device"

        # Extract shapes
        B, H, N = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        num_pages = ckv_cache.shape[0]
        _, M_total_total = kv_indptr.shape

        # Prepare fp32 copies for computation
        qn_fp32 = q_nope.to(torch.float32)  # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32)    # [B, H, Kp_dim]

        Kc_fp32 = ckv_cache.to(torch.float32)  # [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32)  # [num_pages, Kp_dim]

        # Output buffers
        output_fp32 = torch.zeros((B, H, N), dtype=torch.float32, device=q_nope.device)
        lse_fp32 = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Constants
        sm_scale = 1.0  # default scalar; original code uses 1.0

        # For each batch b
        for b_idx in range(B):
            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item())
            M_total_b = end - start
            if M_total_b <= 0:
                # No tokens for this batch element; output zeros and lse -inf
                lse_fp32[b_idx] = -float("inf")
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total_b]

            # Build 2D token grid of shape [ROWS, BLOCK_M], pad with -1
            rows = (M_total_b + self.block_m - 1) // self.block_m
            tok_idx_grid = tok_idx.view(rows, self.block_m).contiguous()
            # Pad with -1 for remainder
            if tok_idx_grid.shape[0] * self.block_m > M_total_b:
                pad_rows = (tok_idx_grid.shape[0] + 1) // 1  # ensure last dim
                pad_len = tok_idx_grid.shape[1]
                # We already made tok_idx_grid with correct rows; no need to adjust

            # Launch Triton kernel for each head h
            for h_idx in range(H):
                qn_h = qn_fp32[b_idx, h_idx].contiguous()        # [N]
                qp_h = qp_fp32[b_idx, h_idx].contiguous()        # [Kp_dim]
                y = output_fp32[b_idx, h_idx]                    # [N], initialize to zeros by allocation

                # Run kernel; ROWS and BLOCK_M are constexpr-like (Python ints passed as kernel args)
                lse_and_output_kernel[(1,)](
                    qn_h,                          # *fp32 [N]
                    qp_h,                          # *fp32 [Kp_dim]
                    Kc_fp32,                       # *fp32 [num_pages, N]
                    Kp_fp32,                       # *fp32 [num_pages, Kp_dim]
                    tok_idx_grid,                  # *int32 [rows, block_m]
                    y,                             # *fp32 [N]
                    N,                             # int32
                    Kp_dim,                        # int32
                    M_total_b,                     # int32
                    sm_scale,                      # fp32
                    lse_fp32[b_idx].contiguous(), # *fp32 scalar pointer
                    self.block_m,                  # BLOCK_M
                    tok_idx_grid.shape[0]         # ROWS
                )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        # Return output [B, H, N] and lse [B, H]
        return output_bf16, lse_fp32