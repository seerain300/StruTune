import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_kernel(
    qn_ptr,      # *fp32, [N]   per (b,h) row
    qp_ptr,      # *fp32, [Kp_dim] per (b,h) row
    Kc_ptr,      # *fp32, [TOTAL_PAGES, N]
    Kp_ptr,      # *fp32, [TOTAL_PAGES, Kp_dim]
    tok_idx_ptr, # *int32, [M_total]
    lse_ptr,     # *fp32, scalar for this (b,h)
    out_ptr,     # *fp32, [N], output vector for this (b,h)
    N,           # int32, head_dim_ckv
    Kp_dim,      # int32, head_dim_kpe
    M_total,     # int32, number of tokens for this batch element
    sm_scale,    # fp32, scaling factor
    BLOCK_M: tl.constexpr,  # chunk size for tokens
    ROWS: tl.constexpr       # number of chunks
):
    # Compute lse: logsumexp over tokens of scaled logits
    row_max = -float("inf")
    sum_exp = 0.0
    for r in range(ROWS):
        base = r * BLOCK_M
        for mm in range(BLOCK_M):
            idx = base + mm
            mask = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=mask, other=-1)  # token index

            # Load Kc row for this token: Kc[tok, :]
            kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=mask, other=0.0)  # [N]
            # Load Kp row for this token: Kp[tok, :]
            kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=mask, other=0.0)  # [Kp_dim]

            # Load qn and qp rows for current head: qn[h, :] and qp[h, :]
            # We access qn_ptr and qp_ptr as 1D vectors of length N and Kp_dim respectively.
            qn_row = tl.load(qn_ptr + tl.arange(0, N))  # [N]
            qp_row = tl.load(qp_ptr + tl.arange(0, Kp_dim))  # [Kp_dim]

            # Compute logits = qn_row @ Kc[tok, :] + qp_row @ Kp[tok, :]
            # Reduce dot products
            dot_ckv = tl.sum(qn_row * kc_row)            # scalar
            dot_kpe = tl.sum(qp_row * kp_row)           # scalar
            logits = dot_ckv + dot_kpe                  # scalar
            logits_scaled = logits * sm_scale           # scalar

            # Accumulate row-wise max and sum_exp
            if mask:
                row_max = tl.maximum(row_max, logits_scaled)
                sum_exp += tl.exp(logits_scaled - row_max)

    # Compute lse
    # lse = log(sum_exp) / ln(2)
    lse_val = tl.log(sum_exp) / math.log(2.0)
    # Store lse to lse_ptr (scalar)
    # Note: Triton expects pointer; we write to address provided
    # Triton will allow scalar write to pointer
    # No need to do pointer arithmetic here; pass lse_ptr as pointer to scalar on host
    tl.store(lse_ptr, lse_val)

    # Second pass: compute output = sum_m attention[m] * Kc[m, :]
    # attention[m] = exp((logits_scaled[m] - lse) / M_total)
    for r in range(ROWS):
        base = r * BLOCK_M
        for mm in range(BLOCK_M):
            idx = base + mm
            mask = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=mask, other=-1)

            kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=mask, other=0.0)  # [N]
            kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=mask, other=0.0)  # [Kp_dim]

            qn_row = tl.load(qn_ptr + tl.arange(0, N))
            qp_row = tl.load(qp_ptr + tl.arange(0, Kp_dim))

            dot_ckv = tl.sum(qn_row * kc_row)
            dot_kpe = tl.sum(qp_row * kp_row)
            logits = dot_ckv + dot_kpe
            logits_scaled = logits * sm_scale

            attn = tl.exp(logits_scaled - lse_val) / M_total  # scalar

            # Accumulate into out vector: out += attn * Kc[tok, :]
            # Kc[tok, :] is loaded as kc_row; multiply by attn (scalar), then add
            out_ptr_vec = out_ptr + tl.arange(0, N)  # [N] vector pointer
            tl.store(out_ptr_vec, tl.load(out_ptr_vec) + attn * kc_row, mask=mask)

    # Done


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Meta-parameters; you can tune these
        self.block_m = 128  # token chunk size
        self.rows = 8       # number of chunks (tunable, e.g., 8 for typical M_total ~ 1k)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        B, H, N = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        TOTAL_PAGES = ckv_cache.shape[0]

        # Cast inputs to fp32 and make contiguous
        qn_fp32 = q_nope.to(torch.float32).contiguous()    # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()      # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous() # [TOTAL_PAGES, N]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous() # [TOTAL_PAGES, Kp_dim]

        # Output buffers (fp32) and lse (fp32)
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=q_nope.device)
        lse_fp32 = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Compute M_total per batch
        M_total_list = (kv_indptr[1:] - kv_indptr[:]).tolist()
        # For each batch element
        for b_idx in range(B):
            M_total_b = int(M_total_list[b_idx])
            # If no tokens, leave output zeros, lse -inf
            if M_total_b == 0:
                lse_fp32[b_idx] = float("-inf")
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[kv_indptr[b_idx]: kv_indptr[b_idx + 1]].to(torch.int32).contiguous()  # [M_total_b]

            # Build 2D token grid for Triton: shape [ROWS, BLOCK_M]
            # ROWS = ceil_div(M_total_b, BLOCK_M)
            ROWS = (M_total_b + self.block_m - 1) // self.block_m
            # Create grid indices
            # We need to form a 2D grid of indices: [ROWS, BLOCK_M]
            # However, Triton kernels can't directly take 2D indices; instead, we pass tok_idx and ROWS, and
            # inside kernel, iterate r in range(ROWS), then mm in range(BLOCK_M) using mask.
            # So we only need tok_idx for masked loads.

            # Launch Triton kernel once per (b, h)
            for h_idx in range(H):
                # Allocate output vector and zero it
                y = torch.zeros((N,), dtype=torch.float32, device=q_nope.device)

                # lse_ptr is a scalar pointer; Triton can write scalar to it
                lse_scalar = torch.empty((), dtype=torch.float32, device=q_nope.device)

                # Pass scalar pointers for qn and qp rows (use flattened views)
                qn_row = qn_fp32[b_idx, h_idx]       # [N]
                qp_row = qp_fp32[b_idx, h_idx]       # [Kp_dim]

                lse_and_output_kernel[(1,)](
                    qn_row,                            # *fp32 [N]
                    qp_row,                            # *fp32 [Kp_dim]
                    Kc_fp32,                           # *fp32 [TOTAL_PAGES, N]
                    Kp_fp32,                           # *fp32 [TOTAL_PAGES, Kp_dim]
                    tok_idx,                           # *int32 [M_total_b]
                    lse_scalar,                        # *fp32 scalar
                    y,                                 # *fp32 [N]
                    N, Kp_dim, M_total_b, sm_scale,   # scalars
                    self.block_m, ROWS                # meta-params
                )

                # Store output for this (b, h)
                output_fp32[b_idx, h_idx] = y

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse_fp32


def run(*args):
    return ModelNew()(*args)
