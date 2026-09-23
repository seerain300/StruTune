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
    tok_idx_ptr, # *int32, [M_total]
    lse_ptr,     # *fp32, scalar per (b,h)
    out_ptr,     # *fp32, [N]
    N,           # int32
    Kp_dim,      # int32
    M_total,     # int32
    sm_scale,    # fp32
    BLOCK_M: tl.constexpr,  # chunk size for token loop
    ROWS: tl.constexpr       # number of chunks = ceil_div(M_total, BLOCK_M)
):
    # Compute LogSumExp in the first pass
    row_max = -float("inf")
    sum_exp = 0.0

    # First pass: compute lse
    for r in range(ROWS):
        base = r * BLOCK_M
        # Loop over mm within the chunk (compile-time loop)
        for mm in range(BLOCK_M):
            idx = base + mm
            mask = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=mask, other=-1)
            # Load qn row
            qn_row = tl.load(qn_ptr + tl.arange(0, N))
            # Load Kc row
            kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=mask, other=0.0)
            # Dot product: sum_i qn_row[i] * kc_row[i]
            dot_qn_kc = tl.sum(qn_row * kc_row, axis=0)
            # Load Kp row
            kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=mask, other=0.0)
            # Compute dot(qp, Kp) if Kp_dim > 0 (handled by host passing correct dims)
            dot_qp_kp = tl.sum(qp_ptr * kp_row, axis=0)  # dummy, see below

            # Note: The above tl.sum(qn_row * kc_row) is correct. For completeness and to avoid
            # potential ambiguity, we can also compute dot_qp_kp by loading scalar elements one by one
            # since qp_ptr is 1D. Here we correct it by loading scalar using mask.
            # Compute dot(qp, Kp) scalar: loop over Kp_dim
            dot_qp_kp = 0.0
            for kd in range(Kp_dim):
                dot_qp_kp += tl.load(qp_ptr + kd) * tl.load(Kp_ptr + tok * Kp_dim + kd)

            logits = dot_qn_kc + dot_qp_kp
            logits_scaled = logits * sm_scale
            # Update row_max and sum_exp
            if mask:
                row_max = tl.maximum(row_max, logits_scaled)
                sum_exp += tl.exp(logits_scaled - row_max)

    # Compute lse: log(sum_exp) / ln(2)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)

    # Store lse to global lse_ptr (scalar)
    tl.store(lse_ptr, lse_val)

    # Second pass: compute output vector y = sum_m attention[m] * Kc[m, :]
    for r in range(ROWS):
        base = r * BLOCK_M
        for mm in range(BLOCK_M):
            idx = base + mm
            mask = idx < M_total
            tok = tl.load(tok_idx_ptr + idx, mask=mask, other=-1)
            qn_row = tl.load(qn_ptr + tl.arange(0, N))
            kc_row = tl.load(Kc_ptr + tok * N + tl.arange(0, N), mask=mask, other=0.0)
            kp_row = tl.load(Kp_ptr + tok * Kp_dim + tl.arange(0, Kp_dim), mask=mask, other=0.0)

            dot_qn_kc = tl.sum(qn_row * kc_row, axis=0)
            dot_qp_kp = 0.0
            for kd in range(Kp_dim):
                dot_qp_kp += tl.load(qp_ptr + kd) * tl.load(Kp_ptr + tok * Kp_dim + kd)

            logits = dot_qn_kc + dot_qp_kp
            logits_scaled = logits * sm_scale

            attn = tl.exp(logits_scaled - lse_val) / M_total

            # Accumulate into out_ptr: y += attn * kc_row
            # out_ptr is 1D fp32 vector [N]
            if mask:
                # Multiply each element of kc_row by attn and add to out_ptr
                for i in range(N):
                    tl.atomic_add(out_ptr + i, attn * kc_row[i])


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_m=256):
        super().__init__()
        self.sm_scale = float(sm_scale)
        self.block_m = int(block_m)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Extract shapes
        B, H, N = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        num_pages = ckv_cache.shape[0]
        # Ensure dtype and contiguity
        qn_fp32 = q_nope.to(torch.float32).contiguous()     # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()       # [B, H, Kp_dim]
        Kc_fp32 = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, N]
        Kp_fp32 = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Kp_dim]

        # Output buffers
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=q_nope.device)
        lse_fp32 = torch.full((B, H), -float("inf"), dtype=torch.float32, device=q_nope.device)

        # Compute M_total per batch
        M_list = (kv_indptr[1:] - kv_indptr[:]).tolist()
        # Handle empty case
        for b_idx in range(B):
            M_total_b = M_list[b_idx] if b_idx < len(M_list) else 0
            if M_total_b == 0:
                lse_fp32[b_idx] = -float("inf")
                output_fp32[b_idx] = torch.zeros((H, N), dtype=torch.float32, device=q_nope.device)
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[kv_indptr[b_idx]:kv_indptr[b_idx + 1]].to(torch.int32).contiguous()  # [M_total_b]
            # Number of chunks
            ROWS = (M_total_b + self.block_m - 1) // self.block_m

            # Launch Triton kernel once per (b,h)
            for h_idx in range(H):
                # Prepare lse scalar pointer
                lse_scalar = lse_fp32[b_idx, h_idx]  # scalar tensor
                # y vector for this head
                y = output_fp32[b_idx, h_idx].contiguous()  # [N]
                # Initialize y to zeros
                y.zero_()

                lse_and_output_kernel[(1,)](
                    qn_fp32[b_idx, h_idx].contiguous(),   # [N]
                    qp_fp32[b_idx, h_idx].contiguous(),   # [Kp_dim]
                    Kc_fp32,                               # [num_pages, N]
                    Kp_fp32,                               # [num_pages, Kp_dim]
                    tok_idx,                               # [M_total_b]
                    lse_scalar,                            # *fp32 scalar
                    y,                                     # *fp32 [N]
                    N, Kp_dim, M_total_b, self.sm_scale,
                    self.block_m, ROWS
                )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse_fp32


def run(*args):
    return ModelNew()(*args)
