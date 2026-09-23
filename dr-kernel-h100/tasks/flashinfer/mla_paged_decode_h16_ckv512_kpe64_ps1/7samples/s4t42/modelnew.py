import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_ptr,           # *float32, [B, N, Dc] flattened
    Kc_ptr,           # *float32, [P, Dc] squeezed
    qp_ptr,           # *float32, [B, N, Dp] flattened
    Kp_ptr,           # *float32, [P, Dp] squeezed
    tok_idx_ptr,      # *int32, [M_b]
    logits_ptr,       # *float32, [B, N, M_b] flattened (row-major: ((b*N + h)*M_b + t))
    B: tl.constexpr,
    N: tl.constexpr,
    Dc: tl.constexpr,
    Dp: tl.constexpr,
    M_b: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    # One program per (b, h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load qn and qp for this head
    qn = tl.load(qn_ptr + (pid_b * N + pid_h) * Dc + tl.arange(0, Dc))
    qp = tl.load(qp_ptr + (pid_b * N + pid_h) * Dp + tl.arange(0, Dp))

    # Loop over tokens in chunks
    for t0 in range(0, M_b, BLOCK_T):
        t_idx = t0 + tl.arange(0, BLOCK_T)
        mask = t_idx < M_b

        # Gather token indices
        tok = tl.load(tok_idx_ptr + t_idx, mask=mask, other=0)

        # Load Kc and Kp rows for these tokens
        Kc_rows = tl.load(Kc_ptr + tok * Dc + tl.arange(0, Dc), mask=mask, other=0.0)  # [BLOCK_T, Dc]
        Kp_rows = tl.load(Kp_ptr + tok * Dp + tl.arange(0, Dp), mask=mask, other=0.0)  # [BLOCK_T, Dp]

        # Compute qn @ Kc_rows.T and qp @ Kp_rows.T
        # Use manual reduction over Dc/Dp to accumulate logits
        # accum0: [BLOCK_T], accum1: [BLOCK_T]
        accum0 = tl.zeros([BLOCK_T], dtype=tl.float32)
        accum1 = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Reduce over Dc for Kc
        for d in range(0, Dc):
            accum0 += qn[d] * Kc_rows[:, d]

        # Reduce over Dp for Kp
        for dp in range(0, Dp):
            accum1 += qp[dp] * Kp_rows[:, dp]

        # Write logits into logits_ptr for this (b,h) row
        # logits_ptr linear index = ((b*N + h)*M_b + t)
        row_base = (pid_b * N + pid_h) * M_b
        t_global = t0 + tl.arange(0, BLOCK_T)
        tl.store(logits_ptr + row_base + t_global, accum0 + accum1, mask=mask)


@triton.jit
def lse_base2_kernel(
    logits_ptr,       # *float32, [B, N, M_b] flattened
    lse_ptr,          # *float32, [B, N] flattened
    B: tl.constexpr,
    N: tl.constexpr,
    M_b: tl.constexpr,
    sm_scale: tl.constexpr  # not used here since logits are already scaled in host
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = (pid_b * N + pid_h) * M_b
    logits_row = tl.load(logits_ptr + base + tl.arange(0, M_b))
    m = tl.max(logits_row)
    sum_exp = tl.sum(tl.exp(logits_row - m))
    lse_val = (m + tl.log(sum_exp)) / math.log(2.0)
    tl.store(lse_ptr + pid_b * N + pid_h, lse_val)


@triton.jit
def softmax_base2_kernel(
    logits_ptr,       # *float32, [B, N, M_b] flattened
    attn_ptr,         # *float32, [B, N, M_b] flattened
    lse_ptr,          # *float32, [B, N] flattened (per (b,h))
    B: tl.constexpr,
    N: tl.constexpr,
    M_b: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = (pid_b * N + pid_h) * M_b
    logits_row = tl.load(logits_ptr + base + tl.arange(0, M_b))
    lse_bh = tl.load(lse_ptr + pid_b * N + pid_h)
    # base-2 softmax: attn = exp((logits - m)/log(2)) * 2^(-lse)
    # Here lse = logsumexp_base2; we need sum for normalization: sum = sum(exp((logits - m)/log(2)))
    # Compute sum: sum_exp = sum(exp((logits - m)/log(2))) with m = max(logits)
    m = tl.max(logits_row)
    sum_exp = tl.sum(tl.exp((logits_row - m) / math.log(2.0)))
    inv_sum = 1.0 / sum_exp
    attn_vec = tl.exp((logits_row - m) / math.log(2.0)) * inv_sum
    tl.store(attn_ptr + base + tl.arange(0, M_b), attn_vec)


@triton.jit
def matvec_proj_kernel(
    attn_ptr,         # *float32, [B, N, M_b] flattened
    Kc_ptr,           # *float32, [P, Dc] flattened (we index via tok_idx on host; here we assume Kc is [M_b, Dc])
    out_ptr,          # *float32, [B, N, Dc] flattened
    B: tl.constexpr,
    N: tl.constexpr,
    Dc: tl.constexpr,
    M_b: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    # One program per (b, h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Prepare output vector for this (b, h)
    out_vec = tl.zeros([Dc], dtype=tl.float32)

    # Iterate over Dc in chunks
    for d0 in range(0, Dc, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        mask_d = d_idx < Dc

        # Accumulator for this chunk
        accum = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Loop over tokens t in this batch
        for t in range(0, M_b):
            attn_t = tl.load(attn_ptr + (pid_b * N + pid_h) * M_b + t)  # scalar
            Kc_vec = tl.load(Kc_ptr + t * Dc + d_idx, mask=mask_d, other=0.0)  # [BLOCK_D]
            accum += attn_t * Kc_vec

        # Store results
        tl.store(out_ptr + (pid_b * N + pid_h) * Dc + d_idx, accum, mask=mask_d)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, *unused):
        # Extract shapes
        device = q_nope.device
        B, N, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        # Squeeze cache to [P, Dc] and [P, Dp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Dp]

        # Allocate outputs
        output = torch.empty((B, N, Dc), dtype=torch.float32, device=device)
        lse = torch.empty((B, N), dtype=torch.float32, device=device)

        # Precompute total M_tot and per-batch M_b
        # kv_indptr shape is [B+1], we assume it's sorted per-batch as in original
        M_tot = int(kv_indptr[-1].item())
        # For each batch b, compute M_b and tok_idx
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M_b = end - start
            if M_b <= 0:
                # No tokens for this batch, skip
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            tok_idx = kv_indices[start:end].to(torch.int32)

            # Prepare pointers
            qn_flat = q_nope[b].to(torch.float32).contiguous().view(N * Dc)
            qp_flat = q_pe[b].to(torch.float32).contiguous().view(N * Dp)

            # Allocate logits and attn buffers for this (b,h)
            logits = torch.empty((N, M_b), dtype=torch.float32, device=device)
            attn = torch.empty((N, M_b), dtype=torch.float32, device=device)

            # Launch compute_logits_kernel: grid (B, N)
            grid = (B, N)
            compute_logits_kernel[grid](
                qn_flat, Kc_all, qp_flat, Kp_all, tok_idx, logits.view(B * N * M_b),
                B=B, N=N, Dc=Dc, Dp=Dp, M_b=M_b, BLOCK_T=128
            )

            # Compute lse per (b,h) and store
            lse_b = torch.empty((N,), dtype=torch.float32, device=device)
            grid_lse = (B, N)
            lse_base2_kernel[grid_lse](
                logits.view(B * N * M_b), lse_b,
                B=B, N=N, M_b=M_b, sm_scale=sm_scale
            )

            # Compute attn per (b,h)
            softmax_base2_kernel[grid_lse](
                logits.view(B * N * M_b), attn.view(B * N * M_b), lse_b,
                B=B, N=N, M_b=M_b
            )

            # Compute out per (b,h) via matvec_proj_kernel: Kc_sub is [M_b, Dc]
            Kc_sub = Kc_all[tok_idx]  # [M_b, Dc]
            # out is [N, Dc]; we will store into output[b] via view
            for h in range(N):
                out_row = torch.empty((Dc,), dtype=torch.float32, device=device)
                grid_proj = (1, 1)  # one program per (b,h)
                matvec_proj_kernel[grid_proj](
                    attn.view(B * N * M_b) + h * M_b, Kc_sub,
                    out_row, B=1, N=1, Dc=Dc, M_b=M_b, BLOCK_D=128
                )
                output[b, h, :] = out_row

        # Cast output to bfloat16 as in original
        output = output.to(torch.bfloat16)
        return output, lse