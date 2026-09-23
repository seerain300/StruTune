import math
import torch

import triton
import triton.language as tl


@triton.jit
def _lse_per_head_kernel(
    q_nope_ptr,          # *f32, shape [B, H, D1], contiguous
    q_pe_ptr,            # *f32, shape [B, H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    lse_out_ptr,         # *f32, shape [B, H], contiguous
    B: tl.int32,         # batch size
    H: tl.int32,         # number of heads
    D1: tl.constexpr,    # 512
    D2: tl.constexpr,    # 64
    L_tokens: tl.int32,  # runtime per b
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    # One Triton program per batch element b
    b = tl.program_id(axis=0)

    # Loop over heads
    for h in range(0, H):
        # Load q vectors for this head (q_nope[b, h, :] and q_pe[b, h, :])
        qn = tl.load(q_nope_ptr + b * (H * D1) + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_ptr + b * (H * D2) + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Initialize per-column max and sum across tokens
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        # Tile over tokens with static inner loop
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t0 = tile * BLOCK_T
            for tt in tl.static_range(0, BLOCK_T):
                t = t0 + tt
                valid = t < L_tokens
                # Load Kc_row and Kp_row (1D vector loads) with masks
                Kc_row = tl.load(Kc_sub_ptr + t * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
                Kp_row = tl.load(Kp_sub_ptr + t * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

                # Compute scalar logits for this token
                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                # Update per-column max and sum (masked by valid)
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Store lse[b, h] = token_max_vec + log(token_sum_vec) / ln(2)
        lse_val = token_max_vec + tl.log(token_sum_vec) * (1.4426950408889634)  # 1 / ln(2)
        tl.store(lse_out_ptr + b * H + h, lse_val)


@triton.jit
def _compute_output_kernel(
    q_nope_ptr,          # *f32, shape [B, H, D1], contiguous
    q_pe_ptr,            # *f32, shape [B, H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    output_ptr,          # *f32, shape [B, H, D1], contiguous
    lse_out_ptr,         # *f32, shape [B, H], contiguous (optional; recomputed here)
    B: tl.int32,         # batch size
    H: tl.int32,         # number of heads
    D1: tl.constexpr,    # 512
    D2: tl.constexpr,    # 64
    L_tokens: tl.int32,  # runtime per b
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    # One Triton program per batch element b
    b = tl.program_id(axis=0)

    # For each head, compute lse to get softmax scaling, then accumulate output
    for h in range(0, H):
        qn = tl.load(q_nope_ptr + b * (H * D1) + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_ptr + b * (H * D2) + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Initialize per-column max and sum across tokens
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        # First pass: compute token_max_vec and token_sum_vec
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t0 = tile * BLOCK_T
            for tt in tl.static_range(0, BLOCK_T):
                t = t0 + tt
                valid = t < L_tokens
                Kc_row = tl.load(Kc_sub_ptr + t * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
                Kp_row = tl.load(Kp_sub_ptr + t * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                logits_scalar = (dot1 + dot2) * sm_scale
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Second pass: accumulate output
        out_vec = tl.zeros((D1,), dtype=tl.float32)
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t0 = tile * BLOCK_T
            for tt in tl.static_range(0, BLOCK_T):
                t = t0 + tt
                valid = t < L_tokens
                Kc_row = tl.load(Kc_sub_ptr + t * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
                Kp_row = tl.load(Kp_sub_ptr + t * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                logits_scalar = (dot1 + dot2) * sm_scale
                attn = tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0) / token_sum_vec
                out_vec += attn * Kc_row

        # Store output for this head
        tl.store(output_ptr + b * (H * D1) + h * D1 + tl.arange(0, D1), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA; Triton requires CUDA
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Triton requires CUDA tensors"

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        D1 = q_nope.shape[2]  # 512
        D2 = q_pe.shape[2]    # 64

        # Squeeze caches to per-token
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D1]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, D2]

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, D1), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch kernels per batch element b
        grid = (batch_size,)
        BLOCK_T = 1024  # constexpr tile size; Triton will unroll static loops

        for b in range(batch_size):
            # Compute L_tokens and gather Kc_sub, Kp_sub for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(0, page_end - page_beg)
            if L_tokens == 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            idxs = kv_indices[page_beg:page_end].to(torch.int64)  # [L_tokens]
            Kc_sub = Kc_all[idxs]  # [L_tokens, D1]
            Kp_sub = Kp_all[idxs]  # [L_tokens, D2]
            Kc_sub = Kc_sub.contiguous()
            Kp_sub = Kp_sub.contiguous()

            # Call lse kernel for this batch element
            _lse_per_head_kernel[grid](
                q_nope, q_pe, Kc_sub, Kp_sub, lse[b:b+1],
                B=batch_size, H=num_qo_heads, D1=D1, D2=D2, L_tokens=L_tokens, sm_scale=sm_scale, BLOCK_T=BLOCK_T,
            )

            # Call output kernel for this batch element
            _compute_output_kernel[grid](
                q_nope, q_pe, Kc_sub, Kp_sub, output[b:b+1], lse[b:b+1],
                B=batch_size, H=num_qo_heads, D1=D1, D2=D2, L_tokens=L_tokens, sm_scale=sm_scale, BLOCK_T=BLOCK_T,
            )

        # Cast output to bfloat16 to match original Model's output dtype
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
