import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_lse_per_head_kernel(
    q_nope_rows_ptr,     # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,       # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [D2], but here used as [L_tokens, D2] contiguous
    lse_out_ptr,         # *f32, shape [B, H], contiguous
    H: tl.int32,         # number of heads (runtime)
    D1: tl.constexpr,    # 512
    D2: tl.constexpr,    # 64
    L_tokens: tl.int32,  # runtime
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    # One Triton program per batch element b
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for head h as 1D vectors
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Compute per-column max across tokens and sum of exp(logits - max)
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        num_tiles = (L_tokens + BLOCK_T - 1) // BLOCK_T
        for tile in range(0, num_tiles):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T], runtime vector
            mask_t = t_idx < L_tokens
            # Load Kc_rows: shape [BLOCK_T, D1]
            Kc_rows = tl.load(
                Kc_sub_ptr + t_idx[:, None] * D1 + tl.arange(0, D1)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D1]
            # Load Kp_rows: shape [BLOCK_T, D2]
            Kp_rows = tl.load(
                Kp_sub_ptr + t_idx[:, None] * D2 + tl.arange(0, D2)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D2]

            for tt in tl.static_range(0, BLOCK_T):
                valid = mask_t[tt]
                # For each token, compute logits_scalar for each column (accumulate over tt)
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]
                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar
                # Update per-column max and sum
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # lse = max + log(sum(exp(.) - max)) / ln(2)
        ln2 = 1.4426950408889634  # 1 / log(2)
        lse_h = token_max_vec + tl.log(token_sum_vec) / ln2
        # Store to lse_out[b, h]
        tl.store(lse_out_ptr + b * H + h, lse_h)


@triton.jit
def _compute_out_kernel(
    q_nope_rows_ptr,     # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,       # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    out_ptr,             # *f32, shape [B, H, D1], contiguous (we'll store fp32)
    H: tl.int32,         # number of heads (runtime)
    D1: tl.constexpr,    # 512
    D2: tl.constexpr,    # 64
    L_tokens: tl.int32,  # runtime
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(axis=0)
    for h in range(0, H):
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Compute per-column max and sum for softmax
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        num_tiles = (L_tokens + BLOCK_T - 1) // BLOCK_T
        for tile in range(0, num_tiles):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)
            mask_t = t_idx < L_tokens
            Kc_rows = tl.load(
                Kc_sub_ptr + t_idx[:, None] * D1 + tl.arange(0, D1)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D1]
            Kp_rows = tl.load(
                Kp_sub_ptr + t_idx[:, None] * D2 + tl.arange(0, D2)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D2]

            for tt in tl.static_range(0, BLOCK_T):
                valid = mask_t[tt]
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]
                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Accumulate output: out[b, h, :] += attn_t * Kc_row for each t
        for tile in range(0, num_tiles):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)
            mask_t = t_idx < L_tokens
            Kc_rows = tl.load(
                Kc_sub_ptr + t_idx[:, None] * D1 + tl.arange(0, D1)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D1]
            Kp_rows = tl.load(
                Kp_sub_ptr + t_idx[:, None] * D2 + tl.arange(0, D2)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D2]

            for tt in tl.static_range(0, BLOCK_T):
                valid = mask_t[tt]
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]
                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                logits_scalar = (dot1 + dot2) * sm_scale
                attn_t = tl.where(valid, tl.exp(logits_scalar - token_max_vec) / token_sum_vec, 0.0)
                out_vec = tl.load(out_ptr + b * H * D1 + h * D1 + tl.arange(0, D1)).to(tl.float32)
                out_vec += attn_t * Kc_row
                tl.store(out_ptr + b * H * D1 + h * D1 + tl.arange(0, D1), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable tile for tokens
        self.block_t = 128  # good default; masks handle L_tokens not divisible

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Triton requires CUDA tensors"
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]

        # Compute per-batch token ranges
        L_tokens = (kv_indptr[1:] - kv_indptr[:-1]).tolist()
        # Squeeze caches: original uses [N, 1, ...], but host ensures correct dims; we can keep as-is and slice by indices
        # Prepare q_nope_rows and q_pe_rows: [H, D1] and [H, D2]
        q_nope_rows = q_nope.to(torch.float32).reshape(H, D1).contiguous()
        q_pe_rows = q_pe.to(torch.float32).reshape(H, D2).contiguous()

        # Allocate outputs
        output = torch.empty((B, H, D1), dtype=torch.float32, device=q_nope.device)  # fp32 for accumulation
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch kernels per batch element
        for b in range(B):
            if L_tokens[b] <= 0:
                # No tokens for this batch element: output zeros, lse zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc_sub and Kp_sub for this batch element
            Kc_sub = ckv_cache[kv_indices[0:L_tokens[b]]].contiguous().to(torch.float32)  # [L_tokens, D1]
            Kp_sub = kpe_cache[kv_indices[0:L_tokens[b]]].contiguous().to(torch.float32)  # [L_tokens, D2]

            # Launch Triton kernel for lse
            _compute_lse_per_head_kernel[(1,)](
                q_nope_rows,
                q_pe_rows,
                Kc_sub,
                Kp_sub,
                lse[b].reshape(1, H),  # pass as [1, H] to match kernel's [B, H] pointer
                H,
                D1,
                D2,
                L_tokens[b],
                sm_scale,
                self.block_t,
            )

            # Launch Triton kernel for output accumulation
            _compute_out_kernel[(1,)](
                q_nope_rows,
                q_pe_rows,
                Kc_sub,
                Kp_sub,
                output[b].reshape(1, H, D1),
                H,
                D1,
                D2,
                L_tokens[b],
                sm_scale,
                self.block_t,
            )

        # Cast output to bfloat16 to match original Model's output dtype
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
