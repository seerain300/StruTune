import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_lse_per_head_kernel(
    q_nope_rows_ptr,     # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,       # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    lse_out_ptr,         # *f32, shape [B, H], contiguous
    H: tl.int32,         # number of heads (runtime)
    D1: tl.constexpr,    # 512
    D2: tl.constexpr,    # 64
    L_tokens: tl.int32,  # runtime
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    # One Triton program per batch element (grid is (B,))
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for head h
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Compute per-column max across tokens and sum of exp(logits - max)
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        num_tiles = (L_tokens + BLOCK_T - 1) // BLOCK_T
        for tile in range(0, num_tiles):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
            mask_t = t_idx < L_tokens

            # Load Kc_rows and Kp_rows for this tile: [BLOCK_T, D1] and [BLOCK_T, D2]
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

            # For each token in the tile, update max and sum-exp
            for tt in tl.static_range(0, BLOCK_T):
                valid = mask_t[tt]
                # If invalid, skip
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]
                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                # Update per-column max
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)

                # Update per-column sum exp (skip invalid tokens)
                if valid:
                    token_sum_vec += tl.exp(logits_scalar - token_max_vec)

        # Compute lse per head: max + log(sumexp) / ln(2)
        lse_val = token_max_vec + tl.log(token_sum_vec) / tl.log(2.0)  # scalar
        # Store lse[b, h]
        tl.store(lse_out_ptr + b * H + h, lse_val)


@triton.jit
def _compute_out_kernel(
    q_nope_rows_ptr,     # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,       # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    out_ptr,             # *f32, shape [B, H, D1], contiguous (we'll store fp32)
    H: tl.int32,
    D1: tl.constexpr,
    D2: tl.constexpr,
    L_tokens: tl.int32,
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    # One Triton program per batch element (grid is (B,))
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for head h
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Compute per-column max across tokens and sum of exp(logits - max)
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        num_tiles = (L_tokens + BLOCK_T - 1) // BLOCK_T
        for tile in range(0, num_tiles):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
            mask_t = t_idx < L_tokens

            # Load Kc_rows and Kp_rows for this tile: [BLOCK_T, D1] and [BLOCK_T, D2]
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
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                if valid:
                    token_sum_vec += tl.exp(logits_scalar - token_max_vec)

        # Now accumulate output: out[b, h, :] += softmax_t * Kc_row for each token t
        for tile in range(0, num_tiles):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
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
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                attn = tl.exp(logits_scalar - token_max_vec) / token_sum_vec  # scalar
                if valid:
                    out_vec = tl.load(out_ptr + b * H * D1 + h * D1 + tl.arange(0, D1), mask=True, other=0.0).to(tl.float32)
                    out_vec += attn * Kc_row
                    tl.store(out_ptr + b * H * D1 + h * D1 + tl.arange(0, D1), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and checks (keep behavior similar to original)
        assert q_nope.shape[1] == 16 and q_nope.shape[2] == 512, "q_nope must be [B, 16, 512]"
        assert q_pe.shape[1] == 16 and q_pe.shape[2] == 64, "q_pe must be [B, 16, 64]"
        assert ckv_cache.shape[-1] == 512 and kpe_cache.shape[-1] == 64, "Cache dims must match"
        device = q_nope.device
        dtype = torch.float32

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]  # number of heads
        D1 = 512
        D2 = 64

        # Compute L_tokens per batch element
        # len_indptr = batch_size + 1
        B = batch_size
        len_indptr = kv_indptr.numel()
        assert len_indptr == B + 1, "kv_indptr must have shape [batch_size + 1]"

        # Prepare output tensors
        output = torch.zeros((B, H, D1), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # For each batch b, compute range
        # Note: The original code asserts len_indptr == B + 1 and uses kv_indptr[b+1] - kv_indptr[b].
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start

            if L_tokens <= 0:
                # No KV entries for this batch element; output zeros and lse zeros
                lse[b] = 0.0
                continue

            # Gather Kc_sub and Kp_sub for this batch element
            # kv_indices has shape [L_tokens]
            idx_list = kv_indices[start:end]  # 1D tensor [L_tokens]
            # Load from caches
            Kc_sub = ckv_cache[idx_list].contiguous().to(torch.float32)  # [L_tokens, D1]
            Kp_sub = kpe_cache[idx_list].contiguous().to(torch.float32)  # [L_tokens, D2]

            # Prepare q_nope_rows and q_pe_rows for this batch: [H, D1] and [H, D2]
            q_nope_b = q_nope[b].to(torch.float32).contiguous().view(H, D1)  # [H, D1]
            q_pe_b = q_pe[b].to(torch.float32).contiguous().view(H, D2)     # [H, D2]

            # Launch Triton kernels:
            # 1) compute lse per head
            BLOCK_T = 128  # tile size for tokens; works well for small L_tokens
            grid = (B,)  # one program per batch element
            _compute_lse_per_head_kernel[grid](
                q_nope_b, q_pe_b, Kc_sub, Kp_sub, lse, H, D1, D2, L_tokens, sm_scale, BLOCK_T
            )

            # 2) compute out per head
            _compute_out_kernel[grid](
                q_nope_b, q_pe_b, Kc_sub, Kp_sub, output, H, D1, D2, L_tokens, sm_scale, BLOCK_T
            )

        # Return output in bf16 (original uses bfloat16), and lse as fp32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
