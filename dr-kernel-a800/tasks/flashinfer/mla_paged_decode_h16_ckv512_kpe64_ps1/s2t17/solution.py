import math
import torch
import triton
import triton.language as tl


@triton.jit
def _lse_per_head_kernel(
    q_nope_rows_ptr,     # *f32, shape [B*H, D1]
    q_pe_rows_ptr,       # *f32, shape [B*H, D2]
    Kc_all_ptr,          # *f32, shape [N, D1]
    Kp_all_ptr,          # *f32, shape [N, D2]
    lse_out_ptr,         # *f32, shape [B, H]
    B: tl.int32,
    H: tl.int32,
    D1: tl.constexpr,    # 512
    D2: tl.constexpr,    # 64
    L_tokens: tl.int32,  # runtime per batch element
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,  # e.g., 1024
):
    # One program per batch element
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for head h (we pass q_nope and q_pe as [B*H, D] and index by b*H + h)
        qn = tl.load(q_nope_rows_ptr + (b * H + h) * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + (b * H + h) * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Per-column max and sum across tokens for logsumexp
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        # Tile over tokens
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
            mask_t = t_idx < L_tokens

            # Load Kc_rows and Kp_rows as [BLOCK_T, D1] and [BLOCK_T, D2]
            Kc_rows = tl.load(
                Kc_all_ptr + t_idx[:, None] * D1 + tl.arange(0, D1)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D1]
            Kp_rows = tl.load(
                Kp_all_ptr + t_idx[:, None] * D2 + tl.arange(0, D2)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D2]

            # Loop over each token in the tile
            for tt in tl.static_range(0, BLOCK_T):
                t = t_idx[tt]
                valid = mask_t[tt]
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                # Update per-column max and sum (masked)
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Store lse[b, h] = token_max_vec + log(token_sum_vec) / ln(2)
        ln2 = 0.6931471805599453  # math.log(2)
        lse_val = token_max_vec + tl.log(token_sum_vec) / ln2
        tl.store(lse_out_ptr + b * H + h, lse_val)


@triton.jit
def _compute_output_kernel(
    q_nope_rows_ptr,     # *f32, shape [B*H, D1]
    q_pe_rows_ptr,       # *f32, shape [B*H, D2]
    Kc_all_ptr,          # *f32, shape [N, D1]
    Kp_all_ptr,          # *f32, shape [N, D2]
    out_ptr,             # *f32, shape [B*H, D1]
    lse_out_ptr,         # *f32, shape [B, H] (not used, but kept for signature compatibility)
    B: tl.int32,
    H: tl.int32,
    D1: tl.constexpr,
    D2: tl.constexpr,
    L_tokens: tl.int32,
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    # One program per batch element
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for head h
        qn = tl.load(q_nope_rows_ptr + (b * H + h) * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + (b * H + h) * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Per-column max and sum across tokens for softmax
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        # Accumulator for output
        out_vec = tl.zeros((D1,), dtype=tl.float32)

        # Tile over tokens
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)
            mask_t = t_idx < L_tokens

            Kc_rows = tl.load(
                Kc_all_ptr + t_idx[:, None] * D1 + tl.arange(0, D1)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D1]
            Kp_rows = tl.load(
                Kp_all_ptr + t_idx[:, None] * D2 + tl.arange(0, D2)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D2]

            for tt in tl.static_range(0, BLOCK_T):
                t = t_idx[tt]
                valid = mask_t[tt]
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                # Update per-column max and sum (masked)
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        ln2 = 0.6931471805599453
        sum_exp = token_sum_vec  # already scaled by exp(logits - max)
        inv_sum = 1.0 / sum_exp

        # Now go back and accumulate output with softmax
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)
            mask_t = t_idx < L_tokens

            Kc_rows = tl.load(
                Kc_all_ptr + t_idx[:, None] * D1 + tl.arange(0, D1)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D1]
            Kp_rows = tl.load(
                Kp_all_ptr + t_idx[:, None] * D2 + tl.arange(0, D2)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_T, D2]

            for tt in tl.static_range(0, BLOCK_T):
                t = t_idx[tt]
                valid = mask_t[tt]
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                attn = tl.where(valid, tl.exp(logits_scalar - token_max_vec) * inv_sum, 0.0)
                out_vec += attn * Kc_row

        # Store accumulated output for head h into out[b*H + h, :]
        tl.store(out_ptr + (b * H + h) * D1 + tl.arange(0, D1), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Inputs:
          q_nope: [B, H, D1], dtype bfloat16 (H=16, D1=512)
          q_pe:   [B, H, D2], dtype bfloat16 (D2=64)
          ckv_cache: [N, 1, D1]
          kpe_cache: [N, 1, D2]
          kv_indptr: [B+1], int32
          kv_indices: [L_tokens] (runtime per batch)
          sm_scale: float32 scalar
        Returns:
          output: [B, H, D1], bfloat16
          lse:    [B, H], float32
        """
        # Ensure tensors are on same device
        device = q_nope.device
        assert q_nope.shape[1] == 16 and q_nope.shape[2] == 512
        assert q_pe.shape[1] == 16 and q_pe.shape[2] == 64
        assert ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == 512
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == 64

        # Squeeze cache tensors to [N, D] and cast to float32 for Triton
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [N, D1]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [N, D2]

        B = q_nope.shape[0]
        H = 16
        D1 = 512
        D2 = 64
        BLOCK_T = 1024  # constexpr tile over tokens

        # Allocate outputs
        lse_out = torch.empty((B, H), dtype=torch.float32, device=device)
        out_buf = torch.empty((B * H, D1), dtype=torch.float32, device=device)

        # Prepare pointers: q_nope and q_pe as [B*H, D]
        q_nope_rows = q_nope.to(torch.float32).contiguous().view(B * H, D1)
        q_pe_rows = q_pe.to(torch.float32).contiguous().view(B * H, D2)

        # Compute L_tokens per batch element
        L_tokens = int(kv_indptr[1].item() - kv_indptr[0].item())  # assuming grid over b=0; Triton kernel loops per b inside

        # Launch kernels: one program per batch element
        _lse_per_head_kernel[(B,)](
            q_nope_rows, q_pe_rows, Kc_all, Kp_all, lse_out,
            B, H, D1, D2, L_tokens,
            sm_scale, BLOCK_T
        )

        _compute_output_kernel[(B,)](
            q_nope_rows, q_pe_rows, Kc_all, Kp_all, out_buf,
            lse_out,  # not used in kernel
            B, H, D1, D2, L_tokens,
            sm_scale, BLOCK_T
        )

        # Reshape output to [B, H, D1] and cast to bfloat16 to match original
        output = out_buf.view(B, H, D1).to(torch.bfloat16)
        return output, lse_out


def run(*args):
    return ModelNew()(*args)
