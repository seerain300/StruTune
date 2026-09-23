import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_lse_per_head_kernel(
    q_nope_rows_ptr,       # *f32, [H, D1], contiguous
    q_pe_rows_ptr,         # *f32, [H, D2], contiguous
    Kc_sub_ptr,            # *f32, [L_tokens, D1], contiguous
    Kp_sub_ptr,            # *f32, [L_tokens, D2], contiguous
    lse_out_ptr,           # *f32, [B, H], contiguous
    H: tl.int32,           # number of heads (runtime)
    D1: tl.constexpr,      # 512
    D2: tl.constexpr,      # 64
    L_tokens: tl.int32,    # runtime
    sm_scale: tl.float32,  # scaling factor
    BLOCK_T: tl.constexpr, # tile size for tokens (e.g., 128)
):
    # One Triton program per batch element b
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for this head (1D vectors with constexpr bounds)
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Compute per-column max and sum across tokens for lse
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.full((D1,), 0.0, dtype=tl.float32)

        num_tiles = (L_tokens + BLOCK_T - 1) // BLOCK_T
        for tile in range(0, num_tiles):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
            mask_t = t_idx < L_tokens

            # Load Kc rows and Kp rows for this tile: shape [BLOCK_T, D1] and [BLOCK_T, D2]
            Kc_rows = tl.load(
                Kc_sub_ptr + t_idx[:, None] * D1 + tl.arange(0, D1)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)
            Kp_rows = tl.load(
                Kp_sub_ptr + t_idx[:, None] * D2 + tl.arange(0, D2)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)

            # Iterate over each token in the tile (statically)
            for tt in tl.static_range(0, BLOCK_T):
                valid = mask_t[tt]
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]

                # Compute logits scalar for this token
                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                # Update per-column token_max
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)

                # Update per-column token_sum only for valid tokens
                if valid:
                    token_sum_vec += tl.exp(logits_scalar - token_max_vec)

        # lse = max + log(sum) / ln(2)
        lse_val = token_max_vec + tl.log(token_sum_vec) / 0.6931471805599453  # 1 / ln(2)
        # Store lse[b, h]
        tl.store(lse_out_ptr + b * H + h, lse_val)


@triton.jit
def _compute_output_kernel(
    q_nope_rows_ptr,       # *f32, [H, D1], contiguous
    q_pe_rows_ptr,         # *f32, [H, D2], contiguous
    Kc_sub_ptr,            # *f32, [L_tokens, D1], contiguous
    Kp_sub_ptr,            # *f32, [L_tokens, D2], contiguous
    out_ptr,               # *f32, [B, H, D1], contiguous
    H: tl.int32,
    D1: tl.constexpr,
    D2: tl.constexpr,
    L_tokens: tl.int32,
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    # One Triton program per batch element b
    b = tl.program_id(axis=0)

    for h in range(0, H):
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Compute token_max and token_sum as in lse kernel
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.full((D1,), 0.0, dtype=tl.float32)

        num_tiles = (L_tokens + BLOCK_T - 1) // BLOCK_T
        for tile in range(0, num_tiles):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)
            mask_t = t_idx < L_tokens

            Kc_rows = tl.load(
                Kc_sub_ptr + t_idx[:, None] * D1 + tl.arange(0, D1)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)
            Kp_rows = tl.load(
                Kp_sub_ptr + t_idx[:, None] * D2 + tl.arange(0, D2)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)

            for tt in tl.static_range(0, BLOCK_T):
                valid = mask_t[tt]
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                logits_scalar = (dot1 + dot2) * sm_scale

                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                if valid:
                    token_sum_vec += tl.exp(logits_scalar - token_max_vec)

        # Accumulate output: out[b, h, :] += attn[t] * Kc_row for each valid t
        out_vec = tl.zeros((D1,), dtype=tl.float32)
        for tile in range(0, num_tiles):
            t_idx = tile * BLOCK_T + tl.arange(0, BLOCK_T)
            mask_t = t_idx < L_tokens

            Kc_rows = tl.load(
                Kc_sub_ptr + t_idx[:, None] * D1 + tl.arange(0, D1)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)
            Kp_rows = tl.load(
                Kp_sub_ptr + t_idx[:, None] * D2 + tl.arange(0, D2)[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)

            for tt in tl.static_range(0, BLOCK_T):
                valid = mask_t[tt]
                Kc_row = Kc_rows[tt, :]  # [D1]
                Kp_row = Kp_rows[tt, :]  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                logits_scalar = (dot1 + dot2) * sm_scale

                # attn = exp(logits - max) / sum_exp
                norm = tl.exp(logits_scalar - token_max_vec)
                attn = norm / token_sum_vec

                if valid:
                    out_vec += attn * Kc_row

        # Store out[b, h, :]
        tl.store(out_ptr + b * (H * D1) + h * D1 + tl.arange(0, D1), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_t=128):
        super().__init__()
        self.sm_scale = float(sm_scale)
        self.block_t = int(block_t)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices):
        # Ensure inputs are CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]

        # Prepare slices for each batch element on host: convert to fp32 and contiguous
        q_nope_rows = []
        q_pe_rows = []
        for b in range(B):
            qnb = q_nope[b].to(torch.float32).contiguous()  # [H, D1]
            qpb = q_pe[b].to(torch.float32).contiguous()   # [H, D2]
            q_nope_rows.append(qnb)
            q_pe_rows.append(qpb)

        # Compute token ranges and slices for Kc and Kp per batch
        Kc_subs = []
        Kp_subs = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                Kc_subs.append(torch.empty((0, D1), dtype=torch.float32, device=ckv_cache.device))
                Kp_subs.append(torch.empty((0, D2), dtype=torch.float32, device=kpe_cache.device))
                continue
            indices = kv_indices[start:end]  # [L_tokens]
            Kc_sub = ckv_cache[indices].to(torch.float32).contiguous()  # [L_tokens, D1]
            Kp_sub = kpe_cache[indices].to(torch.float32).contiguous()  # [L_tokens, D2]
            Kc_subs.append(Kc_sub)
            Kp_subs.append(Kp_sub)

        # Allocate outputs: fp32 for kernels; cast to bf16 at the end
        out = torch.empty((B, H, D1), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch Trit


def run(*args):
    return ModelNew()(*args)
