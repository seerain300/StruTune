import math
import torch
import triton
import triton.language as tl


@triton.jit
def _lse_per_head_kernel(
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
    BLOCK_T: tl.constexpr,  # e.g., 1024
):
    # One Triton program per batch element
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for this head h
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Initialize per-column max and sum across tokens for logsumexp
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        # Iterate over tokens in a single static tile (masked for runtime L_tokens)
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t0 = tile * BLOCK_T
            for tt in tl.static_range(0, BLOCK_T):
                t = t0 + tt
                valid = t < L_tokens
                # Load Kc_row and Kp_row as 1D vectors (masked for OOB)
                Kc_row = tl.load(Kc_sub_ptr + t * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
                Kp_row = tl.load(Kp_sub_ptr + t * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

                # Compute scalar logits for this token
                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                # Update per-column max and sum (masked by valid)
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Store lse for this batch b, head h: lse = max + log(sum_exp) / ln(2)
        lse_val = token_max_vec + tl.log(token_sum_vec) * 1.4426950408889634  # 1/ln(2)
        tl.store(lse_out_ptr + b * H + h, lse_val)


@triton.jit
def _compute_output_kernel(
    q_nope_rows_ptr,     # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,       # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    out_ptr,             # *f32, shape [B, H, D1], contiguous
    H: tl.int32,         # number of heads (runtime)
    D1: tl.constexpr,    # 512
    D2: tl.constexpr,    # 64
    L_tokens: tl.int32,  # runtime
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,  # e.g., 1024
):
    # One Triton program per batch element
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for this head
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Initialize output vector for head h
        out_vec = tl.zeros((D1,), dtype=tl.float32)

        # Initialize per-column max and sum across tokens for softmax scaling
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

                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Second pass: compute attention for each token and accumulate output
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t0 = tile * BLOCK_T
            for tt in tl.static_range(0, BLOCK_T):
                t = t0 + tt
                valid = t < L_tokens
                Kc_row = tl.load(Kc_sub_ptr + t * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
                Kp_row = tl.load(Kp_sub_ptr + t * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                attn = tl.where(valid, tl.exp(logits_scalar - token_max_vec) / token_sum_vec, 0.0)
                out_vec += attn * Kc_row

        # Store accumulated output for this head h
        tl.store(out_ptr + b * (H * D1) + h * D1 + tl.arange(0, D1), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, D1], dtype bfloat16 (H=16, D1=512)
        q_pe: [B, H, D2], dtype bfloat16 (D2=64)
        ckv_cache: [N, 1, D1] -> squeeze to [N, D1]
        kpe_cache: [N, 1, D2] -> squeeze to [N, D2]
        kv_indptr: [B+1], int32
        kv_indices: [L_tokens], int32 (runtime per batch)
        sm_scale: float32
        Returns:
        - output: [B, H, D1], bfloat16
        - lse: [B, H], float32
        """
        assert q_nope.shape[1] == 16 and q_nope.shape[2] == 512
        assert q_pe.shape[1] == 16 and q_pe.shape[2] == 64
        assert ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == 512
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == 64

        B = q_nope.shape[0]
        H = 16
        D1 = 512
        D2 = 64

        device = q_nope.device

        # Squeeze cache tensors
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, D1]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, D2]

        # Allocate outputs (compute


def run(*args):
    return ModelNew()(*args)
