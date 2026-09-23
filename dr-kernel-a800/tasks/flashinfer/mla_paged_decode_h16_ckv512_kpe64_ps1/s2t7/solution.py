import math
import torch

import triton
import triton.language as tl


@triton.jit
def _lse_per_b_kernel(
    q_nope_rows_ptr,     # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,       # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    lse_out_ptr,         # *f32, shape [B, H], contiguous
    H: tl.int32,         # number of heads (runtime)
    D1: tl.constexpr,    # head_dim_ckv (512)
    D2: tl.constexpr,    # head_dim_kpe (64)
    L_tokens: tl.int32,  # runtime (per b)
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    # One Triton program per batch element b (grid is (B,))
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for this head as constexpr vectors
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Initialize per-column max and sum across tokens
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        # Loop over tokens in tiles of size BLOCK_T (static unrolled)
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t0 = tile * BLOCK_T
            for tt in tl.static_range(0, BLOCK_T):
                t = t0 + tt
                valid = t < L_tokens
                # Load Kc_row and Kp_row as 1D vectors
                Kc_row = tl.load(Kc_sub_ptr + t * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
                Kp_row = tl.load(Kp_sub_ptr + t * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

                # Compute scalar logits for this token
                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                # Update per-column max and sum (masked by valid)
                token_max_vec = tl.maximum(token_max_vec, logits_scalar)
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Compute lse: lse = max + log(sum exp(...)) / ln(2)
        lse_val = token_max_vec + tl.log(token_sum_vec) / 1.4426950408889634  # 1 / ln(2)
        tl.store(lse_out_ptr + b * H + h, lse_val)


@triton.jit
def _compute_output_per_b_kernel(
    q_nope_rows_ptr,     # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,       # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    out_ptr,             # *f32, shape [H, D1], contiguous (row for this b,h)
    H: tl.int32,
    D1: tl.constexpr,
    D2: tl.constexpr,
    L_tokens: tl.int32,
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    # One Triton program per batch element b (grid is (B,))
    b = tl.program_id(axis=0)

    for h in range(0, H):
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

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

        # Second pass: accumulate output for each token
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
                attn = tl.where(valid, tl.exp(logits_scalar - token_max_vec) / token_sum_vec, 0.0)
                # out[b, h, :] += attn * Kc_row
                out_row = tl.load(out_ptr + h * D1 + tl.arange(0, D1), mask=True, other=0.0).to(tl.float32)
                out_row += attn * Kc_row
                tl.store(out_ptr + h * D1 + tl.arange(0, D1), out_row)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Ensure inputs are contiguous
        q_nope = q_nope.to(device=device, dtype=torch.float32).contiguous()
        q_pe = q_pe.to(device=device, dtype=torch.float32).contiguous()
        ckv_cache = ckv_cache.to(device=device, dtype=torch.float32).contiguous()
        kpe_cache = kpe_cache.to(device=device, dtype=torch.float32).contiguous()

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]

        # Prepare per-batch slices for q_nope and q_pe rows: [H, D1] and [H, D2]
        q_nope_rows = q_nope.view(B, H, D1)
        q_pe_rows = q_pe.view(B, H, D2)

        # Output buffer (per b, per h, per D1) in fp32 for kernels
        out = torch.empty((B, H, D1), dtype=torch.float32, device=device)
        # lse buffer
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Choose BLOCK_T tile size for tokens
        BLOCK_T = 128

        # Launch kernels per batch element b: grid = (B,)
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                continue
            # Slice Kc and Kp for this batch element
            Kc_sub = ckv_cache[kv_indices[start:end], :].contiguous()  # [L_tokens, D1]
            Kp_sub = kpe_cache[kv_indices[start:end], :].contiguous()  # [L_tokens, D2]

            # Compute lse for this b and each head
            _lse_per_b_kernel[(1,)](
                q_nope_rows[b], q_pe_rows[b], Kc_sub, Kp_sub, lse[b],
                H=H, D1=D1, D2=D2, L_tokens=L_tokens, sm_scale=sm_scale, BLOCK_T=BLOCK_T,
            )

            # Compute output for this b and each head
            _compute_output_per_b_kernel[(1,)](
                q_nope_rows[b], q_pe_rows[b], Kc_sub, Kp_sub, out[b],
                H=H, D1=D1, D2=D2, L_tokens=L_tokens, sm_scale=sm_scale, BLOCK_T=BLOCK_T,
            )

        # Cast output to bfloat16 to match original behavior
        output = out.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
