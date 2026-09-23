import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr,           # *bf16 [B, H, Dc]
    q_pe_ptr,             # *bf16 [B, H, Dp]
    ckv_cache_ptr,        # *bf16 [N, 1, Dc]
    kpe_cache_ptr,        # *bf16 [N, 1, Dp]
    kv_indptr_ptr,        # *int32 [B+1]
    kv_indices_ptr,       # *int32 [L]
    output_ptr,           # *fp32 [B, H, Dc] (host will cast to bf16)
    lse_ptr,              # *fp32 [B*H]
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv (512)
    Dp: tl.constexpr,     # head_dim_kpe (64)
    sm_scale,             # fp32 scalar
    MAX_TOKENS: tl.constexpr,  # upper bound for tokens (>= L_tokens)
):
    b = tl.program_id(0)  # one program per batch element

    # Compute token range for this batch element
    base = tl.load(kv_indptr_ptr + b)          # int32
    end = tl.load(kv_indptr_ptr + b + 1)       # int32
    L_tokens = end - base                       # int32

    # Loop over heads
    for h in range(H):
        # Pointers to qn and qp for this head
        qn_ptr = q_nope_ptr + b * H * Dc + h * Dc
        qp_ptr = q_pe_ptr + b * H * Dp + h * Dp

        # Load qn and qp as float32 vectors
        qn = tl.load(qn_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        qp = tl.load(qp_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        # Compute logsumexp of logits_scaled per token
        max_val = -float('inf')
        sum_exp = 0.0

        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32
                Kc_row_ptr = ckv_cache_ptr + idx * Dc
                Kp_row_ptr = kpe_cache_ptr + idx * Dp

                # Load Kc_row and Kp_row as float32 vectors
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

                # Dot products
                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                val = (dot1 + dot2) * sm_scale

                # Update max and sum_exp (stable logsumexp)
                if val > max_val:
                    max_val = val
                sum_exp += tl.exp(val - max_val)

        # Compute lse = log(sum_exp) + max_val, then scale by 1/ln(2)
        lse_val = tl.log(sum_exp) + max_val
        lse_val = lse_val / tl.log(2.0)

        # Compute final output vector: out[b, h, :] = sum_i attn_i * Kc_all[i, :]
        out_vec = tl.zeros((Dc,), dtype=tl.float32)

        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)
                Kc_row_ptr = ckv_cache_ptr + idx * Dc

                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                Kp_row_ptr = kpe_cache_ptr + idx * Dp
                Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                val = (dot1 + dot2) * sm_scale
                attn_i = tl.exp(val - lse_val)

                out_vec += attn_i * Kc_row

        # Store output for head h
        out_offset = b * H * Dc + h * Dc
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec)

        # Store lse per (b, h)
        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
        - Inputs:
          * q_nope: [B, H, Dc], dtype bfloat16 (B=1, H=16, Dc=512)
          * q_pe: [B, H, Dp], dtype bfloat16 (Dp=64)
          * ckv_cache: [N, 1, Dc], dtype bfloat16
          * kpe_cache: [N, 1, Dp], dtype bfloat16
          * kv_indptr: [B+1], int32
          * kv_indices: [L], int32
          * sm_scale: float32 scalar
        - Returns:
          * output: [B, H, Dc], dtype bfloat16
          * lse: [B, H], dtype float32
        """
        assert q


def run(*args):
    return ModelNew()(*args)
