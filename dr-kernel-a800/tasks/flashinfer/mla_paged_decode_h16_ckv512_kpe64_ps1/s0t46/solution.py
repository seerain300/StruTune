import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr,            # *bf16 [B, H, Dc]
    q_pe_ptr,              # *bf16 [B, H, Dp]
    ckv_cache_ptr,         # *bf16 [N, 1, Dc]
    kpe_cache_ptr,         # *bf16 [N, 1, Dp]
    kv_indptr_ptr,         # *int32 [B+1]
    kv_indices_ptr,        # *int32 [L]
    output_ptr,            # *float32 [B, H, Dc]
    lse_ptr,               # *float32 [B*H]
    B: tl.constexpr,       # batch_size
    H: tl.constexpr,       # num_qo_heads (16)
    Dc: tl.constexpr,      # head_dim_ckv (512)
    Dp: tl.constexpr,      # head_dim_kpe (64)
    SM_SCALE: tl.float32,  # scaling factor
):
    # One program per batch element
    b = tl.program_id(0)

    # Loop over heads
    for h in range(0, H):
        # Determine number of tokens for this batch element
        base = tl.load(kv_indptr_ptr + b)
        end = tl.load(kv_indptr_ptr + b + 1)
        L_tokens = end - base

        # Initialize for logsumexp
        max_val = -float("inf")
        sum_exp = 0.0

        # First pass: compute max over logits
        for i in range(0, 1024):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)

                # Load qn and qp for this head as vectors
                qn_ptr = q_nope_ptr + b * H * Dc + h * Dc + tl.arange(0, Dc)
                qp_ptr = q_pe_ptr + b * H * Dp + h * Dp + tl.arange(0, Dp)

                Kc_row_ptr = ckv_cache_ptr + idx * Dc + tl.arange(0, Dc)
                Kp_row_ptr = kpe_cache_ptr + idx * Dp + tl.arange(0, Dp)

                qn = tl.load(qn_ptr, mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                qp = tl.load(qp_ptr, mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)
                Kc_row = tl.load(Kc_row_ptr, mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                Kp_row = tl.load(Kp_row_ptr, mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

                # Dot products (scalars)
                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                val = (dot1 + dot2) * SM_SCALE

                # Stable update
                if max_val == -float("inf") or val > max_val:
                    sum_exp = tl.exp(val - max_val)
                    max_val = val
                else:
                    sum_exp = sum_exp + tl.exp(val - max_val)

        # Compute lse: logsumexp / log(2)
        lse_val = tl.log(sum_exp) + max_val
        lse_val = lse_val / tl.log(2.0)

        # Second pass: compute final output
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(0, 1024):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)

                qn_ptr = q_nope_ptr + b * H * Dc + h * Dc + tl.arange(0, Dc)
                qp_ptr = q_pe_ptr + b * H * Dp + h * Dp + tl.arange(0, Dp)

                Kc_row_ptr = ckv_cache_ptr + idx * Dc + tl.arange(0, Dc)
                Kp_row_ptr = kpe_cache_ptr + idx * Dp + tl.arange(0, Dp)

                qn = tl.load(qn_ptr, mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                qp = tl.load(qp_ptr, mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)
                Kc_row = tl.load(Kc_row_ptr, mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                Kp_row = tl.load(Kp_row_ptr, mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                val = (dot1 + dot2) * SM_SCALE
                attn_i = tl.exp(val - lse_val)  # softmax over tokens

                out_vec += attn_i * Kc_row

        # Store output for head h (float32); host will cast to bfloat16
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
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA."
        assert kv_indptr.is_cuda and kv_indices.is_cuda, "Indptr and indices must be on CUDA."

        B, H, Dc = q_nope.shape
        Dp = q_pe.shape[-1]
        N = ckv_cache.shape[0]
        L = kv_indices.shape[0]

        # Allocate outputs as float32 for computation; host will cast output to bfloat16
        output = torch.empty((B, H, Dc), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
            output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp, SM_SCALE=float(sm_scale),
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)

        # Return in the expected format: [output, lse]
        return [output_bf16, lse]


def run(*args):
    return ModelNew()(*args)
