import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel_bh(
    q_nope_ptr,   # *bf16 [B, H, Dc]
    q_pe_ptr,     # *bf16 [B, H, Dp]
    ckv_cache_ptr,# *bf16 [N, 1, Dc]
    kpe_cache_ptr,# *bf16 [N, 1, Dp]
    kv_indptr_ptr,  # *int32 [B+1]
    kv_indices_ptr, # *int32 [L]
    output_ptr,     # *bf16 [B, H, Dc] (we will store float32 from kernel and cast outside)
    lse_ptr,        # *float32 [B*H]
    B: tl.constexpr,  # batch_size
    H: tl.constexpr,  # num_qo_heads
    Dc: tl.constexpr, # head_dim_ckv
    Dp: tl.constexpr, # head_dim_kpe
    SM_SCALE: tl.constexpr,  # scaling factor
):
    # One program per batch element
    b = tl.program_id(0)

    # Load the token range for this batch element
    base = tl.load(kv_indptr_ptr + b)       # int32
    end = tl.load(kv_indptr_ptr + b + 1)    # int32
    L_tokens = end - base

    # Loop over heads
    h = 0
    while h < H:
        # Handle empty token segment
        if L_tokens <= 0:
            out_offset = b * H * Dc + h * Dc
            out_vec = tl.zeros((Dc,), dtype=tl.float32)
            # Store output vector (float32), we'll cast to bf16 in host
            # Scalar stores: write each element
            for d in range(Dc):
                tl.store(output_ptr + out_offset + d, out_vec[d])
            # Store lse for this (b, h) as -inf
            lse_offset = b * H + h
            tl.store(lse_ptr + lse_offset, -float("inf"))
            h += 1
            continue

        # Prepare output vector (float32) and running lse components
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        max_val = -float("inf")
        sum_exp = 0.0

        # Pass 1: compute max of scaled logits across tokens
        i = 0
        while i < L_tokens:
            idx = base + i

            # Load q_nope[b, h] and q_pe[b, h]
            qn_ptr = q_nope_ptr + b * H * Dc + h * Dc
            qp_ptr = q_pe_ptr + b * H * Dp + h * Dp
            qn = tl.load(qn_ptr + tl.arange(0, Dc)).to(tl.float32)
            qp = tl.load(qp_ptr + tl.arange(0, Dp)).to(tl.float32)

            # Load Kc_row and Kp_row for this token
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc)).to(tl.float32)
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp)).to(tl.float32)

            # Compute dot products
            dot1 = 0.0
            j = 0
            while j < Dc:
                dot1 += qn[j] * Kc_row[j]
                j += 1

            dot2 = 0.0
            k = 0
            while k < Dp:
                dot2 += qp[k] * Kp_row[k]
                k += 1

            val = (dot1 + dot2) * SM_SCALE
            max_val = tl.maximum(max_val, val)
            i += 1

        # Pass 2: compute sum_exp = sum exp(val - max_val) across tokens
        i = 0
        while i < L_tokens:
            idx = base + i

            qn_ptr = q_nope_ptr + b * H * Dc + h * Dc
            qp_ptr = q_pe_ptr + b * H * Dp + h * Dp
            qn = tl.load(qn_ptr + tl.arange(0, Dc)).to(tl.float32)
            qp = tl.load(qp_ptr + tl.arange(0, Dp)).to(tl.float32)

            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc)).to(tl.float32)
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp)).to(tl.float32)

            dot1 = 0.0
            j = 0
            while j < Dc:
                dot1 += qn[j] * Kc_row[j]
                j += 1

            dot2 = 0.0
            k = 0
            while k < Dp:
                dot2 += qp[k] * Kp_row[k]
                k += 1

            val = (dot1 + dot2) * SM_SCALE
            sum_exp += tl.exp(val - max_val)
            i += 1

        # Compute lse in base-2: lse = log(sum_exp) + max_val; divide by ln(2)
        lse_val = tl.log(sum_exp) + max_val
        lse_val = lse_val / tl.log(2.0)  # 1/ln(2)

        # Pass 3: compute attention and accumulate output
        i = 0
        while i < L_tokens:
            idx = base + i

            qn_ptr = q_nope_ptr + b * H * Dc + h * Dc
            qp_ptr = q_pe_ptr + b * H * Dp + h * Dp
            qn = tl.load(qn_ptr + tl.arange(0, Dc)).to(tl.float32)
            qp = tl.load(qp_ptr + tl.arange(0, Dp)).to(tl.float32)

            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc)).to(tl.float32)
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp)).to(tl.float32)

            dot1 = 0.0
            j = 0
            while j < Dc:
                dot1 += qn[j] * Kc_row[j]
                j += 1

            dot2 = 0.0
            k = 0
            while k < Dp:
                dot2 += qp[k] * Kp_row[k]
                k += 1

            val = (dot1 + dot2) * SM_SCALE
            attn_i = tl.exp(val - lse_val)

            # out_vec += attn_i * Kc_row
            m = 0
            while m < Dc:
                out_vec[m] += attn_i * Kc_row[m]
                m += 1

            i += 1

        # Store output vector (float32), cast to bf16 outside
        out_offset = b * H * Dc + h * Dc
        for d in range(Dc):
            tl.store(output_ptr + out_offset + d, out_vec[d])

        # Store lse for this (b, h)
        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse_val)

        h += 1

# Entry point required by evaluation
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
        Inputs:
          * q_nope: [B, H, Dc], dtype bfloat16
          * q_pe: [B, H, Dp], dtype bfloat16
          * ckv_cache: [N, 1, Dc], dtype bfloat16
          * kpe_cache: [N, 1, Dp], dtype bfloat16
          * kv_indptr: [B+1], int32
          * kv_indices: [L], int32
          * sm_scale: float32 scalar
        Returns:
          * output: [B, H, Dc], dtype bfloat16
          * lse: [B, H], dtype float32
        """
        B, H, Dc = q_nope.shape
        assert q_pe.shape == (B, H, Dp)
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1

        device = q_nope.device

        # Allocate float32 outputs for kernel, cast to bfloat16 after
        output_f32 = torch.empty((B, H, Dc), dtype=torch.float32, device=device)
        lse_f32 = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel_bh[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
            output_f32, lse_f32,
            B=B, H=H, Dc=Dc, Dp=Dp,
            SM_SCALE=float(sm_scale),
        )

        # Cast output to bfloat16 to match original signature
        output = output_f32.to(torch.bfloat16)
        lse = lse_f32
        return output, lse


def run(*args):
    return ModelNew()(*args)
