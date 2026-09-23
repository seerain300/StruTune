import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel_bh(
    q_nope_ptr,           # *bf16 [B, H, Dc]
    q_pe_ptr,             # *bf16 [B, H, Dp]
    ckv_cache_ptr,        # *bf16 [N, 1, Dc]
    kpe_cache_ptr,        # *bf16 [N, 1, Dp]
    kv_indptr_ptr,        # *int32 [B+1]
    kv_indices_ptr,       # *int32 [L]
    output_ptr,           # *bf16 [B, H, Dc]
    lse_ptr,              # *float32 [B*H]
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv (512)
    Dp: tl.constexpr,     # head_dim_kpe (64)
    SM_SCALE: tl.constexpr,  # scaling factor for logits
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)       # int32
    end = tl.load(kv_indptr_ptr + b + 1)    # int32
    L_tokens = end - base

    LOG2 = 1.0 / tl.log(2.0)

    for h in range(H):
        # Initialize output vector (float32) and running lse
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        lse_val = tl.full((), -float("inf"), dtype=tl.float32)

        if L_tokens <= 0:
            out_offset = b * H * Dc + h * Dc
            tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec)
            lse_offset = b * H + h
            tl.store(lse_ptr + lse_offset, -float("inf"))
            continue

        # Load q_nope and q_pe for this head (cast to float32)
        qn_ptr = q_nope_ptr + b * H * Dc + h * Dc
        qp_ptr = q_pe_ptr + b * H * Dp + h * Dp
        qn = tl.load(qn_ptr + tl.arange(0, Dc)).to(tl.float32)  # [Dc]
        qp = tl.load(qp_ptr + tl.arange(0, Dp)).to(tl.float32)  # [Dp]

        # Running max-sum logsumexp update across tokens
        for i in range(L_tokens):
            idx = base + i
            # Load corresponding rows from caches
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc)).to(tl.float32)  # [Dc]
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp)).to(tl.float32)  # [Dp]

            # Dot products
            dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
            dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
            val = (dot1 + dot2) * SM_SCALE

            # Stable update of running max and sum
            new_max = tl.maximum(lse_val, val)
            sum_exp = tl.where(lse_val > val,
                               sum_exp * tl.exp(lse_val - new_max) + tl.exp(val - new_max),
                               sum_exp + tl.exp(val - lse_val))
            lse_val = new_max

        # lse_val now equals log(sum_exp) + max over tokens; scale to base-2
        lse_val = tl.log(sum_exp) + lse_val
        lse_val = lse_val * LOG2  # base-2 logsumexp

        # Now compute attention and final output vector
        for i in range(L_tokens):
            idx = base + i
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc)).to(tl.float32)  # [Dc]
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp)).to(tl.float32)  # [Dp]

            dot1 = tl.sum(qn * Kc_row, axis=0)
            dot2 = tl.sum(qp * Kp_row, axis=0)
            val = (dot1 + dot2) * SM_SCALE
            attn_i = tl.exp(val - lse_val)  # attention weight for this token

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
        Triton-only forward that mirrors the original behavior:
        - q_nope: [B, H, Dc] bfloat16
        - q_pe:   [B, H, Dp] bfloat16
        - ckv_cache: [N, 1, Dc] bfloat16
        - kpe_cache: [N, 1, Dp] bfloat16
        - kv_indptr: [B+1] int32
        - kv_indices: [L] int32
        - sm_scale: float scalar
        Returns:
        - output: [B, H, Dc] bfloat16
        - lse: [B, H] float32
        """
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Allocate outputs
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel_bh[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
            output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp,
            SM_SCALE=float(sm_scale),
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
