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
    kv_indices_ptr,       # *int32 [L] (not used in the kernel)
    output_ptr,           # *bf16 [B, H, Dc] (we write float32 and cast on host)
    lse_ptr,              # *float32 [B*H]
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv
    Dp: tl.constexpr,     # head_dim_kpe
    SM_SCALE: tl.constexpr,  # scaling factor for logits
):
    # One program per batch element
    b = tl.program_id(0)

    # Load the token range for this batch element
    base = tl.load(kv_indptr_ptr + b)       # int32
    end = tl.load(kv_indptr_ptr + b + 1)    # int32
    L_tokens = end - base

    # Precompute factor for converting logsumexp to base-2
    LOG2 = 1.0 / tl.log(2.0)

    # Loop over heads
    for h in range(H):
        # Prepare output vector (float32)
        out_vec = tl.zeros((Dc,), dtype=tl.float32)

        # Load q_nope and q_pe for this head (cast to float32)
        qn_ptr = q_nope_ptr + b * H * Dc + h * Dc
        qp_ptr = q_pe_ptr + b * H * Dp + h * Dp
        qn = tl.load(qn_ptr + tl.arange(0, Dc)).to(tl.float32)
        qp = tl.load(qp_ptr + tl.arange(0, Dp)).to(tl.float32)

        # Initialize running logsumexp state
        running_max = tl.full((), -float("inf"), dtype=tl.float32)
        running_sum = tl.full((), 0.0, dtype=tl.float32)

        # First pass: compute running logsumexp over tokens
        for i in range(L_tokens):
            idx = base + i
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kp_row_ptr = kpe_cache_ptr + idx * Dp

            # Load keys as float32
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc)).to(tl.float32)
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp)).to(tl.float32)

            # Compute logits contribution
            dot1 = tl.sum(qn * Kc_row, axis=0)
            dot2 = tl.sum(qp * Kp_row, axis=0)
            val = (dot1 + dot2) * SM_SCALE

            # Running logsumexp update
            new_max = tl.maximum(running_max, val)
            # sum' = exp(running_max - new_max) * running_sum + exp(val - new_max)
            running_sum = tl.exp(running_max - new_max) * running_sum + tl.exp(val - new_max)
            running_max = new_max

        # Compute final lse: log(running_sum) + running_max, then divide by ln(2)
        lse_val = tl.log(running_sum) + running_max
        lse_val = lse_val * LOG2

        # Second pass: compute attention weights and final output
        for i in range(L_tokens):
            idx = base + i
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc)).to(tl.float32)
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp)).to(tl.float32)

            dot1 = tl.sum(qn * Kc_row, axis=0)
            dot2 = tl.sum(qp * Kp_row, axis=0)
            val = (dot1 + dot2) * SM_SCALE
            attn_i = tl.exp(val - lse_val)
            out_vec += attn_i * Kc_row

        # Store output vector for head h
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
          * q_nope: [B, H, Dc], dtype bfloat16
          * q_pe: [B, H, Dp], dtype bfloat16
          * ckv_cache: [N, 1, Dc], dtype bfloat16
          * kpe_cache: [N, 1, Dp], dtype bfloat16
          * kv_indptr: [B+1], int32
          * kv_indices: [L], int32 (kept for API compatibility)
          * sm_scale: float32 scalar
        - Returns:
          * output: [B, H, Dc], dtype bfloat16
          * lse: [B, H], dtype float32
        """
        # Ensure inputs are contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Allocate outputs (compute in float32 then cast)
        output = torch.empty((B, H, Dc), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp, SM_SCALE=sm_scale,
        )

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
