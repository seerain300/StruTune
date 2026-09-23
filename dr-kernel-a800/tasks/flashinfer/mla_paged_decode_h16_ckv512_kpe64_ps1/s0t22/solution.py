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
    output_ptr,           # *bf16 [B, H, Dc]
    lse_ptr,              # *float32 [B*H]
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv (512)
    Dp: tl.constexpr,     # head_dim_kpe (64)
    MAX_TOKENS: tl.constexpr,  # upper bound for tokens in segment (>= L_tokens)
):
    b = tl.program_id(0)  # one program per batch element
    base = tl.load(kv_indptr_ptr + b)         # int32
    end = tl.load(kv_indptr_ptr + b + 1)      # int32
    L_tokens = end - base                      # number of tokens for this batch element

    # Loop over heads
    for h in range(H):
        # Load queries for this head and convert to float32
        qn_ptr = q_nope_ptr + b * H * Dc + h * Dc
        qn = tl.load(qn_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)

        qp_ptr = q_pe_ptr + b * H * Dp + h * Dp
        qp = tl.load(qp_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        # Initialize logits_scaled for logsumexp
        # We'll compute max and sum_exp via scalar loops, not using a vector with -inf for stability.
        # Use scalar variables for max and sum_exp
        max_val = -float("inf")
        sum_exp = 0.0

        # First pass: compute max over valid tokens
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32 token index
                Kc_row_ptr = ckv_cache_ptr + idx * Dc
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                Kp_row_ptr = kpe_cache_ptr + idx * Dp
                Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                val = (dot1 + dot2) * sm_scale
                # update max
                if val > max_val:
                    max_val = val

        # Second pass: compute sum of exp(val - max_val) over valid tokens
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
                sum_exp += tl.exp(val - max_val)

        # Compute lse = log(sum_exp) + max_val, scaled by 1/ln(2)
        lse_val = tl.log(sum_exp) + max_val
        lse_val = lse_val / tl.log(2.0)

        # Compute final output: out[b, h, :] = sum_i attn_i * Kc_row
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

        # Store output for head h in bfloat16
        out_offset = b * H * Dc + h * Dc
        out_vec_bf16 = out_vec.to(tl.bfloat16)
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec_bf16)

        # Store lse per (b, h) in float32
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
        # Ensure CUDA tensors for Triton
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        N = ckv_cache.shape[0]
        L = kv_indices.shape[0]

        # Allocate outputs
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp, MAX_TOKENS=1024,
            sm_scale=sm_scale,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
