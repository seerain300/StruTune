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
    Dc: tl.constexpr,     # head_dim_ckv
    Dp: tl.constexpr,     # head_dim_kpe
    SM_SCALE: tl.constexpr,  # scaling factor
    MAX_TOKENS: tl.constexpr,  # upper bound for tokens per batch element
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)        # int32
    end = tl.load(kv_indptr_ptr + b + 1)     # int32
    L_tokens = end - base                     # int32

    # Loop over heads
    for h in range(H):
        # Prepare vectors
        # Read q_nope[b, h, :] and q_pe[b, h, :]
        qn_ptr = q_nope_ptr + b * H * Dc + h * Dc
        qp_ptr = q_pe_ptr + b * H * Dp + h * Dp

        # Triton cannot load bf16 directly robustly; we must cast to f32 inside kernel from loaded bf16
        # But since we pass q_nope/q_pe as float32 to the kernel (in ModelNew), loads are already f32.
        qn = tl.load(qn_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)  # float32
        qp = tl.load(qp_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)  # float32

        # Initialize running max and arrays for scaled logits
        acc = -float('inf')  # float32 scalar for max
        # We'll store per-token scaled logits in a small vector to compute sum_exp later
        scaled_logits = tl.full((MAX_TOKENS,), -float('inf'), dtype=tl.float32)

        # Compute scaled logits for each token index (up to MAX_TOKENS) with mask
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32 token index
                Kc_row_ptr = ckv_cache_ptr + idx * Dc
                Kp_row_ptr = kpe_cache_ptr + idx * Dp

                # Load Kc and Kp rows as float32
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)
                Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)

                # Dot products
                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)

                val = (dot1 + dot2) * SM_SCALE
                # Update running max
                acc = tl.maximum(acc, val)
                # Store scaled logit for position i
                scaled_logits = tl.where(tl.arange(0, MAX_TOKENS) == i, val, scaled_logits)

        # Stable sum of exp over valid tokens using running max
        sum_exp = tl.zeros((), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                sum_exp += tl.exp(scaled_logits[i] - acc)

        lse_val = tl.log(sum_exp) + acc
        # Scale by 1/ln(2)
        lse_val = lse_val / tl.log(2.0)

        # Compute final output vector: out[b, h, :] = sum_i attn_i * Kc_rows[i]
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            attn_i = tl.exp(scaled_logits[i] - lse_val)
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)
                Kc_row_ptr = ckv_cache_ptr + idx * Dc
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)
                out_vec += attn_i * Kc_row

        # Store output for head h (float32 inside kernel; cast to bf16 on host)
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
          * kv_indices: [L], int32
          * sm_scale: float32 scalar
        - Returns:
          * output: [B, H, Dc], dtype bfloat16
          * lse: [B, H], dtype float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA device for Triton."
        assert kv_indptr.is_cuda and kv_indices.is_cuda, "kv_indptr and kv_indices must be on CUDA device."

        B, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        N = ckv_cache.shape[0]
        L = kv_indices.shape[0]

        # Ensure contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Convert inputs to float32 for Triton math (kernel will load bf16 and cast internally if needed)
        # However, Triton loads the dtype of the pointer; we pass pointers as-is. We create float32 copies for q/ck/vp to ensure correct math.
        # In practice, load as-is and cast to f32 inside kernel using .to(tl.float32). To avoid any confusion, we keep original dtypes and cast inside kernel.
        # But Triton cannot cast here; we need to ensure q_nope/q_pe are float32 for kernel. We'll convert them to float32 copies.
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        ckv_cache_f32 = ckv_cache.to(torch.float32)
        kpe_cache_f32 = kpe_cache.to(torch.float32)

        # Allocate outputs (float32 for compute, cast later to bfloat16)
        output_f32 = torch.empty((B, H, Dc), dtype=torch.float32, device=q_nope.device)
        lse_f32 = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope_f32, q_pe_f32, ckv_cache_f32, kpe_cache_f32, kv_indptr, kv_indices, output_f32, lse_f32,
            B=B, H=H, Dc=Dc, Dp=Dp, SM_SCALE=sm_scale, MAX_TOKENS=1024,
            num_warps=4, num_stages=2
        )

        # Return output in bfloat16 as in original, lse in float32
        output_bf16 = output_f32.to(torch.bfloat16)
        return output_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
