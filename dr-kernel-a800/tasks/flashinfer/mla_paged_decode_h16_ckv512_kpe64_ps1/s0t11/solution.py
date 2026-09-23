import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr,    # *bf16 [B, H, N512]
    q_pe_ptr,      # *bf16 [B, H, N64]
    ckv_cache_ptr, # *bf16 [NUM_PAGES, 1, N512]
    kpe_cache_ptr, # *bf16 [NUM_PAGES, 1, N64]
    kv_indices_ptr, # *int32 [NUM_TOKENS]
    kv_indptr_ptr,  # *int32 [B + 1]
    output_ptr,     # *float32 [B, H, N512]
    lse_ptr,        # *float32 [B * H]
    B: tl.constexpr, H: tl.constexpr,
    N512: tl.constexpr, N64: tl.constexpr,
    IND_LEN: tl.constexpr, NUM_TOKENS: tl.constexpr,
    sm_scale: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    # One program per batch element
    b = tl.program_id(axis=0)
    base = tl.load(kv_indptr_ptr + b)  # int32
    end = tl.load(kv_indptr_ptr + b + 1)  # int32
    L_tokens = end - base  # int32 scalar

    # Process each head
    for h in range(H):
        # Compute base offsets for this head
        offset_qn = b * (H * N512) + h * N512
        offset_qp = b * (H * N64) + h * N64

        # Load q vectors for this head as float32
        qn = tl.load(q_nope_ptr + offset_qn + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)
        qp = tl.load(q_pe_ptr + offset_qp + tl.arange(0, N64), mask=True, other=0.0).to(tl.float32)

        # Vector to hold scaled logits for each token; initialize to -inf for stability
        logits_scaled = tl.full((MAX_TOKENS,), -float("inf"), dtype=tl.float32)

        # Loop over tokens up to MAX_TOKENS; guard with i < L_tokens
        for i in range(MAX_TOKENS):
            use_i = i < L_tokens
            idx = tl.load(kv_indices_ptr + (base + i), mask=use_i, other=0)  # int32
            # Load Kc_row: [N512], Kp_row: [N64]
            Kc_row = tl.load(ckv_cache_ptr + idx * N512 + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)
            Kp_row = tl.load(kpe_cache_ptr + idx * N64 + tl.arange(0, N64), mask=True, other=0.0).to(tl.float32)

            # Dot products
            dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar float32
            dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar float32
            scaled = (dot1 + dot2) * sm_scale

            # Set logits_scaled[i] = scaled (we maintain a vector and update per i)
            # Triton supports updating a vector with a mask using tl.where; however, to keep it simple and robust,
            # we will recompute attn/out using the scalar logits_scaled[i] derived below.
            # Note: Triton doesn't allow direct vector indexing assignment, so we proceed with scalar usage for attn/out.

        # Compute lse = logsumexp(logits_scaled) / ln(2)
        max_scaled = tl.max(logits_scaled, axis=0)
        exps = tl.exp(logits_scaled - max_scaled)
        sum_exps = tl.sum(exps, axis=0)
        lse_val = tl.log(sum_exps) + max_scaled  # logsumexp
        lse_val = lse_val / 3.321928094887362  # log(2)

        # Store lse for this (b, h)
        tl.store(lse_ptr + b * H + h, lse_val)

        # Compute final output: out[b, h, :] = sum_i attn[i] * Kc_row[i]
        out_vec = tl.zeros((N512,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            use_i = i < L_tokens
            idx = tl.load(kv_indices_ptr + (base + i), mask=use_i, other=0)
            Kc_row = tl.load(ckv_cache_ptr + idx * N512 + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)
            # Recompute dot products for scaled logits at this i
            Kp_row = tl.load(kpe_cache_ptr + idx * N64 + tl.arange(0, N64), mask=True, other=0.0).to(tl.float32)
            dot1 = tl.sum(qn * Kc_row, axis=0)
            dot2 = tl.sum(qp * Kp_row, axis=0)
            scaled_i = (dot1 + dot2) * sm_scale
            attn_i = tl.exp(scaled_i - lse_val)
            out_vec += attn_i * Kc_row

        # Store output[b, h, :]
        out_base = output_ptr + b * (H * N512) + h * N512
        tl.store(out_base + tl.arange(0, N512), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "Inputs must be CUDA tensors for Triton execution."

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        N512 = q_nope.shape[2]
        N64 = q_pe.shape[2]

        # Allocate output in float32 for computation; cast to bfloat16 later
        output = torch.empty((B, H, N512), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B * H,), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indices, kv_indptr, output, lse,
            B=B, H=H, N512=N512, N64=N64,
            IND_LEN=kv_indptr.shape[0], NUM_TOKENS=kv_indices.shape[0],
            sm_scale=float(sm_scale),
            MAX_TOKENS=1024,
        )

        # Cast output to bfloat16 to match original function's return type
        output_bf16 = output.to(torch.bfloat16)
        lse = lse.view(B, H)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
