import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope, q_pe, ckv_cache, kpe_cache, kv_indices, kv_indptr,
    output, lse,
    batch_size, num_qo_heads, head_dim_ckv, head_dim_kpe,
    len_indptr, num_kv_indices,
    sm_scale,
    NUM_QO_HEADS: tl.constexpr, HEAD_DIM_CKV: tl.constexpr, HEAD_DIM_KPE: tl.constexpr,
    MAX_TOKENS: tl.constexpr
):
    # One program per batch element
    b = tl.program_id(axis=0)

    # Ensure we have exactly len_indptr == batch_size + 1 as per original assertions
    # Triton doesn't support dynamic asserts; we rely on caller to pass consistent shapes.
    # Read token range for this batch element
    base = tl.load(kv_indptr + b)
    end = tl.load(kv_indptr + b + 1)
    L_tokens = end - base  # number of tokens for this batch element

    # For each head h
    for h in tl.static_range(NUM_QO_HEADS):
        # Compute base offsets
        offset_qn = b * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
        offset_qp = b * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe

        # Load query vectors qn and qp as float32
        qn = tl.load(q_nope + offset_qn + tl.arange(0, head_dim_ckv), mask=True, other=0.0).to(tl.float32)
        qp = tl.load(q_pe + offset_qp + tl.arange(0, head_dim_kpe), mask=True, other=0.0).to(tl.float32)

        # Compute logits_scaled for this head: [MAX_TOKENS]
        logits_scaled = tl.full((MAX_TOKENS,), -float("inf"), dtype=tl.float32)

        # Loop over tokens up to MAX_TOKENS with masking
        for i in tl.static_range(MAX_TOKENS):
            use_i = i < L_tokens
            # idx is the token index into Kc_all/Kp_all for this batch's range
            idx = tl.load(kv_indices + base + i, mask=use_i, other=0)  # int32

            # Load Kc and Kp rows for this token
            kc_base = ckv_cache + idx * head_dim_ckv
            kp_base = kpe_cache + idx * head_dim_kpe

            kc = tl.load(kc_base + tl.arange(0, head_dim_ckv), mask=True, other=0.0).to(tl.float32)
            kp = tl.load(kp_base + tl.arange(0, head_dim_kpe), mask=True, other=0.0).to(tl.float32)

            # Dot products
            dot1 = tl.sum(qn * kc, axis=0)  # scalar float32
            dot2 = tl.sum(qp * kp, axis=0)  # scalar float32
            scaled = (dot1 + dot2) * sm_scale

            # Write into logits_scaled vector
            logits_scaled = tl.where(tl.arange(0, MAX_TOKENS) == i, scaled, logits_scaled)

        # Compute lse = logsumexp(logits_scaled) / ln(2)
        max_scaled = tl.max(logits_scaled, axis=0)
        exps = tl.exp(logits_scaled - max_scaled)
        sum_exps = tl.sum(exps, axis=0)
        ln2 = 0.6931471805599453  # 1 / log(2)
        lse_val = tl.log(sum_exps) + max_scaled  # logsumexp(scaled_logits)
        lse_val = lse_val / ln2  # divide by ln(2), matching PyTorch's run

        # Store lse[b, h] as a scalar
        tl.store(lse + b * NUM_QO_HEADS + h, lse_val)

        # Compute softmax on scaled logits
        attn = tl.exp(logits_scaled - lse_val)  # [MAX_TOKENS]

        # Output: out = sum_i attn[i] * Kc[i, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)

        for i in tl.static_range(MAX_TOKENS):
            use_i = i < L_tokens
            idx = tl.load(kv_indices + base + i, mask=use_i, other=0)  # int32
            kc_base = ckv_cache + idx * head_dim_ckv
            kc = tl.load(kc_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)
            out_vec += attn[i] * kc

        # Store output[b, h, :]
        out_base = output + b * (NUM_QO_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        tl.store(out_base + tl.arange(0, HEAD_DIM_CKV), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "Inputs must be CUDA tensors for Triton execution."

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Allocate outputs
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (batch_size,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indices, kv_indptr,
            output, lse,
            batch_size, num_qo_heads, head_dim_ckv, head_dim_kpe, kv_indptr.shape[0], kv_indices.shape[0],
            float(sm_scale),
            NUM_QO_HEADS=num_qo_heads,
            HEAD_DIM_CKV=head_dim_ckv,
            HEAD_DIM_KPE=head_dim_kpe,
            MAX_TOKENS=1024,  # upper bound; masks tokens beyond L_tokens
            num_warps=1,
        )

        # Cast output to bfloat16 to match original function's return type
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
