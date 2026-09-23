import torch
import triton
import triton.language as tl
import math


@triton.jit
def attention_kernel(
    q_nope_ptr,     # *bf16, shape [B, H, 512]
    q_pe_ptr,       # *bf16, shape [B, H, 64]
    ckv_cache_ptr,  # *bf16, shape [N, 1, 512]
    kpe_cache_ptr,  # *bf16, shape [N, 1, 64]
    kv_indptr_ptr,  # *int32, shape [B + 1]
    kv_indices_ptr, # *int32, shape [L]
    out_ptr,        # *bf16, shape [B, H, 512]
    lse_ptr,        # *float32, shape [B, H]
    sm_scale,       # float32 scalar
    B: tl.constexpr, H: tl.constexpr, MAX_TOKENS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr, HEAD_DIM_KPE: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch element

    # Compute token range for this batch element
    base = tl.load(kv_indptr_ptr + b)        # int32
    end = tl.load(kv_indptr_ptr + b + 1)     # int32
    L_tokens = end - base                     # int32 scalar

    # Loop over heads
    for h in range(H):
        # Compute offsets for q_nope[b, h, :] and q_pe[b, h, :]
        off_qn = b * H * HEAD_DIM_CKV + h * HEAD_DIM_CKV
        off_qp = b * H * HEAD_DIM_KPE + h * HEAD_DIM_KPE

        # Load qn and qp as bf16, cast to fp32
        qn = tl.load(q_nope_ptr + off_qn)     # [512] bf16
        qp = tl.load(q_pe_ptr + off_qp)       # [64]  bf16
        qn = qn.to(tl.float32)
        qp = qp.to(tl.float32)

        # Initialize logits vector with sentinel for masked tokens
        logits_scaled = tl.full((MAX_TOKENS,), -1.0e20, dtype=tl.float32)

        # Compute logits for each token in the segment
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32 scalar
                # Load Kc_row and Kp_row as bf16, cast to fp32
                off_kc = idx * HEAD_DIM_CKV
                Kc_row = tl.load(ckv_cache_ptr + off_kc)  # [512] bf16
                off_kp = idx * HEAD_DIM_KPE
                Kp_row = tl.load(kpe_cache_ptr + off_kp)  # [64]  bf16
                Kc_row = Kc_row.to(tl.float32)
                Kp_row = Kp_row.to(tl.float32)

                # Dot products via scalar loops
                dot1 = 0.0
                for d in range(HEAD_DIM_CKV):
                    dot1 += qn[d] * Kc_row[d]
                dot2 = 0.0
                for d in range(HEAD_DIM_KPE):
                    dot2 += qp[d] * Kp_row[d]
                logits_scaled[i] = (dot1 + dot2) * sm_scale

        # Stable logsumexp over valid tokens
        max_val = tl.max(logits_scaled, axis=0)                     # scalar
        sum_exp = tl.sum(tl.exp(logits_scaled - max_val), axis=0)  # scalar
        lse_val = tl.log(sum_exp) + max_val                        # scalar
        lse_val = lse_val / math.log(2.0)                          # divide by ln(2)

        # attn for each position
        attn = tl.exp(logits_scaled - lse_val)                     # [MAX_TOKENS]

        # Output accumulation: out_vec = sum_i attn[i] * Kc_row[i]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)          # int32 scalar
                off_kc = idx * HEAD_DIM_CKV
                Kc_row = tl.load(ckv_cache_ptr + off_kc).to(tl.float32)  # [512]
                out_vec += attn[i] * Kc_row

        # Store output for this head
        off_out = b * H * HEAD_DIM_CKV + h * HEAD_DIM_CKV
        tl.store(out_ptr + off_out, out_vec.to(tl.bfloat16))

        # Store lse for this head
        tl.store(lse_ptr + b * H + h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure dtypes and shapes
        assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
        assert ckv_cache.dtype == torch.bfloat16 and kpe_cache.dtype == torch.bfloat16
        assert kv_indptr.dtype == torch.int32 and kv_indices.dtype == torch.int32

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        assert q_pe.shape[1] == H, "num_qo_heads mismatch between q_nope and q_pe"
        assert ckv_cache.shape[2] == 512, "ckv_cache last dim must be 512"
        assert kpe_cache.shape[2] == 64, "kpe_cache last dim must be 64"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr must have length batch_size + 1"

        # Allocate outputs
        output = torch.empty((B, H, 512), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Ensure tensors are contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Launch Triton kernel: one program per batch element
        MAX_TOKENS = 1024  # cover typical segments; mask out tokens beyond L_tokens
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            B=B, H=H, MAX_TOKENS=MAX_TOKENS,
            HEAD_DIM_CKV=512, HEAD_DIM_KPE=64,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
