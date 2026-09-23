import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr,      # *bf16 [B, H, 512]
    q_pe_ptr,        # *bf16 [B, H, 64]
    ckv_cache_ptr,   # *bf16 [N, 1, 512]
    kpe_cache_ptr,   # *bf16 [N, 1, 64]
    kv_indptr_ptr,   # *int32 [B + 1]
    kv_indices_ptr,  # *int32 [M]
    out_ptr,         # *bf16 [B, H, 512] (we will cast from fp32 in kernel)
    lse_ptr,         # *fp32 [B, H]
    B: tl.constexpr,            # batch size
    H: tl.constexpr,            # num heads
    N: tl.constexpr,            # num cache entries (ignored, we use kv_indptr)
    SM_SCALE: tl.constexpr,     # scaling factor
    HEAD_DIM_CKV: tl.constexpr, # 512
    HEAD_DIM_KPE: tl.constexpr, # 64
    MAX_TOKENS: tl.constexpr,   # max segment length we support
):
    b = tl.program_id(0)  # one program per batch element

    # Compute token range for this batch element
    base = tl.load(kv_indptr_ptr + b)            # int32
    end = tl.load(kv_indptr_ptr + b + 1)        # int32
    L_tokens = end - base                        # int32

    # Loop over heads; we assume H is known at launch (from host)
    for h in range(0, H):
        # Load q_nope[b, h, :] and q_pe[b, h, :] as float32
        # q_nope_ptr[b, h, :] -> offset = b*H*512 + h*512
        qn = tl.load(q_nope_ptr + b * H * HEAD_DIM_CKV + h * HEAD_DIM_CKV)
        qp = tl.load(q_pe_ptr + b * H * HEAD_DIM_KPE + h * HEAD_DIM_KPE)

        # Cast to fp32 for computation
        qn = qn.to(tl.float32)   # [512]
        qp = qp.to(tl.float32)   # [64]

        # Prepare logits_scaled vector (avoid -inf to prevent NaNs in log/softmax)
        # Use a large negative number so exp(logits_scaled - lse) -> 0 for masked positions
        large_neg = -1e20
        logits_scaled = [large_neg] * MAX_TOKENS  # vector of float32

        # Accumulate logits for valid tokens
        # For each token i in [0, L_tokens)
        for i in range(0, MAX_TOKENS):
            use_i = i < L_tokens
            # Compute token index
            idx = tl.load(kv_indices_ptr + base + i, mask=use_i, other=0)  # int32
            # Gather Kc_row and Kp_row
            # ckv_cache_ptr[idx, 0, :] -> offset idx * HEAD_DIM_CKV
            Kc_row = tl.load(ckv_cache_ptr + idx * HEAD_DIM_CKV)  # [512] bf16
            Kp_row = tl.load(kpe_cache_ptr + idx * HEAD_DIM_KPE)  # [64]  bf16

            # Cast to fp32 and compute dot products
            Kc_row = Kc_row.to(tl.float32)
            Kp_row = Kp_row.to(tl.float32)

            dot1 = 0.0
            # sum over 512 dims
            for j in range(0, HEAD_DIM_CKV):
                dot1 += qn[j] * Kc_row[j]

            dot2 = 0.0
            for j in range(0, HEAD_DIM_KPE):
                dot2 += qp[j] * Kp_row[j]

            logit = (dot1 + dot2) * SM_SCALE
            logits_scaled[i] = tl.where(use_i, logit, large_neg)

        # Compute lse per head: stable logsumexp over valid logits
        # max_val = max(logits_scaled)
        max_val = logits_scaled[0]
        for i in range(1, MAX_TOKENS):
            max_val = tl.maximum(max_val, logits_scaled[i])

        # sum_exp = sum(exp(logits_scaled - max_val))
        sum_exp = 0.0
        for i in range(0, MAX_TOKENS):
            sum_exp += tl.exp(logits_scaled[i] - max_val)

        lse_val = tl.log(sum_exp) + max_val  # float32
        # Compute in base-2: divide by ln(2)
        ln2 = 0.6931471805599453  # float
        lse_val = lse_val / ln2

        # Store lse for this (b, h)
        tl.store(lse_ptr + b * H + h, lse_val)

        # Compute attention weights and output vector
        out_vec = [0.0] * HEAD_DIM_CKV  # float32
        for i in range(0, MAX_TOKENS):
            use_i = i < L_tokens
            logit_i = logits_scaled[i]
            attn_i = tl.exp(logit_i - lse_val)  # float32, 0 for masked positions
            attn_i = tl.where(use_i, attn_i, 0.0)
            Kc_row_i = tl.load(ckv_cache_ptr + (base + i) * HEAD_DIM_CKV, mask=use_i, other=0)
            Kc_row_i = Kc_row_i.to(tl.float32)  # [512]
            # out_vec += attn_i * Kc_row_i
            for j in range(0, HEAD_DIM_CKV):
                out_vec[j] += attn_i * Kc_row_i[j]

        # Store output vector for this (b, h)
        out_offset = b * H * HEAD_DIM_CKV + h * HEAD_DIM_CKV
        # Store as fp32 then cast to bfloat16 on host side if needed
        for j in range(0, HEAD_DIM_CKV):
            tl.store(out_ptr + out_offset + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA for Triton
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors"
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        # Allocate outputs
        out = torch.empty((B, H, 512), dtype=torch.float32, device=device)  # fp32 compute
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
            out, lse,
            B=B, H=H, N=ckv_cache.shape[0], SM_SCALE=float(sm_scale),
            HEAD_DIM_CKV=512, HEAD_DIM_KPE=64, MAX_TOKENS=1024,
            num_warps=1, num_stages=1
        )

        # Cast output to bfloat16 to match original
        out_bf16 = out.to(torch.bfloat16)

        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
