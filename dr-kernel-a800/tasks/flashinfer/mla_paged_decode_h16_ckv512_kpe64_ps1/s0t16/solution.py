import torch
import triton
import triton.language as tl

# Fixed constants derived from the provided PyTorch code
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64


@triton.jit
def attention_kernel(
    q_nope_ptr,      # *bf16, [B, H, 512]
    q_pe_ptr,        # *bf16, [B, H, 64]
    ckv_cache_ptr,   # *bf16, [num_pages, 1, 512]
    kpe_cache_ptr,   # *bf16, [num_pages, 1, 64]
    kv_indptr_ptr,   # *int32, [B+1]
    kv_indices_ptr,  # *int32, [N_tokens]
    output_ptr,      # *bf16, [B, H, 512]
    lse_ptr,         # *fp32, [B, H]
    sm_scale,        # fp32 scalar
    B: tl.constexpr, H: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch element

    # Compute token range for this batch element
    base = tl.load(kv_indptr_ptr + b)           # int32
    end = tl.load(kv_indptr_ptr + b + 1)        # int32
    L_tokens = end - base                        # int32 scalar

    # Loop over heads
    for h in range(H):
        # Load qn and qp as bf16, convert to fp32
        qn_off = b * H * HEAD_DIM_CKV + h * HEAD_DIM_CKV
        qp_off = b * H * HEAD_DIM_KPE + h * HEAD_DIM_KPE
        qn = tl.load(q_nope_ptr + qn_off)  # [512] bf16
        qp = tl.load(q_pe_ptr + qp_off)   # [64]  bf16
        qn = qn.to(tl.float32)            # [512] fp32
        qp = qp.to(tl.float32)            # [64]  fp32

        # Vector to hold scaled logits for up to MAX_TOKENS positions
        logits_scaled = tl.full((MAX_TOKENS,), -1e20, dtype=tl.float32)  # large negative sentinel

        # Compute per-token logits and fill logits_scaled
        for i in range(MAX_TOKENS):
            use_i = i < L_tokens
            idx = tl.load(kv_indices_ptr + base + i)  # int32
            # Load corresponding cached key rows as bf16 and convert to fp32
            Kc_row = tl.load(ckv_cache_ptr + idx * HEAD_DIM_CKV)  # [512] bf16
            Kp_row = tl.load(kpe_cache_ptr + idx * HEAD_DIM_KPE)  # [64]  bf16
            Kc_row = Kc_row.to(tl.float32)  # [512] fp32
            Kp_row = Kp_row.to(tl.float32)  # [64]  fp32

            # Compute dot products
            dot1 = tl.sum(qn * Kc_row)  # scalar fp32
            dot2 = tl.sum(qp * Kp_row)  # scalar fp32

            val = (dot1 + dot2) * sm_scale
            # Only set valid positions
            logits_scaled = tl.where(use_i, val, logits_scaled)

        # Compute lse in a numerically stable way
        max_val = tl.max(logits_scaled, axis=0)
        sum_exp = tl.sum(tl.exp(logits_scaled - max_val), axis=0)
        lse_val = tl.log(sum_exp) + max_val  # fp32 scalar
        # Divide by ln(2)
        ln2 = 0.6931471805599453  # ln(2)
        lse_val = lse_val / ln2

        # Compute attention weights (softmax)
        attn = tl.exp(logits_scaled - lse_val)  # [MAX_TOKENS] fp32

        # Initialize output vector
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)

        # Accumulate output: out_vec += attn[i] * Kc_row for valid i
        for i in range(MAX_TOKENS):
            idx = tl.load(kv_indices_ptr + base + i)  # int32
            Kc_row = tl.load(ckv_cache_ptr + idx * HEAD_DIM_CKV)  # [512] bf16
            Kc_row = Kc_row.to(tl.float32)  # [512] fp32
            # attn[i] is ~0 for invalid positions due to sentinel logits_scaled being large negative.
            out_vec += attn[i] * Kc_row

        # Store outputs
        out_bf16 = out_vec.to(tl.bfloat16)
        tl.store(output_ptr + qn_off, out_bf16)

        # Store lse[b, h] (float32)
        tl.store(lse_ptr + b * H + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA and tensors are contiguous
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA device for Triton kernel."

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        assert H == NUM_QO_HEADS
        assert q_nope.shape[-1] == HEAD_DIM_CKV
        assert q_pe.shape[-1] == HEAD_DIM_KPE
        assert ckv_cache.shape[-1] == HEAD_DIM_CKV
        assert kpe_cache.shape[-1] == HEAD_DIM_KPE

        # Prepare output and lse
        output = torch.empty((B, H, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Use MAX_TOKENS = 1024 to cover typical segments; mask out tokens beyond L_tokens
        MAX_TOKENS = 1024

        # Ensure inputs are contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            B, H, MAX_TOKENS,
            num_warps=4,  # modest parallelism; can be tuned
            num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
