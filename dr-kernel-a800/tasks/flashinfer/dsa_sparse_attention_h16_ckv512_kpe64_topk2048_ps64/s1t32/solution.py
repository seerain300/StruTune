import math
import torch
import triton
import triton.language as tl


# Kernel 1: Compute logits_scaled[t, h, k] = (q_nope[t, h] @ Kc_all[idx]) + (q_pe[t, h] @ Kp_all[idx]) * sm_scale
@triton.jit
def compute_logits_scaled_kernel(
    q_nope_ptr,      # *f32, [num_tokens, num_qo_heads, head_dim_ckv]
    q_pe_ptr,        # *f32, [num_tokens, num_qo_heads, head_dim_kpe]
    Kc_all_ptr,      # *f32, [num_total_kv, head_dim_ckv]
    Kp_all_ptr,      # *f32, [num_total_kv, head_dim_kpe]
    sparse_i32_ptr,  # *i32, [num_tokens, topk]
    logits_ptr,      # *f32, [num_tokens, num_qo_heads, topk]
    num_tokens: tl.int32,
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    topk: tl.constexpr,
    sm_scale: tl.float32,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    base_q_n = t * num_qo_heads * head_dim_ckv + h * head_dim_ckv
    base_q_p = t * num_qo_heads * head_dim_kpe + h * head_dim_kpe

    for k in range(0, topk):
        idx = tl.load(sparse_i32_ptr + t * topk + k)
        valid = idx != -1

        # Compute contributions: dot products
        contrib1 = 0.0
        for d in range(0, head_dim_ckv):
            q_elem = tl.load(q_nope_ptr + base_q_n + d)
            kc_elem = tl.load(Kc_all_ptr + idx * head_dim_ckv + d) if valid else 0.0
            contrib1 += q_elem * kc_elem

        contrib2 = 0.0
        for e in range(0, head_dim_kpe):
            qp_elem = tl.load(q_pe_ptr + base_q_p + e)
            kp_elem = tl.load(Kp_all_ptr + idx * head_dim_kpe + e) if valid else 0.0
            contrib2 += qp_elem * kp_elem

        scaled = (contrib1 + contrib2) * sm_scale
        tl.store(logits_ptr + t * (num_qo_heads * topk) + h * topk + k, scaled)


# Kernel 2: Compute lse[t, h] and attn[t, h, k] from scaled logits in Triton (two passes).
# We compute lse in natural log units: lse = lnsumexp(logits_scaled) / ln(2)
@triton.jit
def compute_lse_and_attn_kernel(
    logits_ptr,      # *f32, [num_tokens, num_qo_heads, topk]
    lse_ptr,         # *f32, [num_tokens, num_qo_heads]
    attn_ptr,        # *f32, [num_tokens, num_qo_heads, topk]
    num_tokens: tl.int32,
    num_qo_heads: tl.constexpr,
    topk: tl.constexpr,
    ln2: tl.float32,  # 1 / ln(2), but we compute lse as lnsumexp / ln(2) to match original
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # First pass: compute max m and sum_exp = sum(exp(logits - m))
    m = -float('inf')
    sum_exp = 0.0
    for k in range(0, topk):
        val = tl.load(logits_ptr + t * (num_qo_heads * topk) + h * topk + k)
        m = tl.maximum(m, val)
        sum_exp += tl.exp(val - m)

    # lse = lnsumexp / ln(2)
    lse = tl.log(sum_exp) / ln2
    tl.store(lse_ptr + t * num_qo_heads + h, lse)

    # Second pass: compute attn = exp(logits - lse)
    for k in range(0, topk):
        val = tl.load(logits_ptr + t * (num_qo_heads * topk) + h * topk + k)
        attn_k = tl.exp(val - lse)
        tl.store(attn_ptr + t * (num_qo_heads * topk) + h * topk + k, attn_k)


# Kernel 3: Compute output[t, h, :] = sum_k attn[t, h, k] * Kc_all[ sparse_indices[t, k], :]
# Scalar accumulation per head to avoid Triton vectorized masked store issues.
@triton.jit
def compute_output_kernel(
    attn_ptr,        # *f32, [num_tokens, num_qo_heads, topk]
    Kc_all_ptr,      # *f32, [num_total_kv, head_dim_ckv]
    sparse_i32_ptr,  # *i32, [num_tokens, topk]
    output_ptr,      # *f32, [num_tokens, num_qo_heads, head_dim_ckv]
    num_tokens: tl.int32,
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    topk: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    base_out = t * num_qo_heads * head_dim_ckv + h * head_dim_ckv
    # initialize output to zeros
    for d in range(0, head_dim_ckv):
        tl.store(output_ptr + base_out + d, 0.0)

    # accumulate out = sum_k attn[t, h, k] * Kc_all[idx, :]
    for k in range(0, topk):
        idx = tl.load(sparse_i32_ptr + t * topk + k)
        valid = idx != -1
        if valid:
            attn_k = tl.load(attn_ptr + t * (num_qo_heads * topk) + h * topk + k)
            # scalar add across d
            for d in range(0, head_dim_ckv):
                kc_elem = tl.load(Kc_all_ptr + idx * head_dim_ckv + d)
                curr = tl.load(output_ptr + base_out + d)
                tl.store(output_ptr + base_out + d, curr + attn_k * kc_elem)


class ModelNew(torch.nn.Module):
    def __init__(self, topk=2048, head_dim_ckv=512, head_dim_kpe=64, num_qo_heads=16, sm_scale=1.0):
        super().__init__()
        self.topk = topk
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.num_qo_heads = num_qo_heads
        self.sm_scale = sm_scale
        self.ln2 = 1.0 / math.log(2.0)  # 1 / ln(2)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale=None):
        # Ensure inputs are contiguous and on the same device
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        num_qo_heads2, _, head_dim_kpe = q_pe.shape
        assert num_qo_heads == self.num_qo_heads and head_dim_ckv == self.head_dim_ckv and head_dim_kpe == self.head_dim_kpe

        # Cast q tensors to float32 for compute
        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()

        # Flatten K caches to [num_total_kv, dim] and cast to float32
        num_pages = ckv_cache.shape[0]
        assert ckv_cache.shape[1] == 64 and ckv_cache.shape[2] == self.head_dim_ckv
        num_pages_kpe = kpe_cache.shape[0]
        assert kpe_cache.shape[1] == 64 and kpe_cache.shape[2] == self.head_dim_kpe
        total_kv = num_pages * 64
        assert num_pages_kpe == num_pages, "kpe_cache must have same num_pages as ckv_cache"

        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).to(torch.float32).contiguous()
        Kp_all = kpe_cache.reshape(-1, self.head_dim_kpe).to(torch.float32).contiguous()

        # sparse_indices to int32
        sparse_i32 = sparse_indices.to(torch.int32).contiguous()

        # Allocate buffers
        logits_scaled = torch.empty((num_tokens, self.num_qo_heads, self.topk), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((num_tokens, self.num_qo_heads), dtype=torch.float32, device=q_nope.device)
        attn = torch.empty((num_tokens, self.num_qo_heads, self.topk), dtype=torch.float32, device=q_nope.device)
        output = torch.empty((num_tokens, self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=q_nope.device)

        grid = (num_tokens, self.num_qo_heads)

        # Kernel 1: compute logits_scaled (Triton-only)
        compute_logits_scaled_kernel[grid](
            q_nope_f32, q_pe_f32, Kc_all, Kp_all, sparse_i32, logits_scaled,
            num_tokens=num_tokens,
            num_qo_heads=self.num_qo_heads,
            head_dim_ckv=self.head_dim_ckv,
            head_dim_kpe=self.head_dim_kpe,
            topk=self.topk,
            sm_scale=self.sm_scale if sm_scale is None else sm_scale,
            num_warps=4, num_stages=2
        )

        # Kernel 2: compute lse and attn (Triton-only, two-pass stable reduction + softmax equivalent)
        compute_lse_and_attn_kernel[grid](
            logits_scaled, lse, attn,
            num_tokens=num_tokens,
            num_qo_heads=self.num_qo_heads,
            topk=self.topk,
            ln2=self.ln2,
            num_warps=4, num_stages=2
        )

        # Kernel 3: compute final output per (t, h) (Triton-only scalar accumulation)
        compute_output_kernel[grid](
            attn, Kc_all, sparse_i32, output,
            num_tokens=num_tokens,
            num_qo_heads=self.num_qo_heads,
            head_dim_ckv=self.head_dim_ckv,
            topk=self.topk,
            num_warps=4, num_stages=2
        )

        # Return output as bfloat16 to match original interface, lse as float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
