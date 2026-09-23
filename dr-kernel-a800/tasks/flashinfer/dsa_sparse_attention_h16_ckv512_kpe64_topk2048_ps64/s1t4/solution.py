import math
import torch
import triton
import triton.language as tl


# Kernel: Compute lse and attn per (t, h) directly from q_nope, q_pe, Kc_all, Kp_all, sparse_indices.
# We do two passes: first to get max and sum_exp, second to write attn.
@triton.jit
def compute_lse_attn_kernel(
    q_nope_ptr,      # *f32, [num_tokens, num_qo_heads, head_dim_ckv]
    q_pe_ptr,        # *f32, [num_tokens, num_qo_heads, head_dim_kpe]
    Kc_all_ptr,      # *f32, [num_total_kv, head_dim_ckv]
    Kp_all_ptr,      # *f32, [num_total_kv, head_dim_kpe]
    sparse_i32_ptr,  # *i32, [num_tokens, topk]
    lse_ptr,         # *f32, [num_tokens, num_qo_heads]
    attn_ptr,        # *f32, [num_tokens, num_qo_heads, topk]
    num_tokens: tl.int32,
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    topk: tl.constexpr,
    sm_scale: tl.float32,
    ln2: tl.float32,  # 1 / ln(2)
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    base_q_n = t * num_qo_heads * head_dim_ckv + h * head_dim_ckv
    base_q_p = t * num_qo_heads * head_dim_kpe + h * head_dim_kpe

    # First pass: compute max m and sum_exp
    m = -float('inf')
    sum_exp = 0.0
    for k in range(0, topk):
        idx = tl.load(sparse_i32_ptr + t * topk + k)
        valid = idx != -1

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
        # update max
        m = tl.maximum(m, scaled)
        # update sum_exp using shifted exponent for numerical stability
        sum_exp += tl.exp((scaled - m))

    # Compute lse = m + log(sum_exp) / ln(2)
    lse = m + tl.log(sum_exp) * ln2
    tl.store(lse_ptr + t * num_qo_heads + h, lse)

    # Second pass: compute attn = exp(scaled - lse) and store
    for k in range(0, topk):
        idx = tl.load(sparse_i32_ptr + t * topk + k)
        valid = idx != -1

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
        attn_k = tl.exp(scaled - lse) if valid else 0.0
        tl.store(attn_ptr + t * (num_qo_heads * topk) + h * topk + k, attn_k)


# Kernel: Compute final output per (t, h) using attn and Kc_all.
# We do scalar accumulation per output element d over all k (topk).
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
        self.ln2 = 1.0 / math.log(2.0)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale=None):
        # Shapes and assertions
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        num_qo_heads2, _, head_dim_kpe = q_pe.shape
        assert num_qo_heads == self.num_qo_heads, "num_qo_heads mismatch"
        assert head_dim_ckv == self.head_dim_ckv, "head_dim_ckv mismatch"
        assert head_dim_kpe == self.head_dim_kpe, "head_dim_kpe mismatch"

        # Cast q tensors to float32 for computation (matches original behavior)
        q_nope_f32 = q_nope.contiguous().to(torch.float32)
        q_pe_f32 = q_pe.contiguous().to(torch.float32)

        # Flatten K caches to [num_total_kv, dim] and cast to float32
        total_kv = ckv_cache.shape[0] * ckv_cache.shape[1]
        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).contiguous().to(torch.float32)
        Kp_all = kpe_cache.reshape(-1, self.head_dim_kpe).contiguous().to(torch.float32)

        # sparse_indices to int32
        sparse_i32 = sparse_indices.contiguous().to(torch.int32)

        # Allocate buffers
        lse = torch.empty((num_tokens, self.num_qo_heads), dtype=torch.float32, device=q_nope.device)
        attn = torch.empty((num_tokens, self.num_qo_heads, self.topk), dtype=torch.float32, device=q_nope.device)
        output = torch.empty((num_tokens, self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=q_nope.device)

        grid = (num_tokens, self.num_qo_heads)

        # Kernel: compute lse and attn per (t,h)
        compute_lse_attn_kernel[grid](
            q_nope_f32, q_pe_f32, Kc_all, Kp_all, sparse_i32, lse, attn,
            num_tokens=num_tokens,
            num_qo_heads=self.num_qo_heads,
            head_dim_ckv=self.head_dim_ckv,
            head_dim_kpe=self.head_dim_kpe,
            topk=self.topk,
            sm_scale=self.sm_scale if sm_scale is None else sm_scale,
            ln2=self.ln2,
            num_warps=4, num_stages=2
        )

        # Kernel: compute final output per (t,h)
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
