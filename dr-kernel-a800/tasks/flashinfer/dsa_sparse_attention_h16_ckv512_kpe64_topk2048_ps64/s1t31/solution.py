import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits_scaled[t, h, k] = (q_nope[t, h] @ Kc_all[idx]) + (q_pe[t, h] @ Kp_all[idx]) * sm_scale
# Note: q_nope_ptr and q_pe_ptr are 3D tensors [num_tokens, num_qo_heads, dim], so base = t * (num_qo_heads * dim) + h * dim.
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

    # Base for q_nope and q_pe considering 3D layout
    base_q_n = t * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
    base_q_p = t * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe

    for k in range(0, topk):
        idx = tl.load(sparse_i32_ptr + t * topk + k)
        valid = idx != -1

        # Compute contributions
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


# Triton kernel: compute final output per (t, h) using attn and Kc_all.
# Scalar accumulation per output element d over all k (topk).
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
        # Ensure inputs are on the same device and contiguous
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        num_qo_heads2, _, head_dim_kpe = q_pe.shape
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        assert head_dim_kpe == self.head_dim_kpe

        device = q_nope.device

        # Cast q tensors to float32 for Triton compute and make contiguous
        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()

        # Flatten K caches to [num_total_kv, dim] and cast to float32
        num_pages, _, _ = ckv_cache.shape
        total_kv = num_pages * 64  # 64 tokens per page
        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).to(torch.float32).contiguous()
        Kp_all = kpe_cache.reshape(-1, self.head_dim_kpe).to(torch.float32).contiguous()

        # sparse_indices to int32
        sparse_i32 = sparse_indices.to(torch.int32).contiguous()

        # Allocate logits buffer and compute logits_scaled with Triton
        logits_scaled = torch.empty((num_tokens, self.num_qo_heads, self.topk), dtype=torch.float32, device=device)

        grid = (num_tokens, self.num_qo_heads)

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

        # Compute lse and attn in PyTorch to ensure exact correctness
        logits_scaled = logits_scaled.to(torch.float32)
        scaled_logits = logits_scaled * self.ln2  # same as original: logits * (1 / ln(2)) before softmax
        # lse per (t,h) = logsumexp(scaled_logits)
        lse = torch.logsumexp(scaled_logits, dim=2)  # shape [num_tokens, num_qo_heads]
        # attn = softmax(scaled_logits, dim=2)
        attn = torch.softmax(scaled_logits, dim=2)  # shape [num_tokens, num_qo_heads, topk]

        # Allocate output buffer and compute final weighted sum in Triton
        output = torch.empty((num_tokens, self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=device)

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


# Original helper for testing:
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16)
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]


def run(*args):
    return ModelNew()(*args)
