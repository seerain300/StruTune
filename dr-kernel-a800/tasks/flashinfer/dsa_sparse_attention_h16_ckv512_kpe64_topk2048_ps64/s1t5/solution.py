import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits_scaled[t, h, k] = (q_nope[t, h] @ Kc_all[idx]) + (q_pe[t, h] @ Kp_all[idx]) * sm_scale
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
        # Ensure inputs are on the same device and contiguous; cast to float32 for compute
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        num_qo_heads2, _, head_dim_kpe = q_pe.shape
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        assert head_dim_kpe == self.head_dim_kpe

        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()

        # Flatten caches to [num_total_kv, dim] and cast to float32
        num_pages, _, _ = ckv_cache.shape
        total_kv = num_pages * 64  # fixed page size
        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).to(torch.float32).contiguous()
        Kp_all = kpe_cache.reshape(-1, self.head_dim_kpe).to(torch.float32).contiguous()

        sparse_i32 = sparse_indices.to(torch.int32).contiguous()

        # Allocate logits_scaled buffer
        logits_scaled = torch.empty((num_tokens, self.num_qo_heads, self.topk), dtype=torch.float32, device=q_nope.device)

        grid = (num_tokens, self.num_qo_heads)

        # Triton: compute logits_scaled
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

        # PyTorch: compute lse and attn exactly as in the original
        # lse per (t, h): logsumexp(logits_scaled[t, h, :]) / ln(2)
        lse = torch.logsumexp(logits_scaled, dim=1) / self.ln2  # [num_tokens, num_qo_heads]
        # attn per (t,h,k): softmax over topk axis
        attn = torch.softmax(logits_scaled - lse.unsqueeze(1), dim=1)  # [num_tokens, num_qo_heads, topk]

        # Compute final output: output[t, h, :] = sum_k attn[t, h, k] * Kc_all[ sparse_indices[t, k], :]
        # We use PyTorch for the final weighted sum to ensure exact matching (including handling of padding).
        output = torch.zeros((num_tokens, self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=q_nope.device)
        for t in range(num_tokens):
            for h in range(self.num_qo_heads):
                for k in range(self.topk):
                    idx = int(sparse_i32[t, k].item())
                    if idx != -1:
                        attn_k = attn[t, h, k].item()
                        Kc_vec = Kc_all[idx]  # [head_dim_ckv], float32
                        output[t, h] += attn_k * Kc_vec
        # Cast output to bfloat16 to match the original interface
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
