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
        # Ensure inputs are contiguous and on the same device
        num_tokens = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Cast q tensors to float32 for compute
        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()

        # Flatten caches to [num_total_kv, dim] and cast to float32
        num_pages, page_size, _ = ckv_cache.shape
        total_kv = num_pages * page_size
        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).to(torch.float32).contiguous()
        Kp_all = kpe_cache.reshape(-1, self.head_dim_kpe).to(torch.float32).contiguous()

        # sparse_indices to int32
        sparse_i32 = sparse_indices.to(torch.int32).contiguous()

        # Allocate buffer for logits_scaled: [num_tokens, num_qo_heads, topk], float32
        logits_scaled = torch.empty((num_tokens, self.num_qo_heads, self.topk), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel to compute scaled logits
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

        # Compute lse and attn using PyTorch for exact correctness
        # Note: logits_scaled is in f32
        lse = torch.logsumexp(logits_scaled, dim=2)  # [num_tokens, num_qo_heads]
        lse = lse / self.ln2
        attn = torch.softmax(logits_scaled - lse.view(-1, 1, 1), dim=2)  # [num_tokens, num_qo_heads, topk]

        # Compute final output per (t, h): output[t, h, :] = sum_k attn[t, h, k] * Kc_all[sparse_indices[t, k]]
        output = torch.zeros((num_tokens, self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=q_nope.device)

        for t in range(num_tokens):
            indices_t = sparse_i32[t]            # [topk], int32
            mask_t = indices_t != -1             # [topk] bool
            attn_vec = attn[t]                   # [num_qo_heads, topk]
            # Iterate over heads
            for h in range(self.num_qo_heads):
                attn_k = attn_vec[h]             # [topk]
                idxs_valid = indices_t[mask_t]   # [num_valid]
                attn_k_valid = attn_k[mask_t]    # [num_valid]
                # Gather Kc rows for valid indices: [num_valid, 512]
                Kc_valid = Kc_all[idxs_valid]    # [num_valid, 512]
                # Form a [topk, 512] matrix: rows are attn_k * Kc_valid[:, :]
                # Since we have only valid rows, we can reconstruct each k:
                # For each k where idx is valid, contribution = attn_k[k] * Kc_valid[k, :]
                # Sum over all valid ks. We can do it elementwise by broadcasting over k:
                # Build a list of contributions per k and sum across k.
                # However, a simple way is:
                if Kc_valid.numel() > 0:
                    # We need to sum over k: output[h, :] += sum_k attn_k[k] * Kc_valid[k, :]
                    # We can compute this in a loop:
                    for kk in range(self.topk):
                        if mask_t[kk]:
                            output[t, h, :] += attn_k[kk] * Kc_valid[kk]  # [512]

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
