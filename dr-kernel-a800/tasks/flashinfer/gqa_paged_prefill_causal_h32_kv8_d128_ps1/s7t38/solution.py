import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,          # *f32, [total_q, 32, 128]
    k_ptr,          # *f32, [num_pages, 8, 128]
    v_ptr,          # *f32, [num_pages, 8, 128]
    kv_indices_ptr, # *i32, [num_kv_indices]
    output_ptr,     # *f32, [total_q, 32, 128]
    output_lse_ptr, # *f32, [total_q, 32]
    # segment bounds (scalars)
    q_start: tl.int32,
    q_end: tl.int32,
    kv_start: tl.int32,
    kv_end: tl.int32,
    # constants
    head_dim: tl.int32,           # 128
    num_qo_heads: tl.int32,       # 32
    num_kv_heads: tl.int32,       # 8
    gqa_ratio: tl.int32,          # 4
    sm_scale: tl.float32,
    ln2_inv: tl.float32,          # 1 / ln(2)
    MAX_Q_SEG: tl.constexpr,
    MAX_KV_SEG: tl.constexpr,
):
    # One program per segment
    b = tl.program_id(0)

    # Compute segment sizes
    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    dim = tl.arange(0, head_dim)

    # Iterate over query tokens in this segment
    for q_i in range(0, MAX_Q_SEG):
        q_valid = q_i < num_q_tokens_segment
        global_q_idx = q_start + q_i
        if not q_valid:
            continue

        # Iterate over query heads
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio

            # Load q[h] vector
            q_vec = tl.load(q_ptr + global_q_idx * (num_qo_heads * head_dim) + h * head_dim + dim)

            # Compute logsumexp over key list: max_val and sum_exp (numerically stable)
            max_val = -float("inf")
            sum_exp = 0.0

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv
            tl.store(output_lse_ptr + global_q_idx * num_qo_heads + h, lse_val)

            # Compute output vector: softmax-weighted sum over valid keys with causal mask
            # max_kv_idx = min(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)
            delta = num_kv_tokens - num_q_tokens_segment
            max_kv_idx = q_i + 1 + delta
            if max_kv_idx > num_kv_tokens:
                max_kv_idx = num_kv_tokens

            sum_softmax = 0.0
            out_vec = tl.zeros([head_dim], dtype=tl.float32)

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                valid = kk < max_kv_idx
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                prod = tl.sum(q_vec * k_vec, axis=0) * sm_scale
                if valid:
                    expv = tl.exp(prod)
                    sum_softmax += expv
                # For invalid, we still need to consider zero attn
                v_vec = tl.load(v_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                out_vec += (expv if valid else 0.0) * v_vec

            # Normalize by sum_softmax if > 0
            if sum_softmax > 0.0:
                out_vec = out_vec / sum_softmax

            tl.store(output_ptr + global_q_idx * (num_qo_heads * head_dim) + h * head_dim + dim, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, total_q, num_qo_heads=32, head_dim=128, num_kv_heads=8, gqa_ratio=4, max_q_seg=64, max_kv_seg=256):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.gqa_ratio = gqa_ratio
        self.max_q_seg = max_q_seg
        self.max_kv_seg = max_kv_seg

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and dtype
        device = q.device
        q_f32 = q.contiguous().to(torch.float32)                 # [total_q, 32, 128]
        k_f32 = k_cache.contiguous().to(torch.float32)          # [num_pages, 1, 8, 128]
        v_f32 = v_cache.contiguous().to(torch.float32)          # [num_pages, 1, 8, 128]
        # Flatten k/v for kv_heads (original k_cache has shape [num_pages, 1, num_kv_heads, head_dim])
        # We'll still pass the full [num_pages, num_kv_heads, head_dim] view
        total_q = q_f32.shape[0]
        num_qo_heads = self.num_qo_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        gqa_ratio = self.gqa_ratio

        # Allocate outputs (float32 as per original lse dtype)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.numel() - 1,)
        attention_kernel[grid](
            q_f32, k_f32, v_f32, kv_indices.to(torch.int32),
            output, lse,
            qo_indptr[0].item(), qo_indptr[1].item(),
            kv_indptr[0].item(), kv_indptr[1].item(),
            head_dim, num_qo_heads, num_kv_heads, gqa_ratio,
            sm_scale, 1.0 / math.log(2.0),
            MAX_Q_SEG=self.max_q_seg, MAX_KV_SEG=self.max_kv_seg,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
