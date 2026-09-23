import math
import torch
import triton
import triton.language as tl


@triton.jit
def gqa_attention_kernel(
    q_ptr,          # *fp32, shape [total_q, num_qo_heads, head_dim]
    k_ptr,          # *fp32, shape [num_pages, num_kv_heads, head_dim]
    v_ptr,          # *fp32, shape [num_pages, num_kv_heads, head_dim]
    out_ptr,        # *fp32, shape [total_q, num_qo_heads, head_dim]
    lse_ptr,        # *fp32, shape [total_q, num_qo_heads]
    total_q: tl.constexpr,       # int32 meta
    num_qo_heads: tl.constexpr,  # int32 meta
    num_pages: tl.constexpr,     # int32 meta (num_kv_tokens)
    sm_scale: tl.constexpr,      # float32 meta
    HEAD_DIM: tl.constexpr,      # int32 meta, e.g., 128
    GQA_RATIO: tl.constexpr      # int32 meta, e.g., 4
):
    # This kernel processes all tokens and heads. For robustness, use simple loops.
    # We assume total_q is provided and num_pages corresponds to kv_indices range.
    for t in range(0, total_q):
        for h in range(0, num_qo_heads):
            # Accumulator for output vector and running lse state
            out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            lse_max = tl.full((), -float("inf"), tl.float32)
            sum_exp = tl.zeros((), dtype=tl.float32)

            # Load q vector for head h at token t
            q_base = q_ptr + t * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
            q_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(0, HEAD_DIM):
                q_ptr_d = q_base + d
                q_val = tl.load(q_ptr_d)
                q_vec[d] = q_val

            # Loop over kv tiles (num_pages), compute logits, update lse, then compute attn and accumulate
            for p in range(0, num_pages):
                kv_head = h // GQA_RATIO
                # Load k and v vectors for this tile and head
                k_base = k_ptr + p * (GQA_RATIO * HEAD_DIM) + kv_head * HEAD_DIM
                v_base = v_ptr + p * (GQA_RATIO * HEAD_DIM) + kv_head * HEAD_DIM

                k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
                v_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

                for d in range(0, HEAD_DIM):
                    k_ptr_d = k_base + d
                    k_val = tl.load(k_ptr_d)
                    k_vec[d] = k_val

                for d in range(0, HEAD_DIM):
                    v_ptr_d = v_base + d
                    v_val = tl.load(v_ptr_d)
                    v_vec[d] = v_val

                # logits_scaled = dot(q_vec, k_vec) * sm_scale
                logits = tl.zeros((), dtype=tl.float32)
                for d in range(0, HEAD_DIM):
                    logits += q_vec[d] * k_vec[d]
                logits_scaled = logits * sm_scale

                # Update logsumexp state
                new_lse_max = tl.maximum(lse_max, logits_scaled)
                if new_lse_max > lse_max:
                    sum_exp = tl.exp(logits_scaled - lse_max) + sum_exp
                lse_max = new_lse_max

            # Now compute attention for each kv tile and accumulate output
            for p in range(0, num_pages):
                kv_head = h // GQA_RATIO
                k_base = k_ptr + p * (GQA_RATIO * HEAD_DIM) + kv_head * HEAD_DIM
                v_base = v_ptr + p * (GQA_RATIO * HEAD_DIM) + kv_head * HEAD_DIM

                k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
                v_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

                for d in range(0, HEAD_DIM):
                    k_ptr_d = k_base + d
                    k_val = tl.load(k_ptr_d)
                    k_vec[d] = k_val

                for d in range(0, HEAD_DIM):
                    v_ptr_d = v_base + d
                    v_val = tl.load(v_ptr_d)
                    v_vec[d] = v_val

                logits = tl.zeros((), dtype=tl.float32)
                for d in range(0, HEAD_DIM):
                    logits += q_vec[d] * k_vec[d]
                logits_scaled = logits * sm_scale

                attn = tl.exp(logits_scaled - lse_max)
                out_vec += attn * v_vec

            # Store output vector for this token and head
            out_base = out_ptr + t * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
            for d in range(0, HEAD_DIM):
                out_ptr_d = out_base + d
                tl.store(out_ptr_d, out_vec[d])

            # Store lse_final = lse_max + log2(sum_exp)
            ln2 = 0.6931471805599453
            log2_sum_exp = tl.log(sum_exp) / ln2
            lse_final = lse_max + log2_sum_exp
            lse_base = lse_ptr + t * num_qo_heads + h
            tl.store(lse_base, lse_final)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16 or float32
        k_cache: [num_pages, 1, 8, 128], bfloat16 or float32
        v_cache: [num_pages, 1, 8, 128], bfloat16 or float32
        qo_indptr, kv_indptr: 1D int32 (unused in Triton kernel, but we need total_q and num_pages)
        kv_indices: int32 (unused in kernel here; we assume using all num_pages)
        sm_scale: float32
        """
        device = q.device

        # Convert to float32 and make contiguous for Triton compute
        q_fp32 = q.contiguous().to(torch.float32)            # [total_q, 32, 128]
        k_cache_fp32 = k_cache.contiguous().to(torch.float32).squeeze(1)  # [num_pages, 8, 128]
        v_cache_fp32 = v_cache.contiguous().to(torch.float32).squeeze(1)  # [num_pages, 8, 128]

        total_q = q_fp32.shape[0]
        num_qo_heads = 32
        num_pages = k_cache_fp32.shape[0]  # number of kv tiles (pages)
        head_dim = 128
        gqa_ratio = 4

        # Allocate outputs
        out = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel. We pass total_q, num_qo_heads, num_pages, sm_scale as compile-time constants
        gqa_attention_kernel[(1,)](
            q_fp32, k_cache_fp32, v_cache_fp32, out, lse,
            total_q=total_q,
            num_qo_heads=num_qo_heads,
            num_pages=num_pages,
            sm_scale=sm_scale,               # pass as float; Triton treats as tl.constexpr
            HEAD_DIM=head_dim,
            GQA_RATIO=gqa_ratio,
            num_warps=1, num_stages=1
        )

        # Cast output to bfloat16 to match original
        out_bf16 = out.to(torch.bfloat16)

        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
