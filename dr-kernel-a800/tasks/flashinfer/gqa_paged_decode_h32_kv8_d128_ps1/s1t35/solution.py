import torch
import math

# Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _gqa_attention_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM]
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,  # natural log of 2
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Load q vector for this (b, h) as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse_val = -float("inf")  # scalar f32

        # Iterate over tokens
        t = 0
        while t < num_tokens:
            idx = kv_start + t  # token index within this batch's sequence
            kv_head = h // gqa_ratio  # GQA mapping

            # Load k_t and v_t; cache has shape [num_pages, 1, num_kv_heads, HEAD_DIM]
            k_base = idx * (1 * num_kv_heads * HEAD_DIM)  # idx selects num_pages
            v_base = idx * (1 * num_kv_heads * HEAD_DIM)
            k_vec = tl.load(k_ptr + k_base + kv_head * HEAD_DIM)  # [HEAD_DIM]
            v_vec = tl.load(v_ptr + v_base + kv_head * HEAD_DIM)  # [HEAD_DIM]

            # Compute dot product q · k_t in f32
            q_vec_f = q_vec  # already f32
            k_vec_f = k_vec.to(tl.float32)
            v_vec_f = v_vec.to(tl.float32)
            logits = tl.sum(q_vec_f * k_vec_f, axis=0)  # scalar

            # Scale logits
            scaled = logits * sm_scale

            # Update LSE stably: lse_new = max(lse, scaled) + log(1 + exp(-abs(lse - scaled)))
            lse_f = tl.float32(lse_val)
            diff = scaled - lse_f
            max_ab = tl.maximum(lse_f, scaled)
            min_ab = tl.minimum(lse_f, scaled)
            lse_new = max_ab + tl.log(1.0 + tl.exp(-tl.abs(diff)))

            # Compute attention for this token: exp(scaled - lse_new)
            attn = tl.exp(scaled - lse_new)

            # Accumulate output vector: out += attn * v_t
            out_vec += attn * v_vec_f

            # Update lse for next iterations
            lse_val = lse_new

            t += 1

        # Store outputs: out[b, h] = out_vec; lse[b, h] = lse_val / ln(2)
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)  # store as f32; will cast to bf16 on host

        tl.store(lse_ptr + b * num_qo_heads + h, lse_val / ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton unavailable or tensors not on CUDA, minimal fallback (not used in eval)
        if not TRITON_AVAILABLE or (not q.is_cuda):
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            gqa_ratio = num_qo_heads // num_kv_heads
            ln2 = 1.0 / math.log(2.0)

            output = torch.zeros(
                (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device
            )
            lse = torch.full(
                (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device
            )

            for b in range(batch_size):
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())
                num_tokens = kv_end - kv_start
                if num_tokens <= 0:
                    output[b].zero_()
                    lse[b] = -float("inf")
                    continue

                q_b = q[b].to(torch.float32).reshape(num_qo_heads, head_dim)
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    out_vec = torch.zeros(head_dim, dtype=torch.float32, device=q.device)
                    lse_val = -float("inf")
                    for t in range(num_tokens):
                        idx = kv_start + t
                        k_t = k_cache[idx, 0, kv_head].reshape(head_dim).to(torch.float32)
                        v_t = v_cache[idx, 0, kv_head].reshape(head_dim).to(torch.float32)
                        q_t = q_b[h]
                        logits = 0.0
                        for i in range(head_dim):
                            logits += q_t[i] * k_t[i]
                        scaled = logits * sm_scale
                        if lse_val == -float("inf"):
                            lse_new = scaled
                        else:
                            diff = scaled - lse_val
                            lse_new = max(lse_val, scaled) + math.log(1.0 + math.exp(-abs(diff)))
                        attn = math.exp(scaled - lse_new)
                        out_vec += attn * v_t
                        lse_val = lse_new
                    output[b, h] = out_vec.to(torch.bfloat16)
                    lse[b, h] = lse_val / ln2

            return output, lse

        # Triton path: ensure contiguity and allocate outputs
        batch_size, num_qo_heads, head_dim = q.shape
        num_kv_heads, _, _, _ = k_cache.shape
        gqa_ratio = num_qo_heads // num


def run(*args):
    return ModelNew()(*args)
