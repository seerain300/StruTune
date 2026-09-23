import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: per (batch b, query head h), compute:
# - lse[b, h] = logsumexp(scaled_logits) / ln(2)
# - out[b, h, :] = softmax(scaled_logits) @ v_i for each token i
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,            # *float32, [B, NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32,   [B+1]
        kv_indices_ptr,   # *int32,   [NUM_KV_INDICES]
        out_ptr,          # *float32, [B, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, [B, NUM_QO_HEADS]
        B: tl.constexpr,                 # batch_size
        NUM_QO_HEADS: tl.constexpr,      # 32
        NUM_KV_HEADS: tl.constexpr,      # 8
        HEAD_DIM: tl.constexpr,          # 128
        NUM_TOKS: tl.constexpr,          # upper bound (e.g., 8192)
        sm_scale: tl.constexpr,          # float32 scalar
        half_ln2_inv: tl.constexpr       # float32, 1 / ln(2)
    ):
        b = tl.program_id(0)  # one program per batch
        h = tl.program_id(1)  # one program per query head

        # GQA mapping: kv_head = h // (NUM_QO_HEADS // NUM_KV_HEADS) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h] vector
        q_vec = tl.load(q_ptr + b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

        # Compute lse components
        max_s = -float("inf")
        sum_exp = 0.0  # scalar float32

        # Pass 1: compute max and sum(exp(s - max)) across tokens
        start = tl.load(kv_indptr_ptr + b)     # int32
        end = tl.load(kv_indptr_ptr + b + 1)   # int32
        num_tokens_actual = end - start        # int32

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows for this kv_head
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32
        # Store lse for this (b, h)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar float32

            # Accumulate output vector
            out_vec += attn * v_vec  # elementwise

        # Store output for this (b, h)
        out_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_off, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, num_qo_heads=32, num_kv_heads=8, head_dim=128, num_tokens_upper_bound=8192):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_tokens_upper_bound = num_tokens_upper_bound
        self.sm_scale = 1.0 / math.sqrt(head_dim)
        self.half_ln2_inv = 1.0 / math.log(2.0)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton is not available, fallback to a pure-PyTorch implementation
        if not TRITON_AVAILABLE:
            # Compute using original logic in PyTorch (for robustness)
            B, NUM_QO_HEADS, HEAD_DIM = q.shape
            assert NUM_QO_HEADS == self.num_qo_heads and self.head_dim == HEAD_DIM
            num_kv_heads = self.num_kv_heads
            gqa_ratio = NUM_QO_HEADS // num_kv_heads
            output = torch.empty((B, NUM_QO_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
            lse = torch.full((B, NUM_QO_HEADS), -float("inf"), dtype=torch.float32, device=q.device)

            for b in range(B):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                if start >= end:
                    output[b].zero_()
                    lse[b].zero_()
                    continue

                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]

                # k_cache, v_cache are [num_pages, 1, num_kv_heads, head_dim]
                k_batch = k_cache[token_indices].squeeze(1)  # [num_tokens, num_kv_heads, head_dim]
                v_batch = v_cache[token_indices].squeeze(1)  # [num_tokens, num_kv_heads, head_dim]

                for h in range(NUM_QO_HEADS):
                    kv_head = h // gqa_ratio
                    q_head = q[b, h]  # [head_dim], bfloat16 -> use float32 compute
                    k_head = k_batch[:, kv_head]  # [num_tokens, head_dim], float32
                    v_head = v_batch[:, kv_head]  # [num_tokens, head_dim], float32

                    logits = torch.matmul(q_head.float(), k_head.float().T)  # [num_tokens]
                    scaled = logits * sm_scale
                    lse[b, h] = torch.logsumexp(scaled, dim=0) / math.log(2.0)

                    attn = torch.softmax(scaled, dim=0)
                    out_head = torch.matmul(attn, v_head.float())  # [head_dim], float32
                    output[b, h] = out_head.to(torch.bfloat16)

            return output, lse

        # Triton path: ensure CUDA tensors
        B = q.shape[0]
        NUM_QO_HEADS = self.num_qo_heads
        NUM_KV_HEADS = self.num_kv_heads
        HEAD_DIM = self.head_dim
        NUM_TOKS = self.num_tokens_upper_bound

        # Cast to float32 for compute
        q_f32 = q.contiguous().to(torch.float32)                # [B, 32, 128]
        k_f32 = k_cache.contiguous().to(torch.float32)          # [num_pages, 8, 128]
        v_f32 = v_cache.contiguous().to(torch.float32)          # [num_pages, 8, 128]
        kv_indptr_i32 = kv_indptr.contiguous().to(torch.int32)  # [B+1]
        kv_indices_i32 = kv_indices.contiguous().to(torch.int32)  # [num_kv_indices]

        # Allocate outputs (float32 compute)
        out = torch.empty((B, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, NUM_QO_HEADS), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, NUM_QO_HEADS)
        _attention_bh_kernel[grid](
            q_f32, k_f32, v_f32, kv_indptr_i32, kv_indices_i32, out, lse,
            B, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, NUM_TOKS, self.sm_scale, self.half_ln2_inv,
            num_warps=4, num_stages=2
        )

        # Return bfloat16 output (matching original), and float32 lse
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
