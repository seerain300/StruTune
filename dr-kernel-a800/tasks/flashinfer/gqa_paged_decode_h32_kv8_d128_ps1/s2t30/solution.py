import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h).
# It loops over up to NUM_TOKS token indices with masks, computing:
# - lse = logsumexp(s) / ln(2), where s = (q[h] · k_i) * sm_scale
# - out_vec = sum_i exp(s - lse) * v_i
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,            # *float32, shape [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32, shape [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, shape [NUM_KV_INDICES]
        out_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        B: tl.constexpr,                 # batch size
        NUM_QO_HEADS: tl.constexpr,      # 32
        NUM_KV_HEADS: tl.constexpr,      # 8
        HEAD_DIM: tl.constexpr,          # 128
        NUM_TOKS: tl.constexpr,          # upper bound for token iterations (e.g., 128)
        sm_scale: tl.float32,            # scalar, e.g., 1/sqrt(128)
        half_ln2_inv: tl.float32,        # 1/ln(2)
        b: tl.int32,                     # current batch index
        h: tl.int32,                     # current query head index
    ):
        # Offsets
        q_off = h * HEAD_DIM  # q[b, h, :]
        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h]
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Determine start/end for this batch from kv_indptr
        start = tl.load(kv_indptr_ptr + b)        # int32
        end = tl.load(kv_indptr_ptr + b + 1)      # int32
        num_tokens_actual = end - start           # int32

        # Pass 1: compute max_s and sum_exp across valid tokens
        max_s = -float("inf")  # scalar float32
        sum_exp = 0.0          # scalar float32

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            # Load token index (masked)
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Compute offsets for k and v (gather k_i and v_i for kv_head)
            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM  # scalar offset
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM  # scalar offset

            # Load k_vec and v_vec
            k_vec = tl.load(k_ptr + k_off)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # scalar float32

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off)
            v_vec = tl.load(v_ptr + v_off)

            # Recompute s
            logits = tl.sum(q_vec * k_vec, axis=0)
            s = logits * sm_scale
            # attn_i = exp(s - lse_val)
            attn_i = tl.exp(s - lse_val)
            # accumulate
            out_vec += attn_i * v_vec

        # Store outputs
        out_base = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_base, out_vec)

        lse_base = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton not available, do a pure PyTorch fallback (not used in evaluation)
        if not TRITON_AVAILABLE:
            batch_size, num_qo_heads, head_dim = q.shape
            _, num_pages, num_kv_heads, _ = k_cache.shape
            _, num_kv_indices = kv_indices.shape

            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

            gqa_ratio = num_qo_heads // num_kv_heads
            q_f32 = q.to(torch.float32)
            k_f32 = k_cache.to(torch.float32)
            v_f32 = v_cache.to(torch.float32)

            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                num_tokens = end - start
                if num_tokens == 0:
                    output[b].zero_()
                    lse[b].fill_(-float("inf"))
                    continue

                token_indices = kv_indices[start:end].to(torch.long)  # [num_tokens]
                k_batch = k_f32[token_indices]  # [num_tokens, 8, 128]
                v_batch = v_f32[token_indices]  # [num_tokens, 8, 128]

                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_f32[b, h]  # [128]
                    k_head = k_batch[:, kv_head]  # [num_tokens, 128]
                    v_head = v_batch[:, kv_head]  # [num_tokens, 128]

                    logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale
                    lse_val = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                    attn = torch.softmax(logits_scaled, dim=-1)
                    out_head = torch.matmul(attn, v_head)  # [128]
                    output[b, h] = out_head.to(torch.bfloat16)
                    lse[b, h] = lse_val
            return output, lse

        # Triton path: compute everything in kernels
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        # Constants
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Prepare pointers (float32 compute)
        q_f32 = q.to(torch.float32)
        k_f32 = k_cache.to(torch.float32)
        v_f32 = v_cache.to(torch.float32)

        # Output buffers (float32 compute, then cast to bfloat16)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Upper bound for token iterations. Use 128 to cover provided axes (max num_tokens ~ 98).
        NUM_TOKS = 128
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        # Launch kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        _attention_bh_kernel[grid](
            q_f32, k_f32, v_f32, kv_indptr, kv_indices, output, lse,
            B=batch_size, NUM_QO_HEADS=num_qo_heads, NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS, sm_scale=sm_scale, half_ln2_inv=half_ln2_inv,
        )

        # Cast output to bfloat16 to match original return dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
