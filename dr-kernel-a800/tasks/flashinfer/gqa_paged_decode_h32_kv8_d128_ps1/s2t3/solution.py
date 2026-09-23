import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h).
# It loops over up to NUM_TOKS tokens with masks, computing:
# - lse = logsumexp(s) / ln(2), where s = (q[h] @ k_i) * sm_scale
# - output vector out = sum_i softmax(s_i) * v_i
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
        batch_size,       # int32
        NUM_TOKS: tl.constexpr,   # upper bound on num tokens per batch (e.g., 8192)
        HEAD_DIM: tl.constexpr,   # e.g., 128
        NUM_QO_HEADS: tl.constexpr,  # e.g., 32
        NUM_KV_HEADS: tl.constexpr,  # e.g., 8
        sm_scale,         # float32 scalar
        half_ln2_inv,     # float32 scalar, equals 1 / ln(2)
    ):
        # program ids
        b = tl.program_id(0)  # batch id
        h = tl.program_id(1)  # query head id

        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h] as vector
        q_off = h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Load start and end of this batch's token range
        start = tl.load(kv_indptr_ptr + b)             # int32
        end = tl.load(kv_indptr_ptr + b + 1)          # int32
        num_tokens_actual = end - start               # int32

        # Pass 1: accumulate max and sum(exp(s - max)) across tokens
        max_s = -float("inf")
        sum_exp = 0.0  # scalar float32

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            # Load token index (masked)
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows for this idx and kv_head
            # k_ptr layout: [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM], contiguous:
            # element (pg, head, dim) offset = pg * (NUM_KV_HEADS * HEAD_DIM) + head * HEAD_DIM + dim
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
                sum_exp += tl.exp(s - max_s)

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
            s = logits * sm_scale

            if mask_i:
                attn = tl.exp(s - lse_val)  # softmax over s
                out_vec += attn * v_vec

        # Store output
        out_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_off, out_vec)

        # Store lse
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton available; otherwise fallback to PyTorch (kept for safety)
        if not TRITON_AVAILABLE:
            batch_size, num_qo_heads, head_dim = q.shape
            _, num_pages, num_kv_heads, _ = k_cache.shape
            assert num_qo_heads == 32
            assert num_kv_heads == 8
            assert head_dim == 128
            device = q.device
            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
            gqa_ratio = num_qo_heads // num_kv_heads
            k_cache_flat = k_cache.squeeze(1).to(torch.float32)
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)
            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]
                if num_tokens == 0:
                    continue
                q_b = q[b].to(torch.float32)  # [32, 128]
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_h = q_b[h]  # [128]
                    k_heads = k_cache_flat[token_indices, kv_head]  # [num_tokens, 128]
                    v_heads = v_cache_flat[token_indices, kv_head]  # [num_tokens, 128]
                    logits = torch.matmul(q_h, k_heads.T)  # [num_tokens]
                    s = logits * sm_scale
                    lse_b = torch.logsumexp(s, dim=0) / math.log(2.0)
                    attn = torch.softmax(s, dim=0)  # [num_tokens]
                    out_h = torch.matmul(attn, v_heads)  # [128]
                    output[b, h] = out_h.to(torch.bfloat16)
                    lse[b, h] = lse_b
            return output, lse

        # Triton path: ensure inputs on CUDA and contiguous
        device = q.device
        assert device.type == "cuda", "This Triton implementation requires CUDA tensors."
        q_f32 = q.to(torch.float32).contiguous()           # [B, 32, 128]
        k_f32 = k_cache.to(torch.float32).contiguous()     # [N, 1, 8, 128] -> squeezed below
        v_f32 = v_cache.to(torch.float32).contiguous()     # [N, 1, 8, 128] -> squeezed below
        # kv_indptr, kv_indices are int32
        kv_indptr_i32 = kv_indptr.to(torch.int32).contiguous()
        kv_indices_i32 = kv_indices.to(torch.int32).contiguous()

        # Extract shapes
        batch_size = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        assert num_qo_heads == 32
        # Squeeze cached tensors to [N, 8, 128]
        k_f32 = k_f32.squeeze(1)
        v_f32 = v_f32.squeeze(1)
        num_kv_heads = k_f32.shape[1]
        assert num_kv_heads == 8
        num_pages = k_f32.shape[0]
        assert head_dim == 128

        # Allocate outputs (float32 for compute; we cast to bfloat16 at the end)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch one program per (b, h)
        grid = (batch_size, num_qo_heads)
        NUM_TOKS = 8192  # upper bound; masks handle actual num_tokens
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2) as float32

        _attention_bh_kernel[grid](
            q_f32, k_f32, v_f32,
            kv_indptr_i32, kv_indices_i32,
            output, lse,
            batch_size,
            NUM_TOKS=NUM_TOKS,
            HEAD_DIM=head_dim,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            sm_scale=sm_scale,
            half_ln2_inv=half_ln2_inv,
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 as required by original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
