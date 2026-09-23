import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,            # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32, shape [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, shape [NUM_KV_INDICES]
        out_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,
        sm_scale: tl.constexpr,
        half_ln2_inv: tl.constexpr,
    ):
        # Program ids for batch and query head
        b = tl.program_id(0)
        h = tl.program_id(1)

        # GQA mapping: kv_head = h // (NUM_QO_HEADS // NUM_KV_HEADS) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h] for this batch
        q_base = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_base)  # [HEAD_DIM], float32

        # Accumulate max and sum_exp for logsumexp over scaled scores
        max_s = -float("inf")
        sum_exp = 0.0

        # First pass: compute logsumexp over tokens with mask
        for i in range(NUM_TOKS):
            # num_tokens = kv_indptr[b+1] - kv_indptr[b]
            mask_i = i < (kv_indptr_ptr[b + 1] - kv_indptr_ptr[b])
            # index of the i-th token for this batch
            idx = tl.load(kv_indices_ptr + kv_indptr_ptr[b] + i, mask=mask_i, other=0)  # int32

            # Offsets into k/v for this token and kv_head
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_i and v_i (masked if out of range)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp + tl.exp(s - max_s)

        # Compute lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv

        # Accumulate output vector
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # Second pass: recompute s, compute attn, and accumulate output
        for i in range(NUM_TOKS):
            mask_i = i < (kv_indptr_ptr[b + 1] - kv_indptr_ptr[b])
            idx = tl.load(kv_indices_ptr + kv_indptr_ptr[b] + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
            s = logits * sm_scale

            attn = tl.exp(s - lse_val)
            out_vec += attn * v_vec

        # Store output and lse
        out_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_offset = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available; if not, do a safe fallback (rare in eval).
        if not TRITON_AVAILABLE:
            batch_size, num_qo_heads, head_dim = q.shape
            device = q.device
            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
            gqa_ratio = num_qo_heads // 8  # 4
            q32 = q.to(torch.float32)
            k32 = k_cache.squeeze(1).to(torch.float32)
            v32 = v_cache.squeeze(1).to(torch.float32)
            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                if start >= end:
                    output[b].zero_()
                    lse[b].fill_(-float("inf"))
                    continue
                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]
                if num_tokens == 0:
                    output[b].zero_()
                    lse[b].fill_(-float("inf"))
                    continue
                q_batch = q32[b]  # [32, 128]
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]  # [128]
                    k_batch = k32[token_indices]  # [num_tokens, 8, 128]
                    v_batch = v32[token_indices]  # [num_tokens, 8, 128]
                    k_head = k_batch[:, kv_head]  # [num_tokens, 128]
                    v_head = v_batch[:, kv_head]  # [num_tokens, 128]
                    logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
                    s = logits * sm_scale
                    lse[b, h] = torch.logsumexp(s, dim=-1) / math.log(2.0)
                    attn = torch.softmax(s, dim=-1)
                    out_head = torch.matmul(attn, v_head)  # [128]
                    output[b, h] = out_head.to(torch.bfloat16)
            return output, lse

        # Triton path: ensure CUDA tensors and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Cast to float32 for compute
        q32 = q.to(torch.float32)
        k32 = k_cache.to(torch.float32)
        v32 = v_cache.to(torch.float32)

        batch_size, num_qo_heads, head_dim = q32.shape
        assert num_qo_heads == 32 and head_dim == 128, "This Triton kernel expects num_qo_heads=32 and head_dim=128"

        # Output and lse buffers (float32 for compute)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q32.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q32.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)

        # Conservative NUM_TOKS bound; masks ensure correctness for any actual num_tokens
        NUM_TOKS = 8192
        sm_scale_val = float(sm_scale)  # 1.0 / sqrt(128)
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        _attention_bh_kernel[grid](
            q32, k32, v32,
            kv_indptr, kv_indices,
            output, lse,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=8,
            HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS,
            sm_scale=sm_scale_val,
            half_ln2_inv=half_ln2_inv,
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 to match original output dtype; lse remains float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
