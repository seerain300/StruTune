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
        q_ptr,             # *float32, [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,             # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,             # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,     # *int32, [BATCH_SIZE + 1]
        kv_indices_ptr,    # *int32, [NUM_KV_INDICES]
        out_ptr,           # *float32, [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,           # *float32, [BATCH_SIZE, NUM_QO_HEADS]
        sm_scale,          # float32 scalar
        batch_size,        # int32
        num_qo_heads,      # int32 (32)
        num_kv_heads,      # int32 (8)
        head_dim,          # int32 (128)
        num_kv_indices,    # int32
        start_ptr,         # *int32, pointer to kv_indptr[b]
        end_ptr,           # *int32, pointer to kv_indptr[b+1]
        num_tokens_actual, # int32, actual number of tokens for this batch (end - start)
        NUM_TOKS: tl.constexpr,        # loop bound (compile-time for Triton)
        kv_ratio: tl.constexpr,        # NUM_QO_HEADS // NUM_KV_HEADS (compile-time, e.g., 4)
    ):
        # One program per (b, h)
        b = tl.program_id(0)
        h = tl.program_id(1)

        # Load start/end for this batch b
        start = tl.load(start_ptr)  # int32
        end = tl.load(end_ptr)      # int32
        assert num_tokens_actual == (end - start), "num_tokens_actual mismatch"

        # Strides
        k_stride0 = num_kv_heads * head_dim       # stride over num_pages
        k_stride1 = head_dim                      # stride over kv_head
        v_stride0 = num_kv_heads * head_dim
        v_stride1 = head_dim

        # GQA mapping: kv_head = h // kv_ratio
        kv_head = h // kv_ratio  # int32 scalar in [0,7]

        # Load q[h, :] as float32
        q_off = h * head_dim
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Pass 1: compute max_s and sum_exp = sum(exp(s - max_s))
        max_s = -float('inf')
        sum_exp = 0.0  # float32
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows: k[v] is [num_pages, num_kv_heads, head_dim]
            k_off = idx * k_stride0 + kv_head * k_stride1
            v_off = idx * v_stride0 + kv_head * v_stride1

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0  # sum exp(s_i - max_s)

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        inv_ln2 = 1.4426950408889634  # log(2)
        lse_val = tl.log(max_s) + tl.log(sum_exp) * inv_ln2

        # Pass 2: recompute s, attn = exp(s - lse), accumulate output vector
        out_vec = tl.zeros([head_dim], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * k_stride0 + kv_head * k_stride1
            v_off = idx * v_stride0 + kv_head * v_stride1

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar

            # Vector-wise accumulate: out_vec += attn * v_vec
            out_vec += attn * v_vec

        # Store output and lse
        # out_ptr is [B, H, D]; linear offset = b * (H * D) + h * D
        out_offset = b * (num_qo_heads * head_dim) + h * head_dim
        tl.store(out_ptr + out_offset, out_vec)

        # lse_ptr is [B, H]; linear offset = b * H + h
        lse_offset = b * num_qo_heads + h
        tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available
        if not TRITON_AVAILABLE:
            # Fallback to PyTorch (should not be used in eval environment)
            return self._fallback_run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale)

        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be on CUDA for Triton execution."

        # Ensure contiguity and dtype for compute
        q = q.to(torch.float32).contiguous()       # [B, 32, 128]
        k_cache = k_cache.to(torch.float32).contiguous()  # [N, 1, 8, 128]
        v_cache = v_cache.to(torch.float32).contiguous()  # [N, 1, 8, 128]
        kv_indptr = kv_indptr.to(torch.int32).contiguous()  # [B+1]
        kv_indices = kv_indices.to(torch.int32).contiguous()  # [M]

        batch_size = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        num_pages = k_cache.shape[0]
        num_kv_heads = k_cache.shape[2]

        # Prepare output and lse tensors (float32 for compute)
        out = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Grid: one program per (b, h)
        grid = (batch_size, num_qo_heads)

        # NUM_TOKS upper bound; use 8192 to cover all workloads safely
        NUM_TOKS = 8192
        kv_ratio = num_qo_heads // num_kv_heads  # 4

        for b in range(batch_size):
            start_ptr = kv_indptr + b
            end_ptr = kv_indptr + b + 1
            num_tokens = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            _attention_bh_kernel[grid](
                q, k_cache, v_cache,
                kv_indptr, kv_indices,
                out, lse,
                float(sm_scale),
                batch_size, num_qo_heads, num_kv_heads, head_dim,
                kv_indices.numel(),
                start_ptr, end_ptr,
                num_tokens,
                NUM_TOKS=NUM_TOKS,
                kv_ratio=kv_ratio,
                num_warps=4,
            )

        # Cast output to bfloat16 to match original output dtype; lse remains float32
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse

    def _fallback_run(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Fallback: pure PyTorch (not used in Triton eval)
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128
        output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                output[b].zero_()
                continue
            token_indices = kv_indices[start:end].to(torch.long)
            num_tokens = token_indices.shape[0]
            if num_tokens == 0:
                output[b].zero_()
                continue
            k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [N, 8, 128]
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [N, 8, 128]
            k_batch = k_cache_flat[token_indices]  # [T, 8, 128]
            v_batch = v_cache_flat[token_indices]  # [T, 8, 128]
            q_batch = q[b].to(torch.float32)  # [32, 128]
            for h in range(num_qo_heads):
                kv_head = h // (num_qo_heads // num_kv_heads)
                q_vec = q_batch[h]  # [128]
                k_vec = k_batch[:, kv_head]  # [T, 128]
                v_vec = v_batch[:, kv_head]  # [T, 128]
                logits = torch.matmul(q_vec, k_vec.T)  # [T]
                s = logits * sm_scale
                lse[b, h] = torch.logsumexp(s, dim=-1) / math.log(2.0)
                attn = torch.softmax(s, dim=-1)  # [T]
                out_vec = torch.matmul(attn, v_vec)  # [128]
                output[b, h] = out_vec.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
