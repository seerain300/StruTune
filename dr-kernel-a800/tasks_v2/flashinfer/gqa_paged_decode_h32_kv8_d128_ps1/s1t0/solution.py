import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _attention_single_head_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM], but we load per b,h
        k_ptr, v_ptr,    # *f16 or *bf16, shape [num_pages, 1, num_kv_heads, HEAD_DIM] (we will index by token_indices)
        kv_indices_ptr,  # *i32, shape [num_kv_indices]
        kv_indptr_ptr,   # *i32, shape [len_indptr] (used to compute num_tokens per batch)
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        b,               # int32, current batch index
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        sm_scale: tl.constexpr,
        kv_start,        # int32
        kv_end,          # int32
        GQA_RATIO: tl.constexpr
    ):
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        # safety
        if pid >= b * num_qo_heads:
            return

        # Load q vector for this (b, h) and cast to f32 for compute
        q_offset = b * (num_qo_heads * head_dim) + h * head_dim
        q_vec = tl.load(q_ptr + q_offset + tl.arange(0, head_dim))
        q_vec = q_vec.to(tl.float32)  # [HEAD_DIM] f32

        kv_head = h // GQA_RATIO

        # Initialize accumulators
        lse_acc = -float("inf")
        out_vec = tl.zeros((head_dim,), dtype=tl.float32)

        # Iterate over tokens in [kv_start, kv_end)
        # We assume num_tokens = kv_end - kv_start
        for t in range(0, kv_end - kv_start):
            idx = kv_start + t  # linear token index
            # Gather k and v for this token and kv_head; cast to f32
            k_t_offset = idx * (num_kv_heads * head_dim) + kv_head * head_dim
            k_t = tl.load(k_ptr + k_t_offset + tl.arange(0, head_dim)).to(tl.float32)
            v_t = tl.load(v_ptr + k_t_offset + tl.arange(0, head_dim)).to(tl.float32)

            # logits = q_vec @ k_t  (dot product)
            logits = tl.sum(q_vec * k_t, axis=0)  # scalar
            s_t = logits * sm_scale

            # Streaming LSE update
            m = tl.maximum(lse_acc, s_t)
            sum_exp = tl.exp(lse_acc - m) + tl.exp(s_t - m)
            lse_acc = m + tl.log(sum_exp)

            # attention weight
            attn_t = tl.exp(s_t - lse_acc)  # scalar

            # accumulate output
            out_vec += attn_t * v_t

        # Store results
        out_offset = b * (num_qo_heads * head_dim) + h * head_dim
        tl.store(out_ptr + out_offset + tl.arange(0, head_dim), out_vec)
        tl.store(lse_ptr + b * num_qo_heads + h, lse_acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Fallback to original PyTorch if Triton not available
        if not TRITON_AVAILABLE:
            # Same behavior as the original Model.forward
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            device = q.device
            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
            gqa_ratio = num_qo_heads // num_kv_heads
            k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
            for b in range(batch_size):
                page_start = int(kv_indptr[b].item())
                page_end = int(kv_indptr[b + 1].item())
                token_indices = kv_indices[page_start:page_end].to(torch.long)
                num_tokens = token_indices.shape[0]
                if num_tokens == 0:
                    output[b].zero_()
                    continue
                q_batch = q[b].to(torch.float32)
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]  # [head_dim]
                    k_head = k_cache_flat[token_indices, kv_head]  # [num_tokens, head_dim]
                    v_head = v_cache_flat[token_indices, kv_head]  # [num_tokens, head_dim]
                    logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale
                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
                    out_head = torch.matmul(attn, v_head)  # [head_dim]
                    output[b, h] = out_head.to(torch.bfloat16)
            return output, lse

        # Triton path
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape

        # Ensure inputs are contiguous
        q_f32 = q.to(torch.float32).contiguous()
        # Do NOT pre-copy k/v; we will gather per token in the kernel
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Outputs
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Grid: one program per (b, h)
        grid = (batch_size * num_qo_heads,)

        # GQA ratio is assumed 4 (32/8); constants for kernel
        GQA_RATIO = num_qo_heads // num_kv_heads

        # For each batch, compute num_tokens = kv_indptr[b+1] - kv_indptr[b]
        for b in range(batch_size):
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_tokens = kv_end - kv_start

            # If no tokens, zero output and skip
            if num_tokens <= 0:
                # Keep output zeros by not launching; or set zeros explicitly
                continue

            _attention_single_head_kernel[grid](
                q_f32, k_cache, v_cache, kv_indices, kv_indptr,
                output, lse,
                b,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                sm_scale=sm_scale,
                kv_start=kv_start,
                kv_end=kv_end,
                GQA_RATIO=GQA_RATIO,
                num_warps=1,  # small problem size; 1 warp is fine
                num_stages=1
            )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)

        # Match original: divide LSE by log(2)
        lse_div = lse / math.log(2.0)

        return output_bf16, lse_div


def run(*args):
    return ModelNew()(*args)
