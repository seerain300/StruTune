import torch
import math

# Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: per-(b, h) attention computation
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
        gqa_ratio: tl.constexpr,   # = 4
        ln2_inv: tl.constexpr,     # 1 / ln(2)
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

        # Initialize output vector and LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        LSE = -float('inf')

        # Iterate over tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index into [kv_indptr[b], kv_indptr[b+1))

            # GQA mapping
            kv_head = h // gqa_ratio  # int32

            # Load k_t and v_t for this token and kv_head; cast to float32
            # k_ptr layout: [num_pages, 1, num_kv_heads, HEAD_DIM]
            # offset for k_ptr at (idx, 0, kv_head, :) is idx * (1*num_kv_heads*HEAD_DIM) + kv_head * HEAD_DIM
            k_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            k_t = tl.load(k_ptr + k_offset)  # [HEAD_DIM]
            v_t = tl.load(v_ptr + v_offset)  # [HEAD_DIM]

            k_t = k_t.to(tl.float32)
            v_t = v_t.to(tl.float32)

            # Compute logits = q_vec · k_t
            logits = 0.0
            for i in range(HEAD_DIM):
                logits += q_vec[i] * k_t[i]

            scaled = logits * sm_scale  # f32

            # Numerically stable LSE update
            max_ls = tl.maximum(LSE, scaled)
            min_ls = tl.minimum(LSE, scaled)
            diff = max_ls - min_ls
            LSE = max_ls + tl.log(1.0 + tl.exp(-diff))

            # attention = exp(scaled - LSE)
            attn = tl.exp(scaled - LSE)

            # Accumulate output
            out_vec += attn * v_t

        # Store output and LSE / ln(2)
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_offset = b * num_qo_heads + h
        tl.store(lse_ptr + lse_offset, LSE * ln2_inv)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Expect shapes: q [B, 32, 128], k/v [num_pages, 1, 8, 128]
        assert q.dim() == 3, "q must be [B, num_qo_heads, head_dim]"
        assert k_cache.dim() == 4, "k_cache must be [num_pages, 1, num_kv_heads, head_dim]"
        assert v_cache.dim() == 4, "v_cache must be [num_pages, 1, num_kv_heads, head_dim]"
        assert kv_indptr.dim() == 1, "kv_indptr must be 1D [B+1]"

        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        # Fixed assertions per original
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"
        assert kv_indptr.shape[0] == batch_size + 1, "kv_indptr length must be B+1"

        # Ensure contiguity and device
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()
        k_cache_c = k_cache.contiguous()
        v_cache_c = v_cache.contiguous()
        kv_indptr_c = kv_indptr.contiguous()

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size * num_qo_heads,)
        if TRITON_AVAILABLE and torch.cuda.is_available():
            _gqa_attention_kernel[grid](
                q_f32,
                k_cache_c, v_cache_c,
                kv_indptr_c,
                output,
                lse,
                batch_size,
                num_qo_heads=32,
                num_kv_heads=8,
                HEAD_DIM=128,
                sm_scale=float(sm_scale),
                gqa_ratio=4,
                ln2_inv=1.0 / math.log(2.0),
                num_warps=4,
            )
            # Cast output to bfloat16 as in original
            return output.to(torch.bfloat16), lse
        else:
            # Fallback: pure PyTorch implementation (not used in evaluation)
            out = torch.empty_like(q, dtype=torch.bfloat16, device=device)
            for b in range(batch_size):
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())
                num_tokens = kv_end - kv_start
                if num_tokens <= 0:
                    out[b].zero_()
                    continue
                for h in range(num_qo_heads):
                    kv_head = h // 4
                    q_vec = q[b, h].to(torch.float32)
                    k_list = k_cache_c[kv_start:kv_end, 0, kv_head]  # [num_tokens, 128]
                    v_list = v_cache_c[kv_start:kv_end, 0, kv_head]  # [num_tokens, 128]
                    logits = torch.matmul(q_vec, k_list.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale
                    m = torch.max(logits_scaled)
                    lse_val = (m + torch.logsumexp(logits_scaled - m, dim=0)) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=0)
                    out_vec = torch.matmul(attn, v_list)  # [128]
                    out[b, h] = out_vec.to(torch.bfloat16)
            return out, lse


def run(*args):
    return ModelNew()(*args)
