import math
import torch

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
        ln2_inv: tl.constexpr,  # 1 / ln(2)
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
        lse = -float('inf')  # float32 scalar

        # Iterate tokens
        # Note: num_tokens and kv_start are scalars; we loop explicitly
        t = 0
        while t < num_tokens:
            idx = kv_start + t  # int32 scalar

            # Compute kv_head according to GQA: kv_head = h // (num_qo_heads // num_kv_heads)
            # Here num_qo_heads // num_kv_heads = 4
            kv_head = h // (num_qo_heads // num_kv_heads)

            # Compute offsets for k/v: [num_pages, 1, num_kv_heads, HEAD_DIM] -> index by (idx, kv_head, :)
            # k_ptr and v_ptr are contiguous across the last dim (HEAD_DIM)
            # Offset for k: idx * (1 * num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            # Simplify: idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            k_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_t and v_t as float32 (original may be bf16/f16; we convert here)
            # k_ptr/v_ptr are *bf16/*f16; load as f32 via tl.load(..., dtype=tl.float32)
            k_t = tl.load(k_ptr + k_offset, mask=True, other=0.0, dtype=tl.float32)  # [HEAD_DIM]
            v_t = tl.load(v_ptr + v_offset, mask=True, other=0.0, dtype=tl.float32)  # [HEAD_DIM]

            # Dot product: q_vec · k_t
            dot_sum = 0.0
            for j in range(HEAD_DIM):
                dot_sum += q_vec[j] * k_t[j]

            logits = dot_sum
            scaled = logits * sm_scale

            # Numerically stable LSE update
            # if lse == -inf: lse = scaled
            # else: lse = max(lse, scaled) + log(1 + exp(-abs(lse - scaled)))
            # Implement branchlessly using where
            # Note: Triton supports tl.maximum and tl.log, tl.exp
            new_lse = tl.maximum(lse, scaled) + tl.log(1.0 + tl.exp(-tl.abs(lse - scaled)))
            lse = new_lse

            # Attention weight
            attn = tl.exp(scaled - lse)

            # Accumulate output
            out_vec += attn * v_t

            t += 1

        # Store results
        # out_ptr is *f32 with shape [B, num_qo_heads, HEAD_DIM]
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        # Store lse / ln(2)
        lse_scaled = lse * ln2_inv
        tl.store(lse_ptr + b * num_qo_heads + h, lse_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, num_qo_heads, head_dim], dtype bfloat16
        k_cache: [num_pages, 1, num_kv_heads, head_dim], dtype bfloat16/f16
        v_cache: [num_pages, 1, num_kv_heads, head_dim], dtype bfloat16/f16
        kv_indptr: [B+1], int32, kv_indptr[0]=0, kv_indptr[B]=total_tokens
        kv_indices: int32, not used in computation (original run does not use it)
        sm_scale: float32 scalar
        Returns: (output [B, num_qo_heads, head_dim], bfloat16), lse [B, num_qo_heads], float32
        """
        assert q.dim() == 3, "q must be [B, num_qo_heads, head_dim]"
        assert k_cache.dim() == 4 and k_cache.shape[1] == 1, "k_cache must be [num_pages, 1, num_kv_heads, head_dim]"
        assert v_cache.dim() == 4 and v_cache.shape[1] == 1, "v_cache must be [num_pages, 1, num_kv_heads, head_dim]"
        assert kv_indptr.dim() == 1, "kv_indptr must be 1D"
        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape

        # Ensure contiguity and device consistency
        device = q.device
        if TRITON_AVAILABLE and torch.cuda.is_available():
            # Move to CUDA for Triton kernel
            q_f32 = q.to(torch.float32).contiguous()
            k_cache_c = k_cache.contiguous()
            v_cache_c = v_cache.contiguous()
            kv_indptr_c = kv_indptr.contiguous()

            # Allocate outputs (float32 compute, bfloat16 output)
            output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
            lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

            # Launch Triton kernel: one program per (b, h)
            grid = (batch_size * num_qo_heads,)
            _gqa_attention_kernel[grid](
                q_f32,
                k_cache_c, v_cache_c,
                kv_indptr_c,
                output,
                lse,
                batch_size,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                HEAD_DIM=head_dim,
                sm_scale=float(sm_scale),
                ln2_inv=1.0 / math.log(2.0),
                num_warps=4,
                num_stages=2,
            )

            # Cast output to bfloat16 to match original
            output_bf16 = output.to(torch.bfloat16)
            return output_bf16, lse
        else:
            # CPU fallback (not used in evaluation, but provided for robustness)
            output = torch.zeros(
                (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
            )
            lse = torch.full(
                (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
            )
            for b in range(batch_size):
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())
                num_tokens = kv_end - kv_start
                if num_tokens <= 0:
                    continue
                q_b = q[b].to(torch.float32)  # [num_qo_heads, head_dim]
                for h in range(num_qo_heads):
                    kv_head = h // (num_qo_heads // num_kv_heads)
                    q_vec = q_b[h]  # [head_dim], float32
                    out_vec = torch.zeros((head_dim,), dtype=torch.float32, device=device)
                    lse_val = float("-inf")
                    for t in range(num_tokens):
                        idx = kv_start + t
                        k_t = k_cache[idx, 0, kv_head].to(torch.float32)  # [head_dim]
                        v_t = v_cache[idx, 0, kv_head].to(torch.float32)  # [head_dim]
                        logits = torch.dot(q_vec, k_t)  # float32
                        scaled = logits * sm_scale
                        new_lse = torch.log1p(torch.exp(scaled - lse_val)) + lse_val  # simple update
                        lse_val = new_lse
                        attn = torch.exp(scaled - lse_val)
                        out_vec += attn * v_t
                    output[b, h] = out_vec.to(torch.bfloat16)
                    lse[b, h] = (lse_val / math.log(2.0)).to(torch.float32)
            return output, lse


def run(*args):
    return ModelNew()(*args)
