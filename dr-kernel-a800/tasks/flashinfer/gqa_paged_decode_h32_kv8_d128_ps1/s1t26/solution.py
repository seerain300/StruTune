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
        k_ptr, v_ptr,    # *bf16 or *f16 or *f32, shape [num_pages, 1, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2_inv: tl.constexpr,  # 1 / ln(2) as float
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
        lse = -float('inf')  # f32 scalar

        # Process tokens in [kv_start, kv_end)
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index into k/v
            kv_head = h // gqa_ratio

            # Compute k_t and v_t as float32 (cast from original dtype in memory)
            # Memory layout: [num_pages, 1, num_kv_heads, HEAD_DIM] contiguous
            # k_ptr stride for num_kv_heads and HEAD_DIM is 1 * num_kv_heads * HEAD_DIM + kv_head * HEAD_DIM
            k_off = idx * (1 * num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (1 * num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            k_t = tl.load(k_ptr + k_off)  # [HEAD_DIM]
            v_t = tl.load(v_ptr + v_off)  # [HEAD_DIM]
            k_t = k_t.to(tl.float32)
            v_t = v_t.to(tl.float32)

            # Dot product
            logits = 0.0
            for d in range(HEAD_DIM):
                logits += q_vec[d] * k_t[d]
            scaled = logits * sm_scale

            # Numerically stable LSE update:
            # new_LSE = max(lse, scaled) + log(1 + exp(-abs(lse - scaled)))
            max_ls_scaled = tl.maximum(lse, scaled)
            diff = lse - scaled
            new_lse = max_ls_scaled + tl.log(1.0 + tl.exp(-tl.abs(diff)))
            lse = new_lse

            attn = tl.exp(scaled - lse)  # softmax weight
            out_vec += attn * v_t

        # Store results
        out_offset = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)
        tl.store(lse_ptr + b * num_qo_heads + h, lse * ln2_inv)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Type and shape checks consistent with original Model.run
        assert q.ndim == 3 and k_cache.ndim == 4 and v_cache.ndim == 4
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        assert kv_indptr.shape[0] == batch_size + 1
        # kv_indices is not used in the original run; just validate its numel
        assert kv_indices.numel() == kv_indptr[-1].item()

        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Triton execution: move to CUDA if available; otherwise CPU fallback
        use_triton = TRITON_AVAILABLE and torch.cuda.is_available()
        if use_triton:
            # Compute q in float32 for kernel; k/v in original dtype (kernel casts as needed)
            q_gpu = q.to(torch.float32).contiguous()
            k_gpu = k_cache  # keep original dtype for in-kernel casting
            v_gpu = v_cache  # keep original dtype for in-kernel casting

            output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q_gpu.device)
            lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q_gpu.device)

            # Launch Triton kernel: one program per (b, h)
            grid = (batch_size * num_qo_heads,)
            _gqa_attention_kernel[grid](
                q_gpu,
                k_gpu, v_gpu,
                kv_indptr,
                output,
                lse,
                batch_size,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                HEAD_DIM=head_dim,
                sm_scale=float(sm_scale),
                gqa_ratio=4,  # 32 // 8
                ln2_inv=1.0 / math.log(2.0),
            )

            # Cast output to bfloat16 as in original and return lse / ln(2) (already computed in kernel)
            output_bf16 = output.to(torch.bfloat16)
            return output_bf16, lse
        else:
            # CPU fallback: pure PyTorch implementation mirroring original logic
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
                    continue

                q_batch = q[b].to(torch.float32)
                for h in range(num_qo_heads):
                    kv_head = h // 4
                    q_head = q_batch[h]  # [head_dim]
                    # Gather k_list and v_list from caches
                    k_list = k_cache[kv_start:kv_end, 0, kv_head].to(torch.float32)  # [num_tokens, head_dim]
                    v_list = v_cache[kv_start:kv_end, 0, kv_head].to(torch.float32)  # [num_tokens, head_dim]
                    logits = torch.matmul(q_head, k_list.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale
                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
                    out_head = torch.matmul(attn, v_list)        # [head_dim]
                    output[b, h] = out_head.to(torch.bfloat16)

            return output, lse


def run(*args):
    return ModelNew()(*args)
