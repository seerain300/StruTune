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
        ln2_inv: tl.constexpr,   # 1 / ln(2)
        GQA_RATIO: tl.constexpr
    ):
        # One program per (batch, head)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B:
            return

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)   # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1) # int32
        num_tokens = kv_end - kv_start          # int32 scalar in kernel

        # Initialize accumulators
        lse_acc = tl.full((), -float("inf"), tl.float32)
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        # Load q vector for this head (float32)
        q_off = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM] f32

        # Loop over tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index into cache
            kv_head = h // GQA_RATIO  # GQA mapping

            # Compute linear offsets for k_t and v_t
            # k_ptr layout: [num_pages, 1, num_kv_heads, HEAD_DIM] contiguous
            base_k = idx * (num_kv_heads * HEAD_DIM)
            k_off = base_k + kv_head * HEAD_DIM
            base_v = idx * (num_kv_heads * HEAD_DIM)
            v_off = base_v + kv_head * HEAD_DIM

            # Load k_t, v_t (original dtype), cast to f32 for compute
            k_t_raw = tl.load(k_ptr + k_off)  # [HEAD_DIM]
            v_t_raw = tl.load(v_ptr + v_off)  # [HEAD_DIM]
            k_t = k_t_raw.to(tl.float32)
            v_t = v_t_raw.to(tl.float32)

            # Compute logits = q_vec @ k_t
            logits = tl.sum(q_vec[:, None] * k_t[None, :], axis=0)  # scalar f32

            # Scale and update LSE stably
            s = logits * sm_scale
            m = tl.maximum(lse_acc, s)
            new_lse = m + tl.log(1.0 + tl.exp(-tl.abs(lse_acc - s)))
            lse_acc = new_lse

            # Attention weight
            attn = tl.exp(s - lse_acc)  # scalar f32

            # Accumulate output
            out_vec += attn * v_t

        # Store output vector for (b, h)
        out_off = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_off, out_vec)

        # Store LSE divided by ln(2)
        lse_off = b * num_qo_heads + h
        tl.store(lse_ptr + lse_off, lse_acc * ln2_inv)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton unavailable or tensors not CUDA, fallback to original PyTorch path
        if not TRITON_AVAILABLE or not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda):
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            gqa_ratio = num_qo_heads // num_kv_heads

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
                    continue

                q_batch = q[b].to(torch.float32)
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]  # [head_dim]
                    # Gather k_list and v_list using kv_indptr range
                    k_list = k_cache[kv_start:kv_end, 0, kv_head]  # [num_tokens, head_dim]
                    v_list = v_cache[kv_start:kv_end, 0, kv_head]  # [num_tokens, head_dim]
                    logits = torch.matmul(q_head, k_list.T)        # [num_tokens]
                    logits_scaled = logits * sm_scale
                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=-1)     # [num_tokens]
                    out_head = torch.matmul(attn, v_list)          # [head_dim]
                    output[b, h] = out_head.to(torch.bfloat16)
            return output, lse

        # Triton path
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        GQA_RATIO = num_qo_heads // num_kv_heads
        ln2_inv = 1.0 / math.log(2.0)

        # Ensure contiguity
        q_f32 = q.to(torch.float32).contiguous()  # compute q in f32
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Allocate outputs (compute


def run(*args):
    return ModelNew()(*args)
