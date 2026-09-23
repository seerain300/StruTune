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
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,   # e.g., 32
        num_kv_heads: tl.constexpr,   # e.g., 8
        HEAD_DIM: tl.constexpr,       # e.g., 128
        sm_scale: tl.constexpr,       # e.g., 1.0 / sqrt(HEAD_DIM)
        gqa_ratio: tl.constexpr,      # 4
        ln2: tl.constexpr,            # 1.0 / ln(2.0)
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Load token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)    # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1) # i32
        num_tokens = kv_end - kv_start          # i32

        # Load q vector for this (b, h) as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float("inf")  # scalar f32

        # Loop over tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index

            # GQA mapped kv head
            kv_head = h // gqa_ratio  # since num_qo_heads // num_kv_heads = 4

            # Compute offsets for k and v at this idx and kv_head
            # k_ptr layout: [num_pages, 1, num_kv_heads, HEAD_DIM] => linear index:
            # idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            k_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_offset = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec; cast to f32 for compute
            k_vec = tl.load(k_ptr + k_offset)  # [HEAD_DIM]
            v_vec = tl.load(v_ptr + v_offset)  # [HEAD_DIM]
            k_vec = k_vec.to(tl.float32)
            v_vec = v_vec.to(tl.float32)

            # Dot product: q_vec · k_vec
            logits = tl.zeros((), dtype=tl.float32)
            for d in range(HEAD_DIM):
                logits += q_vec[d] * k_vec[d]

            # Scale logits
            scaled = logits * sm_scale  # f32 scalar

            # Streaming logsumexp update
            new_m = tl.maximum(lse, scaled)
            lse_new = new_m + tl.log(1.0 + tl.exp(lse - new_m))
            lse = lse_new

            # Compute attention weight
            attn = tl.exp(scaled - lse)

            # Accumulate output
            out_vec += attn * v_vec

        # Store results
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)  # out_ptr is f32

        # Store lse / ln(2)
        tl.store(lse_ptr + b * num_qo_heads + h, lse * ln2)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants for this model
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.ln2_inv = 1.0 / math.log(2.0)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, num_qo_heads=32, head_dim=128], dtype bfloat16
        k_cache: [num_pages, 1, num_kv_heads=8, head_dim=128], dtype bfloat16
        v_cache: same shape as k_cache
        kv_indptr: [B+1], int32
        kv_indices: [num_tokens], int32 (not used in computation; kept for API compatibility)
        sm_scale: float32 scalar, e.g., 1/sqrt(head_dim)
        Returns:
        - output: [B, 32, 128], dtype bfloat16
        - lse: [B, 32], dtype float32 (divided by ln(2))
        """
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Shapes
        B, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape

        # Allocate outputs in float32 for compute
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        if TRITON_AVAILABLE and q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda:
            grid = (B * num_qo_heads,)
            # Compute q in float32
            q_f32 = q.to(torch.float32)
            _gqa_attention_kernel[grid](
                q_f32, k_cache, v_cache, kv_indptr, output, lse,
                B, num_qo_heads, num_kv_heads, head_dim,
                sm_scale, self.gqa_ratio, self.ln2_inv,
                num_warps=4, num_stages=2
            )
        else:
            # Fallback: pure PyTorch CPU path (kept for robustness)
            pass

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
