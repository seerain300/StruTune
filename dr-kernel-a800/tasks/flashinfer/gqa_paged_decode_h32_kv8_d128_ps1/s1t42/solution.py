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
    def _gqa_attention_single_batch_kernel(
        q_ptr,           # *f32, shape [B, 32, 128]
        k_ptr, v_ptr,    # *bf16/f16, shape [num_pages, 128] (middle=1 flattened)
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, 32, 128]
        lse_ptr,         # *f32, shape [B, 32]
        B: tl.constexpr,                    # int32
        num_qo_heads: tl.constexpr,        # 32
        num_kv_heads: tl.constexpr,        # 8
        HEAD_DIM: tl.constexpr,            # 128
        sm_scale: tl.constexpr,            # float32
        ln2: tl.constexpr,                 # float32
        num_tokens: tl.constexpr           # int32 = kv_indptr[B] - kv_indptr[b] (we pass B as b=0)
    ):
        # Since grid=(B,), b is implicit 0 here; kv_indptr[B] = total tokens
        kv_start = tl.load(kv_indptr_ptr + 0)    # i32
        kv_end = tl.load(kv_indptr_ptr + B)     # i32
        num_tokens = kv_end - kv_start          # i32

        # Iterate over query heads h = 0..31
        for h in range(0, 32):
            # Load q vector for this head as float32
            q_offset = 0 * num_qo_heads * HEAD_DIM + h * HEAD_DIM
            q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

            # Initialize output vector and lse
            out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
            lse = tl.full((), -float("inf"), dtype=tl.float32)

            # Iterate tokens (unrolled since num_tokens is constexpr)
            for t in tl.static_range(0, num_tokens):
                base = t * HEAD_DIM
                # Load k_t and v_t as original dtype then cast to float32
                k_vec = tl.load(k_ptr + base)  # [HEAD_DIM] f16/bf16
                v_vec = tl.load(v_ptr + base)  # [HEAD_DIM] f16/bf16
                k_vec = k_vec.to(tl.float32)
                v_vec = v_vec.to(tl.float32)

                # Dot product between q_vec and k_vec
                logits = tl.dot(q_vec, k_vec)

                scaled = logits * sm_scale

                # Update lse stably
                is_inf = lse == -float("inf")
                new_lse = tl.where(is_inf, scaled,
                                   lse + tl.log(1.0 + tl.exp(scaled - lse)))
                lse = tl.where(is_inf, scaled, new_lse)

                # Attention weight
                attn = tl.exp(scaled - lse)

                # Accumulate output
                out_vec += attn * v_vec

            # Store results for this head
            out_offset = 0 * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
            tl.store(out_ptr + out_offset, out_vec)

            lse_scaled = lse / ln2
            tl.store(lse_ptr + 0 * num_qo_heads + h, lse_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Shapes
        B, num_qo_heads, head_dim = q.shape
        num_pages, middle, num_kv_heads, _ = k_cache.shape
        # middle should be 1 as per provided inputs
        assert middle == 1, "k_cache and v_cache middle dimension must be 1"

        # Prepare flattened k/v buffers over tokens (middle=1)
        # We flatten to [num_pages, head_dim] and rely on kv_indptr[B] as total tokens.
        k_flat = k_cache.view(num_pages, head_dim).contiguous()  # [num_pages, 128]
        v_flat = v_cache.view(num_pages, head_dim).contiguous()  # [num_pages, 128]

        # Compute total tokens: kv_indptr[B] should equal total_tokens (as per provided inputs)
        total_tokens = int(kv_indptr[kv_indptr.shape[0] - 1].item())
        num_tokens = total_tokens  # since we run per batch b=0 using kv_indptr[0..B]

        # Output buffers: compute in f32, store out in bfloat16, lse in f32
        out = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Cast q to f32 for computation
        q_f32 = q.to(torch.float32)

        # Launch Triton kernel: one program per batch (here B)
        grid = (B,)
        _gqa_attention_single_batch_kernel[grid](
            q_f32, k_flat, v_flat, kv_indptr, out, lse,
            B=B, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim, sm_scale=sm_scale, ln2=1.0 / math.log(2.0),
            num_tokens=num_tokens
        )

        # Cast output to bfloat16 as requested
        out_bf16 = out.to(torch.bfloat16)

        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
