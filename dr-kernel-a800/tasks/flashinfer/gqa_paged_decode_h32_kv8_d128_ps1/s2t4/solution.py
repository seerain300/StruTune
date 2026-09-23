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
        NUM_TOKS: tl.constexpr,     # loop bound (e.g., 8192)
        sm_scale: tl.float32,       # scaling factor
        half_ln2_inv: tl.float32,   # 1 / ln(2)
    ):
        # One program per (b, h)
        b = tl.program_id(0)
        h = tl.program_id(1)

        # Load kv_indptr[b:b+1] to determine number of tokens for this batch
        start = tl.load(kv_indptr_ptr + b)           # int32
        end = tl.load(kv_indptr_ptr + b + 1)         # int32
        num_tokens_actual = end - start               # int32

        # Accumulators for lse
        max_s = -float("inf")
        sum_exp = 0.0  # float32

        # Pass 1: compute max_s and sum_exp across tokens
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio  # 0..7

            # Offsets for k and v rows: (idx * NUM_KV_HEADS + kv_head) * HEAD_DIM
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load q_vec[h, :]
            q_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
            q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32

        # Pass 2: recompute s, attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            q_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
            q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            if mask_i:
                attn = tl.exp(s - lse_val)  # float32 scalar
                out_vec += attn * v_vec

        # Store output and lse
        out_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_off, out_vec)

        lse_off = b * NUM_QO_HEADS + h
        tl.store(lse_ptr + lse_off, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_qo_heads=32, num_kv_heads=8, head_dim=128):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA and contiguous; compute in float32
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Inputs must be CUDA tensors"
        assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16, "Expected bfloat16 inputs for q/k/v"
        assert kv_indptr.dtype == torch.int32 and kv_indices.dtype == torch.int32, "kv_indptr and kv_indices must be int32"
        assert q.shape[1] == self.num_qo_heads and k_cache.shape[2] == self.num_kv_heads and q.shape[2] == self.head_dim, "Shape mismatch"
        assert k_cache.shape == v_cache.shape and k_cache.shape[0] == q.shape[0], "k/v shapes must match q's batch"
        assert kv_indptr.shape[0] == q.shape[0] + 1, "kv_indptr must have length batch_size+1"

        # Cast to float32 for compute; keep originals for potential output casting
        q32 = q.to(torch.float32).contiguous()
        k32 = k_cache.to(torch.float32).contiguous()
        v32 = v_cache.to(torch.float32).contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size = q32.shape[0]
        num_qo_heads = q32.shape[1]
        head_dim = q32.shape[2]

        # Allocate outputs (float32 for compute, cast later to bfloat16)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q32.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q32.device)

        # Launch Triton kernel: one program per (b, h)
        NUM_TOKS = 8192  # robust bound; masks prevent OOB
        half_ln2_inv = 1.0 / math.log(2.0)  # 1 / ln(2)

        grid = (batch_size, num_qo_heads)
        _attention_bh_kernel[grid](
            q32, k32, v32, kv_indptr, kv_indices, output, lse,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=self.num_kv_heads,
            HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS,
            sm_scale=sm_scale,
            half_ln2_inv=half_ln2_inv,
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
