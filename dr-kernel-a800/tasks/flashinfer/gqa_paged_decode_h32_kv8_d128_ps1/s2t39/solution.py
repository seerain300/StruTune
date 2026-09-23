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
        sm_scale,         # float32 scalar
        NUM_TOKS: tl.constexpr,      # fixed loop bound
        BATCH_SIZE: tl.constexpr,    # meta-args for compilation
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HALF_LN2: tl.constexpr,      # 1 / ln(2) as float32
    ):
        # program ids
        b = tl.program_id(0)  # batch index
        h = tl.program_id(1)  # query head index

        # GQA mapping: kv_head = h // (NUM_QO_HEADS // NUM_KV_HEADS) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h] vector: shape [HEAD_DIM], float32
        q_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset + tl.arange(0, HEAD_DIM))  # [HEAD_DIM], float32

        # Compute lse components across tokens
        max_s = -float("inf")
        sum_exp = 0.0  # scalar float32

        # start and end indices from kv_indptr
        start = tl.load(kv_indptr_ptr + b)     # int32
        end = tl.load(kv_indptr_ptr + b + 1)   # int32
        num_tokens_actual = end - start        # int32

        # Pass 1: compute max and sum(exp(s - max)) across tokens
        for i in range(NUM_TOKS):
            idx = tl.load(kv_indices_ptr + start + i)  # int32
            mask_i = i < num_tokens_actual

            # Offsets for k and v rows for this kv_head
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec (invalid loads produce zeros due to mask)
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
        lse_val = tl.log(max_s) + tl.log(sum_exp) * HALF_LN2  # float32
        # Store lse for this (b, h)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            idx = tl.load(kv_indices_ptr + start + i)  # int32
            mask_i = i < num_tokens_actual

            # Offsets for k and v rows for this kv_head
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec (invalid loads produce zeros due to mask)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # attn_i = exp(s - lse)
            attn_i = tl.exp(s - lse_val)  # scalar float32

            # Accumulate output vector
            if mask_i:
                out_vec += attn_i * v_vec

        # Store output vector for this (b, h)
        tl.store(out_ptr + b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [batch_size, 32, 128]
        k_cache: [num_pages, 1, 8, 128]
        v_cache: [num_pages, 1, 8, 128]
        kv_indptr: [len_indptr] (e.g., [2] for batch_size=1), int32
        kv_indices: [num_kv_indices], int32
        sm_scale: float32 scalar, e.g., 1.0 / sqrt(128)
        Returns: output [batch_size, 32, 128] bfloat16, lse [batch_size, 32] float32
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        device = q.device
        assert device.type == "cuda", "This Triton implementation requires CUDA tensors"

        batch_size = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        num_pages, _, num_kv_heads, _ = k_cache.shape

        # Make inputs contiguous and float32 for compute
        q_f = q.to(torch.float32).contiguous()
        k_f = k_cache.to(torch.float32).contiguous()
        v_f = v_cache.to(torch.float32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        # Output and lse buffers (float32 for compute)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        # Choose a large but conservative loop bound; mask handles actual token count
        NUM_TOKS = 8192
        # 1 / ln(2)
        HALF_LN2 = 1.4426950408889634  # float32

        _attention_bh_kernel[grid](
            q_f, k_f, v_f, kv_indptr, kv_indices, output, lse, sm_scale,
            NUM_TOKS=NUM_TOKS,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            HALF_LN2=HALF_LN2,
            num_warps=4,  # reasonable default for 128-D work
            num_stages=2,
        )

        # Return output as bfloat16 (original), and lse as float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
