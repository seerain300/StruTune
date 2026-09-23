import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h).
# It loops over up to NUM_TOKS tokens with masks, computing:
# - lse[b, h] = logsumexp(scaled_logits) * (1 / ln(2))
# - output[b, h, :] = sum_i softmax(scaled_logits_i) * v_i
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
        NUM_TOKS: tl.constexpr,      # fixed loop bound (e.g., 8192)
        BATCH_SIZE: tl.constexpr,    # meta-args for clarity
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HALF_LN2: tl.constexpr,      # 1 / ln(2) = 1.4426950408889634
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
            idx = tl.load(kv_indices_ptr + start + i)  # int32; i < num_tokens_actual guaranteed by mask
            mask_i = i < num_tokens_actual

            # Offsets for k and v rows for this kv_head
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_vec and v_vec with mask (invalid loads produce zeros)
            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only for valid iterations
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = (log(max_s) + log(sum_exp)) * (1/ln(2))
        lse_val = (tl.log(max_s) + tl.log(sum_exp)) * HALF_LN2  # float32
        # Store lse for this (b, h)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            idx = tl.load(kv_indices_ptr + start + i)  # int32
            mask_i = i < num_tokens_actual

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # attn = exp(s - lse_val) for valid i, else 0
            attn = tl.where(mask_i, tl.exp(s - lse_val), 0.0)

            # Accumulate output vector
            out_vec += attn * v_vec

        # Store output vector for this (b, h)
        out_offset = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_offset + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size: int = 1, num_qo_heads: int = 32, num_kv_heads: int = 8, head_dim: int = 128, num_tok_bound: int = 8192):
        super().__init__()
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_tok_bound = num_tok_bound
        # sm_scale
        self.sm_scale = 1.0 / math.sqrt(head_dim)
        # 1 / ln(2)
        self.half_ln2 = 1.4426950408889634  # log(2)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices):
        """
        q: [batch_size, 32, 128], dtype bfloat16 or float32. We move to CUDA and cast to float32 for compute.
        k_cache, v_cache: [num_pages, 1, 8, 128], dtype bfloat16 or float32.
        kv_indptr: [batch_size+1], int32.
        kv_indices: [num_kv_indices], int32.
        Returns: (output [batch_size, 32, 128] bfloat16), lse [batch_size, 32] float32.
        """
        assert TRITON_AVAILABLE, "Triton is not available."

        # Ensure CUDA and contiguous; compute in float32
        device = q.device
        q32 = q.contiguous().to(torch.float32)           # [B, 32, 128]
        k32 = k_cache.contiguous().to(torch.float32)     # [N, 1, 8, 128] -> we ignore the '1' dim in loads
        v32 = v_cache.contiguous().to(torch.float32)     # [N, 1, 8, 128]
        kv_indptr32 = kv_indptr.contiguous().to(torch.int32)
        kv_indices32 = kv_indices.contiguous().to(torch.int32)

        BATCH_SIZE = q32.shape[0]
        NUM_QO_HEADS = q32.shape[1]
        HEAD_DIM = q32.shape[2]
        NUM_KV_HEADS = k32.shape[2]
        NUM_TOKS = self.num_tok_bound  # fixed bound for Triton loop

        # Allocate output and lse in float32 for computation
        out32 = torch.empty((BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=device)
        lse32 = torch.empty((BATCH_SIZE, NUM_QO_HEADS), dtype=torch.float32, device=device)

        # Launch Triton kernel: grid = (BATCH_SIZE, NUM_QO_HEADS)
        _attention_bh_kernel[(BATCH_SIZE, NUM_QO_HEADS)](
            q32, k32, v32,
            kv_indptr32, kv_indices32,
            out32, lse32,
            self.sm_scale,
            NUM_TOKS=NUM_TOKS,
            BATCH_SIZE=BATCH_SIZE,
            NUM_QO_HEADS=NUM_QO_HEADS,
            NUM_KV_HEADS=NUM_KV_HEADS,
            HEAD_DIM=HEAD_DIM,
            HALF_LN2=self.half_ln2,
            num_warps=4,  # modest number of warps for small vectors
        )

        # Return output in bfloat16 (as original), lse in float32
        out_bf16 = out32.to(torch.bfloat16)
        return out_bf16, lse32


def run(*args):
    return ModelNew()(*args)
