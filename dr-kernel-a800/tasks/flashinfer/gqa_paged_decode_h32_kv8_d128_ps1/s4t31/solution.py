import torch
import triton
import triton.language as tl
import math

# Constants
HEAD_DIM = 128
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS  # 4

LOG2_INV = 1.4426950408889634  # 1 / ln(2)


@triton.jit
def softmax_and_attention_bh(
    q_ptr,           # *fp32, [B, 32, 128]
    k_cache_ptr,     # *fp32, [N, 8, 128]
    v_cache_ptr,     # *fp32, [N, 8, 128]
    kv_indptr_ptr,   # *int32, [B+1]
    kv_indices_ptr,  # *int32, [num_tokens]
    output_ptr,      # *fp32, [B, 32, 128]
    lse_ptr,         # *fp32, [B, 32]
    SM_SCALE: tl.constexpr,  # scalar float32
    HEAD_DIM: tl.constexpr,  # 128
    NUM_KV_HEADS: tl.constexpr,  # 8
    GQA_RATIO: tl.constexpr,  # 4
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load q[b, h, :]
    q_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM + tl.arange(0, HEAD_DIM)
    q_vec = tl.load(q_ptr + q_off)  # [128], fp32

    # Compute number of tokens for this batch: num_tokens = kv_indptr[b+1] - kv_indptr[b]
    start_b = tl.load(kv_indptr_ptr + b)       # int32
    end_b = tl.load(kv_indptr_ptr + b + 1)     # int32
    num_tokens_b = end_b - start_b             # int32 scalar

    # Initialize numerically stable logsumexp
    running_max = -float("inf")
    running_sum = 0.0

    # First pass: compute lse over all tokens for this (b, h)
    t = 0
    while t < num_tokens_b:
        # idx = kv_indices[kv_indptr[b] + t]
        idx = tl.load(kv_indices_ptr + start_b + t)  # int32
        kv_head = h // GQA_RATIO  # int32
        # Load k_vec and v_vec for this token and head: shapes [128]
        k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
        v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
        k_vec = tl.load(k_cache_ptr + k_off)  # [128]
        v_vec = tl.load(v_cache_ptr + v_off)  # [128]
        logits = tl.sum(q_vec * k_vec, axis=0)  # scalar fp32
        scaled = logits * SM_SCALE
        running_max_new = tl.maximum(running_max, scaled)
        exp_term = tl.exp(scaled - running_max)
        running_sum = running_sum * tl.exp(running_max - running_max_new) + exp_term
        running_max = running_max_new
        t += 1

    # lse = log(running_sum) + running_max; divide by ln(2)
    lse = tl.log(running_sum) + running_max
    lse = lse * LOG2_INV
    tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse)

    # Second pass: compute output[b, h, :] = sum_j attn_j * v_vec_j
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    t = 0
    while t < num_tokens_b:
        idx = tl.load(kv_indices_ptr + start_b + t)
        kv_head = h // GQA_RATIO
        k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
        v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
        k_vec = tl.load(k_cache_ptr + k_off)  # [128]
        v_vec = tl.load(v_cache_ptr + v_off)  # [128]
        logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits * SM_SCALE
        attn = tl.exp(scaled - lse)  # scalar
        out_vec += attn * v_vec  # elementwise multiply then sum across v_vec? No: attn is scalar; add scalar times vector to vector is elementwise multiply
        t += 1

    # Store output[b, h, :]
    out_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM + tl.arange(0, HEAD_DIM)
    tl.store(output_ptr + out_off, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and correct shapes
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All inputs must be CUDA tensors"
        B, QH, HD = q.shape
        assert QH == NUM_QO_HEADS and HD == HEAD_DIM, "q must be [B, 32, 128]"
        # k_cache and v_cache must be [N, 8, 128]
        Nk, Kh, Dh = k_cache.shape
        Nv, Vh, Dv = v_cache.shape
        assert Kh == NUM_KV_HEADS and Dh == HEAD_DIM and Vh == NUM_KV_HEADS and Dv == HEAD_DIM, "k_cache and v_cache must be [N, 8, 128]"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr must have shape [B+1]"
        num_tokens = int(kv_indptr[-1].item())  # total tokens across all batches
        assert kv_indices.shape[0] == num_tokens, "kv_indices length must equal total tokens"
        assert kv_indptr[B].item() == 0 and kv_indptr[B+1].item() == num_tokens, "kv_indptr must be valid"

        # Prepare inputs: cast to fp32 and ensure contiguous
        q_fp32 = q.to(torch.float32).contiguous()
        k_cache_fp32 = k_cache.to(torch.float32).contiguous()
        v_cache_fp32 = v_cache.to(torch.float32).contiguous()
        kv_indptr_fp32 = kv_indptr.to(torch.int32).contiguous()
        kv_indices_fp32 = kv_indices.to(torch.int32).contiguous()

        # Allocate outputs
        output = torch.empty((B, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, NUM_QO_HEADS), dtype=torch.float32, device=q.device)

        # Launch Triton kernel with grid (B, 32)
        grid = (B, NUM_QO_HEADS)
        softmax_and_attention_bh[grid](
            q_fp32,
            k_cache_fp32,
            v_cache_fp32,
            kv_indptr_fp32,
            kv_indices_fp32,
            output,
            lse,
            SM_SCALE=sm_scale,
            HEAD_DIM=HEAD_DIM,
            NUM_KV_HEADS=NUM_KV_HEADS,
            GQA_RATIO=GQA_RATIO,
            num_warps=4,
        )

        # Return output as bfloat16 and lse as float32 (matching original)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
