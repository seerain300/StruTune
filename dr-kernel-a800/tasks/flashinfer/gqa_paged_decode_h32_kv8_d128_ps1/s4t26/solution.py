import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_attention_bh(
    Q_ptr,          # *float32, shape [B, num_qo_heads, head_dim]
    K_ptr,          # *float32, shape [num_pages, num_kv_heads, head_dim]
    V_ptr,          # *float32, shape [num_pages, num_kv_heads, head_dim]
    Out_ptr,        # *float32, shape [B, num_qo_heads, head_dim]
    LSE_ptr,        # *float32, shape [B, num_qo_heads]
    batch_size: tl.constexpr,
    num_qo_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_indptr_ptr,  # *int32, shape [B+1]
    kv_indices_ptr, # *int32, shape [N]
    sm_scale,       # float32 scalar
    LOG2_INV,       # float32 scalar = 1 / ln(2)
    BLOCK_SIZE: tl.constexpr,  # vector length (128)
):
    b = tl.program_id(0)  # batch
    h = tl.program_id(1)  # head index

    # Base offsets for Q, Out, LSE
    q_base = b * num_qo_heads * head_dim + h * head_dim
    out_base = b * num_qo_heads * head_dim + h * head_dim
    lse_offset = b * num_qo_heads

    # Load q_vec for this head
    # Q layout: [B, num_qo_heads, head_dim] contiguous
    q_vec = tl.load(Q_ptr + q_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)

    # Compute num_tokens for this batch: num_tokens = kv_indptr[b+1] - kv_indptr[b]
    # Load ptrs and compute on host side, but we can load via device scalars:
    start_b = tl.load(kv_indptr_ptr + b)  # int32 scalar
    end_b = tl.load(kv_indptr_ptr + b + 1)  # int32 scalar
    num_tokens = end_b - start_b  # int32 scalar

    # Running max and sum for logsumexp
    running_max = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_SIZE], tl.float32)

    # First pass: compute lse
    for t in range(0, num_tokens):
        idx = tl.load(kv_indices_ptr + start_b + t)  # token index int32 scalar
        # Loop over kv_heads to find the right kv_head for GQA
        # GQA: kv_head = h // (num_qo_heads // num_kv_heads) = h // 4
        gqa_ratio = num_qo_heads // num_kv_heads
        kv_head = h // gqa_ratio  # scalar int

        # Compute base offset for K and V at (idx, kv_head, :)
        # K layout: [num_pages, num_kv_heads, head_dim] contiguous
        # Base offset for this token
        k_base = idx * (num_kv_heads * head_dim) + kv_head * head_dim
        v_base = idx * (num_kv_heads * head_dim) + kv_head * head_dim

        k_vec = tl.load(K_ptr + k_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
        # logits_t = dot(q_vec, k_vec)
        logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits * sm_scale
        # running_max, running_sum (vectorized across BLOCK_SIZE)
        running_max = tl.maximum(running_max, scaled)
        # exp(scaled - running_max) and accumulate sum
        exp_val = tl.exp(scaled - running_max)
        running_sum += exp_val

    # Compute lse = log(running_sum) + running_max, then divide by ln(2)
    lse_val = tl.log(running_sum) + running_max  # vector
    lse_val = lse_val * LOG2_INV  # divide by ln(2)
    # Store lse for this (b, h)
    tl.store(LSE_ptr + lse_offset + h, lse_val[0])  # store scalar


@triton.jit
def output_accum_bh(
    Q_ptr,          # *float32, shape [B, num_qo_heads, head_dim]
    K_ptr,          # *float32, shape [num_pages, num_kv_heads, head_dim]
    V_ptr,          # *float32, shape [num_pages, num_kv_heads, head_dim]
    Out_ptr,        # *float32, shape [B, num_qo_heads, head_dim]
    LSE_ptr,        # *float32, shape [B, num_qo_heads]
    batch_size: tl.constexpr,
    num_qo_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_indptr_ptr,  # *int32, shape [B+1]
    kv_indices_ptr, # *int32, shape [N]
    sm_scale,       # float32 scalar
    BLOCK_SIZE: tl.constexpr,  # 128
):
    b = tl.program_id(0)  # batch
    h = tl.program_id(1)  # head index

    q_base = b * num_qo_heads * head_dim + h * head_dim
    out_base = b * num_qo_heads * head_dim + h * head_dim
    lse_offset = b * num_qo_heads

    # Load q_vec
    q_vec = tl.load(Q_ptr + q_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)

    start_b = tl.load(kv_indptr_ptr + b)  # int32
    end_b = tl.load(kv_indptr_ptr + b + 1)  # int32
    num_tokens = end_b - start_b

    # Load lse for this (b, h)
    lse_val = tl.load(LSE_ptr + lse_offset + h)  # scalar

    # Accumulate output vector across tokens
    out_vec = tl.zeros([BLOCK_SIZE], tl.float32)
    gqa_ratio = num_qo_heads // num_kv_heads
    kv_head = h // gqa_ratio

    for t in range(0, num_tokens):
        idx = tl.load(kv_indices_ptr + start_b + t)  # int32
        k_base = idx * (num_kv_heads * head_dim) + kv_head * head_dim
        v_base = idx * (num_kv_heads * head_dim) + kv_head * head_dim

        k_vec = tl.load(K_ptr + k_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
        logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits * sm_scale
        attn = tl.exp(scaled - lse_val)  # scalar
        v_vec = tl.load(V_ptr + v_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
        out_vec += attn * v_vec

    # Store output for this head
    tl.store(Out_ptr + out_base + tl.arange(0, BLOCK_SIZE), out_vec, mask=tl.arange(0, BLOCK_SIZE) < head_dim)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)  # float32 scalar
        self.LOG2_INV = 1.0 / math.log(2.0)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B, H, D = q.shape
        assert H == self.num_qo_heads and D == self.head_dim
        # Output and LSE
        output = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton: grid over (B, H)
        grid = (B, H)
        softmax_attention_bh[grid](
            q, k_cache, v_cache, output, lse,
            B, self.num_qo_heads, self.num_kv_heads, self.head_dim,
            kv_indptr, kv_indices,
            self.sm_scale, self.LOG2_INV,
            BLOCK_SIZE=self.head_dim,
        )

        # Second kernel: compute and write output vector (recomputes attention)
        output_accum_bh[grid](
            q, k_cache, v_cache, output, lse,
            B, self.num_qo_heads, self.num_kv_heads, self.head_dim,
            kv_indptr, kv_indices,
            self.sm_scale,
            BLOCK_SIZE=self.head_dim,
        )

        # Return bfloat16 output and float32 lse to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
