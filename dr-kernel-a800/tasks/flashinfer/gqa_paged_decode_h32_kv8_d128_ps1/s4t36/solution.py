import torch
import math
import triton
import triton.language as tl


@triton.jit
def softmax_attention_bh(
    Q_ptr,            # *f32, shape [B, num_qo_heads, head_dim]
    K_ptr,            # *f32, shape [N, num_kv_heads, head_dim]
    V_ptr,            # *f32, shape [N, num_kv_heads, head_dim]
    Out_ptr,          # *f32, shape [B, num_qo_heads, head_dim]
    LSE_ptr,          # *f32, shape [B, num_qo_heads]
    B: tl.constexpr,  # batch size (constexpr for codegen)
    N: tl.constexpr,  # total tokens across batches (unused for shape, kept for signature)
    num_qo_heads: tl.constexpr,  # 32
    num_kv_heads: tl.constexpr,  # 8
    head_dim: tl.constexpr,      # 128
    num_tokens_b,     # int32 runtime: number of tokens for this batch
    start_index,      # int32 runtime: starting index in kv_indices for this batch
    gqa_ratio,        # int32 runtime: num_qo_heads // num_kv_heads (4)
    SM_SCALE,         # f32 runtime scalar: 1.0 / sqrt(head_dim) (provided)
    LOG2_INVERSE,     # f32 runtime scalar: 1.0 / ln(2) (approx 1.4426950408889634)
):
    # Each program handles one (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load Q[b, h, :]
    q_off = b * num_qo_heads * head_dim + h * head_dim
    q_vec = tl.load(Q_ptr + q_off + tl.arange(0, head_dim))

    # First pass: compute logsumexp over scaled logits for all tokens in this batch
    running_max = -float('inf')
    running_sum = 0.0

    for t in range(0, num_tokens_b):
        token_idx = start_index + t
        kv_head = h // gqa_ratio

        # K and V are [N, num_kv_heads, head_dim]; we slice by token_idx and kv_head
        k_off = token_idx * (num_kv_heads * head_dim) + kv_head * head_dim
        v_off = token_idx * (num_kv_heads * head_dim) + kv_head * head_dim

        k_vec = tl.load(K_ptr + k_off + tl.arange(0, head_dim))
        v_vec = tl.load(V_ptr + v_off + tl.arange(0, head_dim))

        # logits = dot(q_vec, k_vec)
        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE

        # numerically stable update for logsumexp
        running_max = tl.maximum(running_max, scaled)
        running_sum += tl.exp(scaled - running_max)

    # lse = logsumexp(scaled) / log(2)
    lse_value = tl.log(running_sum) + running_max
    lse_value = lse_value * LOG2_INVERSE

    # Second pass: compute attention and accumulate output
    out_vec = tl.zeros([head_dim], dtype=tl.float32)

    for t in range(0, num_tokens_b):
        token_idx = start_index + t
        kv_head = h // gqa_ratio

        k_off = token_idx * (num_kv_heads * head_dim) + kv_head * head_dim
        v_off = token_idx * (num_kv_heads * head_dim) + kv_head * head_dim

        k_vec = tl.load(K_ptr + k_off + tl.arange(0, head_dim))
        v_vec = tl.load(V_ptr + v_off + tl.arange(0, head_dim))

        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE
        attn = tl.exp(scaled - lse_value)  # softmax over scaled logits

        out_vec += attn * v_vec

    # Store output and lse
    out_off = b * num_qo_heads * head_dim + h * head_dim
    tl.store(Out_ptr + out_off + tl.arange(0, head_dim), out_vec)
    tl.store(LSE_ptr + b * num_qo_heads + h, lse_value)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda, "Inputs must be on CUDA for Triton kernels."
        device = q.device

        # Shapes: q [B, 32, 128], k_cache/v_cache [num_pages, 8, 128], kv_indptr [B+1], kv_indices [num_tokens]
        B, num_qo_heads, head_dim = q.shape
        N, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Compute num_tokens per batch: num_tokens[b] = kv_indptr[b+1] - kv_indptr[b]
        num_tokens = [int(kv_indptr[b + 1].item() - kv_indptr[b].item()) for b in range(B)]
        # Compute start index per batch: starts[b] = sum_{i=0..b-1} num_tokens[i]
        starts = [0] * B
        for b in range(1, B):
            starts[b] = starts[b - 1] + num_tokens[b - 1]

        # Prepare output tensors
        out = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        gqa_ratio = num_qo_heads // num_kv_heads  # = 4

        # Launch Triton kernel: one program per (b, h)
        LOG2_INVERSE = 1.4426950408889634  # 1 / ln(2)

        # We need K_t and V_t per batch: rows indexed by kv_indices[starts[b]:starts[b]+num_tokens[b]]
        # Use torch.index_select on 3D tensors k_cache and v_cache (these are [N, num_kv_heads, head_dim])
        for b in range(B):
            token_count = num_tokens[b]
            start_idx = starts[b]
            if token_count == 0:
                # No KV for this batch element
                out[b].zero_()
                lse[b].zero_()
                continue

            token_indices = kv_indices[start_idx:start_idx + token_count].to(torch.int64).contiguous()
            K_t = torch.index_select(k_cache, 0, token_indices).contiguous()  # [token_count, 8, 128]
            V_t = torch.index_select(v_cache, 0, token_indices).contiguous()  #


def run(*args):
    return ModelNew()(*args)
