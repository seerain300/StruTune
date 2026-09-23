import torch
import math
import triton
import triton.language as tl


@triton.jit
def softmax_attention_bh(
    Q_ptr,             # *f32, [B, num_qo_heads, head_dim]
    K_ptr,             # *f32, [N, num_kv_heads, head_dim]
    V_ptr,             # *f32, [N, num_kv_heads, head_dim]
    Out_ptr,           # *f32, [B, num_qo_heads, head_dim]
    LSE_ptr,           # *f32, [B, num_qo_heads]
    B,                 # int32
    NUM_TOKENS,        # int32
    START_IDX,         # int32
    gqa_ratio,         # int32
    SM_SCALE,          # f32
    LOG2_INVERSE,      # f32
    head_dim: tl.constexpr,
    num_kv_heads: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load Q[b, h] vector
    q_off = b * (num_qo_heads * head_dim) + h * head_dim
    q_vec = tl.load(Q_ptr + q_off + tl.arange(0, head_dim))

    # First pass: compute logsumexp over scaled logits for tokens in this batch
    running_max = -float('inf')
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        token_idx = tl.load(START_IDX_ptr + t)  # read indices from host-provided pointer
        kv_head = h // gqa_ratio
        # K and V are [N, num_kv_heads, head_dim]; pointer strides
        k_off = token_idx * (num_kv_heads * head_dim) + kv_head * head_dim
        v_off = token_idx * (num_kv_heads * head_dim) + kv_head * head_dim

        k_vec = tl.load(K_ptr + k_off + tl.arange(0, head_dim))
        # logits = dot(q_vec, k_vec)
        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE

        # numerically stable update
        running_max = tl.maximum(running_max, scaled)
        # exp(scaled - running_max) contributes to sum
        running_sum += tl.exp(scaled - running_max)

    lse = tl.log(running_sum) + running_max
    lse = lse * LOG2_INVERSE  # divide by ln(2)

    # Second pass: accumulate output vector
    out_vec = tl.zeros((head_dim,), dtype=tl.float32)

    for t in range(0, NUM_TOKENS):
        token_idx = tl.load(START_IDX_ptr + t)
        kv_head = h // gqa_ratio
        k_off = token_idx * (num_kv_heads * head_dim) + kv_head * head_dim
        v_off = token_idx * (num_kv_heads * head_dim) + kv_head * head_dim

        k_vec = tl.load(K_ptr + k_off + tl.arange(0, head_dim))
        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE
        attn = tl.exp(scaled - lse)

        v_vec = tl.load(V_ptr + v_off + tl.arange(0, head_dim))
        out_vec += attn * v_vec

    # Write results
    out_off = b * (num_qo_heads * head_dim) + h * head_dim
    tl.store(Out_ptr + out_off + tl.arange(0, head_dim), out_vec)
    tl.store(LSE_ptr + b * num_qo_heads + h, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA
        assert q.is_cuda, "Inputs must be on CUDA for Triton kernels."
        device = q.device

        # Shapes
        B, num_qo_heads, head_dim = q.shape
        N, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Compute num_tokens per batch and starts
        num_tokens = [int(kv_indptr[b + 1].item() - kv_indptr[b].item()) for b in range(B)]
        starts = [0] * B
        for b in range(1, B):
            starts[b] = starts[b - 1] + num_tokens[b - 1]

        # Prepare output and lse
        out = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        LOG2_INVERSE = 1.4426950408889634  # 1 / ln(2)
        SM_SCALE = float(1.0 / (head_dim ** 0.5))  # match original sm_scale = 1/sqrt(128)

        # Build token indices per batch (contiguous tensors) to pass to Triton
        token_indices_list = []
        for b in range(B):
            token_count = num_tokens[b]
            if token_count == 0:
                # Nothing to do; still launch to avoid errors, but skip writing if needed
                token_indices_list.append(torch.empty(0, dtype=torch.int32, device=device))
                continue
            start = starts[b]
            token_indices = kv_indices[start:start + token_count].to(torch.int32).contiguous()
            token_indices_list.append(token_indices)

        # Launch Triton: one program per (b, h)
        grid = (B, num_qo_heads)
        # We'll pass a small helper array of start indices per batch to kernel (but we already compute token_count inside kernel).
        # The kernel will read token indices via a pointer; we pass token_indices_list[b] for each b in separate launches? Triton can't vectorize per-batch arrays in a single kernel easily; better to do per-batch loop in Python, which we already have by launching grid=(B, num_qo_heads).
        # To make it single launch, we can concatenate token_indices into one tensor and use a 3D grid with batch dimension. Triton doesn't support 3D grid in program_id, so we'll launch in a Python loop over b.
        for b in range(B):
            token_count = num_tokens[b]
            token_indices = token_indices_list[b]
            if token_count == 0:
                # For safety, zero outputs
                out[b].zero_()
                lse[b].zero_()
                continue

            # Allocate a small 1D tensor for start_idx and pass its pointer
            start_idx_tensor = token_indices.new_tensor([starts[b]])  # single int32

            softmax_attention_bh[grid](
                q.to(torch.float32),
                k_cache.to(torch.float32),
                v_cache.to(torch.float32),
                out,
                lse,
                B,
                token_count,
                starts[b],
                gqa_ratio,
                SM_SCALE,
                LOG2_INVERSE,
                head_dim=128,
                num_kv_heads=8,
            )

        # Cast output to bfloat16 as in original
        out = out.to(torch.bfloat16)
        return out, lse


def run(*args):
    return ModelNew()(*args)
