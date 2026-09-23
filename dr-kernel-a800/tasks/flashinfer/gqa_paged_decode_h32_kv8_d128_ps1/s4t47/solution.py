import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    q_ptr,             # *fp32 [B, num_qo_heads, D]
    k_ptr,             # *fp32 [num_tokens_b, H, D]
    v_ptr,             # *fp32 [num_tokens_b, H, D]
    out_ptr,           # *fp32 [B, num_qo_heads, D]
    lse_ptr,           # *fp32 [B, num_qo_heads]
    num_tokens_b,      # int32 scalar: number of tokens for this batch element
    sm_scale,          # fp32 scalar: 1/sqrt(D)
    H,                 # int32: num_kv_heads
    D: tl.constexpr,   # head_dim (128), compile-time for vector ops
    gqa_ratio: tl.constexpr,  # num_qo_heads // num_kv_heads (4)
):
    # program ids
    b = tl.program_id(0)
    h = tl.program_id(1)

    # GQA mapping
    kv_head = h // gqa_ratio  # 0..7

    # Pointers to q vector for this (b, h)
    q_row_start = b * (num_qo_heads * D) + h * D
    q_vec = tl.load(q_ptr + q_row_start + tl.arange(0, D))

    # Numerically stable logsumexp over scaled logits
    running_max = -float("inf")
    running_sum = 0.0

    t = 0
    while t < num_tokens_b:
        # K_t[t, kv_head, :] and V_t[t, kv_head, :]
        k_row_start = t * (H * D) + kv_head * D
        v_row_start = t * (H * D) + kv_head * D

        k_vec = tl.load(k_ptr + k_row_start + tl.arange(0, D))
        v_vec = tl.load(v_ptr + v_row_start + tl.arange(0, D))

        # logits_t = dot(q_vec, k_vec)
        logits = tl.sum(q_vec * k_vec, axis=0)  # scalar fp32
        scaled = logits * sm_scale

        # update running max/sum
        if scaled > running_max:
            running_sum = running_sum * tl.exp(running_max - scaled) + 1.0
            running_max = scaled
        else:
            running_sum += tl.exp(scaled - running_max)
        t += 1

    # lse = log(running_sum) + running_max, divide by ln(2) (multiply by 1/ln(2))
    log2_inv = 1.4426950408889634
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * log2_inv

    # Store lse for this (b, h)
    tl.store(lse_ptr + b * num_qo_heads + h, lse_val)

    # Second pass: compute output[b, h, :] = sum_j exp(scaled_j) * v_j
    out_vec = tl.zeros([D], dtype=tl.float32)
    t = 0
    while t < num_tokens_b:
        k_row_start = t * (H * D) + kv_head * D
        v_row_start = t * (H * D) + kv_head * D

        k_vec = tl.load(k_ptr + k_row_start + tl.arange(0, D))
        v_vec = tl.load(v_ptr + v_row_start + tl.arange(0, D))

        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * sm_scale
        attn = tl.exp(scaled - lse_val)  # scalar fp32
        out_vec += attn * v_vec
        t += 1

    # Store output vector for this (b, h)
    out_row_start = b * (num_qo_heads * D) + h * D
    tl.store(out_ptr + out_row_start + tl.arange(0, D), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # q: [B, 32, 128], k_cache/v_cache: [num_pages, H, D], H=8, D=128
        B, num_qo_heads, D = q.shape
        assert num_qo_heads == 32

        H = k_cache.shape[1]
        assert k_cache.shape == v_cache.shape and len(k_cache.shape) == 3
        num_pages = k_cache.shape[0]
        assert k_cache.shape[2] == D
        assert v_cache.shape == (num_pages, H, D)

        # Output buffers (fp32 compute; cast to bfloat16 at end)
        output = torch.empty((B, num_qo_heads, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Compute num_tokens per batch element on host and pass to kernel
        num_tokens_per_batch = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_per_batch.append(end - start)
        num_tokens_per_batch = torch.tensor(num_tokens_per_batch, dtype=torch.int32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, num_qo_heads)
        softmax_and_attention_single_bh[grid](
            q.to(torch.float32),                # Triton expects fp32; cast q
            k_cache.to(torch.float32),          # ensure fp32
            v_cache.to(torch.float32),          # ensure fp32
            output,
            lse,
            num_tokens_per_batch,               # per-batch token counts
            sm_scale,                           # scalar fp32, e.g., 1/sqrt(128)
            H,                                  # num_kv_heads
            D,                                  # head_dim (128)
            num_qo_heads // H,                  # gqa_ratio = 4
            num_warps=4,
        )

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
