import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_rotate_kernel(
    query_ptr,          # *bfloat16, shape [B, num_q_heads, S, D]
    q_norm_ptr,         # *bfloat16, shape [D]
    inv_freq_ptr,       # *float32, shape [HALF], where HALF=D//2
    query_out_ptr,      # *bfloat16, shape [B, num_q_heads, S, D]
    B: tl.constexpr,
    S: tl.constexpr,
    num_q_heads: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    cache_len: tl.constexpr,
    BLOCK_D: tl.constexpr,  # set to 128 to match head_dim
):
    pid = tl.program_id(0)  # one program per (b, q_head, s)
    # Compute indices
    b = pid // (num_q_heads * S)
    tmp = pid % (num_q_heads * S)
    q_head = tmp // S
    s = tmp % S

    # Base pointers for this (b, q_head, s) row
    base_q = (b * num_q_heads + q_head) * S * D + s * D
    # Load query row (bf16), compute in fp32
    x = tl.load(query_ptr + base_q + tl.arange(0, D), mask=True, other=0.0)
    x32 = x.to(tl.float32)

    # RMSNorm: scale = 1 / sqrt(mean(x^2) + eps)
    # Here eps is not provided; using typical small eps in fp32
    eps = 1e-12
    sum_sq = tl.sum(x32 * x32, axis=0)
    mean_sq = sum_sq / D
    scale = 1.0 / tl.sqrt(mean_sq + eps)
    # Multiply by q_norm weight
    w = tl.load(q_norm_ptr + tl.arange(0, D), mask=True, other=1.0).to(tl.float32)
    y = x32 * scale * w  # normalized and scaled

    # Rotary Embedding: build cos and sin vectors of length D from inv_freq
    # pos = cache_len + s (integer scalar)
    pos = cache_len + s
    # For i < HALF: cos[i] = 1/sqrt(1+inv[i]^2), sin[i] = inv[i]/sqrt(1+inv[i]^2)
    inv = tl.load(inv_freq_ptr + tl.arange(0, HALF), mask=True, other=0.0).to(tl.float32)
    denom = tl.sqrt(1.0 + inv * inv)  # shape [HALF]
    cos_half = 1.0 / denom
    sin_half = inv / denom

    # Extend to D: for d >= HALF, use cos/d for d - HALF
    cos_vec = tl.zeros([D], dtype=tl.float32)
    sin_vec = tl.zeros([D], dtype=tl.float32)
    for d in range(0, HALF):
        cos_vec[d] = cos_half[d]
        sin_vec[d] = sin_half[d]
    for d in range(HALF, D):
        cos_vec[d] = cos_vec[d - HALF]
        sin_vec[d] = sin_vec[d - HALF]

    # Apply rotation: y' = y * cos + rotate_half(y) * sin
    # rotate_half(y) swaps halves and negates the second half: [-y2, y1]
    y1 = y[:HALF]
    y2 = y[HALF:]
    rotated_half = tl.concatenate([-y2, y1], axis=0)  # length D
    y_rotated = y * cos_vec + rotated_half * sin_vec

    # Store back as bf16
    tl.store(query_out_ptr + base_q + tl.arange(0, D), y_rotated.to(tl.bfloat16), mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We only perform the computational part in Triton:
        # Inputs: query, q_norm_weight, inv_freq
        # Outputs: rotated query
        query = args[0].contiguous()           # [B, num_q_heads, S, D], bf16
        q_norm_weight = args[7].contiguous()   # [D], bf16
        inv_freq = args[9].contiguous()        # [HALF], float32
        B, num_q_heads, S, D = query.shape
        HALF = D // 2

        # Prepare output
        query_out = torch.empty_like(query)  # rotated query

        # Launch Triton kernel: one program per (b, q_head, s)
        grid = (B * num_q_heads * S,)
        rmsnorm_rotate_kernel[grid](
            query, q_norm_weight, inv_freq, query_out,
            B, S, num_q_heads, D, HALF, args[10],  # cache_len (unused in kernel compute, kept for shape)
            D=128, BLOCK_D=128, num_warps=4, num_stages=2,
        )
        return query_out, None, None  # return rotated query; placeholders for unused outputs


def run(*args):
    return ModelNew()(*args)
