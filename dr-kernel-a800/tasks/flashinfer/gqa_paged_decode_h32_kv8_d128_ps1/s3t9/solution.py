import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_bh_kernel(
    q_ptr,          # *bfloat16, [B, H, D], contiguous
    k_ptr,          # *float32,  [N, D] (k_cache.squeeze(1).to(float32))
    v_ptr,          # *float32,  [N, D] (v_cache.squeeze(1).to(float32))
    kv_indptr_ptr,  # *int32,    [B+1]
    kv_indices_ptr, # *int32,    [num_kv_indices]
    output_ptr,     # *bfloat16, [B, H, D]
    lse_ptr,        # *float32,  [B, H]
    sm_scale,       # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,           # num kv heads (e.g., 8)
    gqa_ratio: tl.constexpr,   # H // N, e.g., 4
):
    # Each program handles one (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Determine range of tokens for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    num_tokens = end - start

    # GQA mapping: query head h uses KV head kvh = h // gqa_ratio
    kvh = h // gqa_ratio

    # Initialize LSE accumulators
    max_logit = -float("inf")
    sum_exp = 0.0

    # First pass: compute max_logit and sum_exp over tokens
    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)

        # Load q[b, h, :] as float32: contiguous [B, H, D] => offset b*(H*D) + h*D
        q_off = b * (H * D) + h * D
        q_vec = tl.load(q_ptr + q_off, other=0.0).to(tl.float32)  # [D] as scalar vector via loop

        # Load k[kvh, tok_idx, :] as float32 from k_ptr [N, D]
        k_off = kvh * D + tok_idx * D
        k_vec = tl.load(k_ptr + k_off, other=0.0).to(tl.float32)  # [D]

        # Compute dot product
        dot = 0.0
        # Manually accumulate dot = sum_i q_vec[i] * k_vec[i]
        # Since D is constexpr (128), we can loop scalar-wise to avoid Triton type issues
        for i in range(D):
            dot += q_vec[i] * k_vec[i]

        logits_scaled = dot * sm_scale
        max_logit = tl.maximum(max_logit, logits_scaled)
        sum_exp += tl.exp(logits_scaled - max_logit)

        t += 1

    # Compute base-2 LSE
    ln2 = 0.6931471805599453  # float32 constant
    lse_val = max_logit + tl.log(sum_exp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute output[b, h, :] = sum_t softmax(logits_scaled) * v[kvh, tok_idx, :]
    # Initialize output accumulator as float32 for numerical stability
    acc = tl.zeros((D,), dtype=tl.float32)

    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)

        q_off = b * (H * D) + h * D
        q_vec = tl.load(q_ptr + q_off, other=0.0).to(tl.float32)

        k_off = kvh * D + tok_idx * D
        k_vec = tl.load(k_ptr + k_off, other=0.0).to(tl.float32)

        dot = 0.0
        for i in range(D):
            dot += q_vec[i] * k_vec[i]

        logits_scaled = dot * sm_scale
        attn = tl.exp(logits_scaled - max_logit) / sum_exp

        v_off = kvh * D + tok_idx * D
        v_vec = tl.load(v_ptr + v_off, other=0.0).to(tl.float32)  # [D]

        acc += attn * v_vec

        t += 1

    # Store output[b, h, :] as bfloat16
    out_off = b * (H * D) + h * D
    tl.store(output_ptr + out_off + tl.arange(0, D), acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be CUDA tensors."

        # Shapes
        B, H, D = q.shape
        # k_cache, v_cache: [P, 1, N, D] -> squeeze dim 1 to [N, D]
        k_squeezed = k_cache.squeeze(1).to(torch.float32)
        v_squeezed = v_cache.squeeze(1).to(torch.float32)

        device = q.device

        # Output and LSE
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # GQA ratio
        gqa_ratio = H // 8  # num_kv_heads == 8 (asserted in original)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)

        attention_bh_kernel[grid](
            q, k_squeezed, v_squeezed, kv_indptr, kv_indices, output, lse, sm_scale,
            B, H, D, 8, gqa_ratio,
            num_warps=1, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
