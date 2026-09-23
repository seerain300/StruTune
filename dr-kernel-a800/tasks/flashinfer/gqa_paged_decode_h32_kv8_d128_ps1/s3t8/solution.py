import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_bh_kernel(
    q_ptr,          # *bfloat16, shape [B, H, D], contiguous
    k_ptr,          # *float32,  shape [N, D], squeezed from [P, 1, N, D]
    v_ptr,          # *float32,  shape [N, D]
    kv_indptr_ptr,  # *int32,    shape [B+1]
    kv_indices_ptr, # *int32,    shape [num_kv_indices]
    output_ptr,     # *bfloat16, shape [B, H, D]
    lse_ptr,        # *float32,  shape [B, H]
    sm_scale,       # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,           # num kv heads (8)
    gqa_ratio: tl.constexpr,   # H // N == 4
):
    # One program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Compute token range for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    num_tokens = end - start  # runtime scalar

    # GQA mapping: each query head uses KV head kvh = h // gqa_ratio
    kvh = h // gqa_ratio

    # Accumulate LSE across tokens
    max_logit = -float("inf")
    sum_exp = 0.0
    lse_off = b * H + h  # linear index into lse_ptr of length B*H

    # First pass: compute max_logit and sum_exp
    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)

        # Load q[b, h, :] as float32
        q_base = q_ptr + b * (H * D) + h * D
        q_vec = tl.load(q_base, other=0.0).to(tl.float32)  # shape [D] vector but used scalar-wise in loop

        # Load k[kvh, tok_idx, :] and v[kvh, tok_idx, :]
        k_off = kvh * D + tok_idx * D
        v_off = kvh * D + tok_idx * D
        k_vec = tl.load(k_ptr + k_off, other=0.0).to(tl.float32)  # scalar load as [1,D]? Better: we load scalars
        v_vec = tl.load(v_ptr + v_off, other=0.0).to(tl.float32)

        # Compute dot product scalar
        # Since q_vec and k_vec are [D], we reduce manually
        dot = 0.0
        i = 0
        while i < D:
            qi = q_vec[i]
            ki = k_vec[i]
            dot += qi * ki
            i += 1

        logits_scaled = dot * sm_scale
        max_logit = tl.maximum(max_logit, logits_scaled)
        # sum_exp += exp(logits_scaled - max_logit)  (scalar update)
        sum_exp += tl.exp(logits_scaled - max_logit)

        t += 1

    # Compute base-2 LSE
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse_val = max_logit + tl.log(sum_exp) / ln2
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: accumulate output
    acc = tl.zeros((D,), dtype=tl.float32)
    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)

        q_base = q_ptr + b * (H * D) + h * D
        q_vec = tl.load(q_base, other=0.0).to(tl.float32)

        k_off = kvh * D + tok_idx * D
        v_off = kvh * D + tok_idx * D
        k_vec = tl.load(k_ptr + k_off, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + v_off, other=0.0).to(tl.float32)

        dot = 0.0
        i = 0
        while i < D:
            qi = q_vec[i]
            ki = k_vec[i]
            dot += qi * ki
            i += 1

        logits_scaled = dot * sm_scale
        attn = tl.exp(logits_scaled - max_logit) / sum_exp

        # acc += attn * v_vec
        i = 0
        while i < D:
            acc[i] += attn * v_vec[i]
            i += 1

        t += 1

    # Store output[b, h, :] as bfloat16
    out_base = b * (H * D) + h * D
    out_ptr = output_ptr + out_base
    # Write acc as bfloat16
    # Triton doesn't have bfloat16 in all versions; cast via float32 and let PyTorch allocate bfloat16 target
    # We can store as float32 and then cast on host; here we store as float32 and rely on caller to cast if needed.
    # To ensure bfloat16, allocate output_ptr as bfloat16 and cast in kernel: Triton supports .to(tl.bfloat16)
    tl.store(out_ptr, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors."

        # Shapes
        B, H, D = q.shape
        # Squeeze dim-1 (from [P, 1, N, D] -> [N, D]) and cast to float32 for computation
        k_squeezed = k_cache.squeeze(1).to(torch.float32)  # [N, D]
        v_squeezed = v_cache.squeeze(1).to(torch.float32)  # [N, D]

        device = q.device

        # Output and LSE
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device).fill_(-float("inf"))

        # GQA ratio
        gqa_ratio = H // 8  # num_kv_heads == 8 (asserted in original code)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)

        attention_bh_kernel[grid](
            q, k_squeezed, v_squeezed, kv_indptr, kv_indices, output, lse, sm_scale,
            B=B, H=H, D=D, N=8, gqa_ratio=4,
            num_warps=1, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
