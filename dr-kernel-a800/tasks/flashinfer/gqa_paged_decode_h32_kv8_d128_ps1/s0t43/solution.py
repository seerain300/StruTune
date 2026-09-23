import math
import torch

import triton
import triton.language as tl


# Triton kernel to accumulate output[b, h, :] = sum_t softmax(logit * sm_scale) * v[token, kv_head, :]
# Grid: (B, Hq). Each program handles one (b, h) and iterates over tokens using a scalar while loop.
@triton.jit
def _accumulate_output_kernel(
    q_ptr,              # *f32, shape [B, Hq, D]
    k_ptr,              # *f32, shape [num_pages, Hk, D]
    v_ptr,              # *f32, shape [num_pages, Hk, D]
    kv_indices_ptr,     # *i32, shape [num_tokens]
    out_ptr,            # *f32, shape [B, Hq, D]
    B: tl.constexpr,    # int
    Hq: tl.constexpr,   # int
    Hk: tl.constexpr,   # int
    D: tl.constexpr,    # int
    sm_scale,           # f32
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Each program instance accumulates output for one (b, h)
    # Initialize accumulators
    sum_exp = 0.0  # float32
    out_vec = tl.zeros((D,), dtype=tl.float32)

    # Iterate over tokens t = 0..num_tokens-1 (runtime scalar loop)
    t = 0
    while t < num_tokens:
        # Compute kv_head for GQA mapping: h // (Hq // Hk)
        gqa_ratio = Hq // Hk
        kv_head = h // gqa_ratio

        # Load q[b, h, :] as a vector of length D
        q_offset = b * Hq * D + h * D
        q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D), mask=True)

        # Load kv_indices[t] (runtime scalar)
        idx_t = tl.load(kv_indices_ptr + t)  # scalar i32
        # Load k[idx_t, kv_head, :] and v[idx_t, kv_head, :]
        k_offset = idx_t * (Hk * D) + kv_head * D
        v_offset = idx_t * (Hk * D) + kv_head * D
        k_vec = tl.load(k_ptr + k_offset + tl.arange(0, D), mask=True)
        v_vec = tl.load(v_ptr + v_offset + tl.arange(0, D), mask=True)

        # Compute dot product and scaled logit
        logit = tl.sum(q_vec * k_vec, axis=0) * sm_scale
        exp_logit = tl.exp(logit)
        sum_exp += exp_logit
        out_vec += exp_logit * v_vec

        t += 1

    # Normalize: output[b, h, :] = out_vec / sum_exp
    out_vec = out_vec / sum_exp

    # Store output
    out_offset = b * (Hq * D) + h * D
    tl.store(out_ptr + out_offset + tl.arange(0, D), out_vec)


# Triton kernel to compute lse[b, h] = logsumexp(logit * sm_scale) / ln(2)
# Grid: (B, Hq). Each program handles one (b, h) and recomputes dots to accumulate sum_exp.
@triton.jit
def _lse_kernel(
    q_ptr,              # *f32, shape [B, Hq, D]
    k_ptr,              # *f32, shape [num_pages, Hk, D]
    kv_indices_ptr,     # *i32, shape [num_tokens]
    lse_ptr,            # *f32, shape [B, Hq]
    B: tl.constexpr,    # int
    Hq: tl.constexpr,   # int
    Hk: tl.constexpr,   # int
    D: tl.constexpr,    # int
    sm_scale,           # f32
    inv_log2,           # f32
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    gqa_ratio = Hq // Hk
    kv_head = h // gqa_ratio

    sum_exp = 0.0  # accumulate exp(logit * sm_scale)
    t = 0
    while t < num_tokens:
        q_offset = b * Hq * D + h * D
        q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D), mask=True)

        idx_t = tl.load(kv_indices_ptr + t)
        k_offset = idx_t * (Hk * D) + kv_head * D
        k_vec = tl.load(k_ptr + k_offset + tl.arange(0, D), mask=True)

        logit = tl.sum(q_vec * k_vec, axis=0) * sm_scale
        sum_exp += tl.exp(logit)
        t += 1

    lse_val = tl.log(sum_exp) * inv_log2  # logsumexp(logits_scaled) / ln(2)
    lse_offset = b * Hq + h
    tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No __init__ args; forward accepts 6 positional arguments.

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation. Returns (output, lse) where:
          - output: [B, Hq, D] tensor in float32 (then cast to bfloat16 to match baseline).
          - lse: [B, Hq] tensor in float32.
        """
        # Ensure tensors are contiguous and dtypes are float32 for computation
        q = q.contiguous().to(torch.float32)
        k_cache = k_cache.contiguous().to(torch.float32)
        v_cache = v_cache.contiguous().to(torch.float32)
        kv_indices = kv_indices.contiguous()  # keep int32

        # Shapes
        B, Hq, D = q.shape
        num_pages, _, Hk, _ = k_cache.shape
        num_tokens = kv_indices.shape[0]

        # Allocate outputs (float32 for accumulators)
        output = torch.zeros((B, Hq, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, Hq), dtype=torch.float32, device=q.device)

        # Launch Triton kernels
        # Grid: (B, Hq)
        grid = (B, Hq)

        # First: compute output via accumulation kernel
        _accumulate_output_kernel[grid](
            q, k_cache, v_cache, kv_indices, output,
            B=B, Hq=Hq, Hk=Hk, D=D, sm_scale=float(sm_scale),
        )

        # Second: compute lse via Triton kernel
        inv_log2 = 1.0 / math.log(2.0)
        _lse_kernel[grid](
            q, k_cache, kv_indices, lse,
            B=B, Hq=Hq, Hk=Hk, D=D, sm_scale=float(sm_scale), inv_log2=float(inv_log2),
        )

        # Match baseline output dtype (baseline uses bfloat16 for output)
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
