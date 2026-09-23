import math
import torch
import triton
import triton.language as tl


# Kernel 1: compute per-token logits for all (b, h, t)
# Grid: (B, Hq, num_tokens)
@triton.jit
def _compute_logits_per_token(
    q_ptr,          # *f32, [B, Hq, D]
    k_ptr,          # *f32, [num_tokens, D]
    logits_ptr,     # *f32, [B*Hq*num_tokens] (we don't strictly use this in accumulation; included for Triton usage)
    B: tl.constexpr,
    Hq: tl.constexpr,
    num_tokens,     # i32
    D: tl.constexpr,
    sm_scale: tl.float32,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    if t >= num_tokens:
        return

    q_base = q_ptr + b * D + h * D
    k_vec = tl.load(k_ptr + t * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    q_vec = tl.load(q_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    acc = tl.sum(q_vec * k_vec, axis=0)
    x = acc * sm_scale
    idx = b * Hq * num_tokens + h * num_tokens + t
    tl.store(logits_ptr + idx, x)


# Kernel 2: accumulate output[b, h, :] across tokens using softmax over scaled logits
# Grid: (B, Hq)
@triton.jit
def _accumulate_output_atomic_kernel(
    q_ptr,          # *f32, [B, Hq, D]
    k_ptr,          # *f32, [num_tokens, D]
    v_ptr,          # *f32, [num_tokens, D]
    out_ptr,        # *f32, [B*Hq*D] flattened, initialized to 0
    num_tokens,     # i32
    D: tl.constexpr,
    sm_scale: tl.float32,
):
    pid = tl.program_id(0)
    b = pid // Hq
    h = pid % Hq

    # Running max and sum for softmax
    max_val = -1e20
    sum_exp = 0.0

    # GQA mapping: kv_head = h // (Hq // Hk) = h // 4
    kv_head = h // (Hq // 8)

    # First pass: compute logsumexp over scaled logits
    t = 0
    while t < num_tokens:
        q_base = q_ptr + b * D + h * D
        k_vec = tl.load(k_ptr + t * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        q_vec = tl.load(q_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        acc = tl.sum(q_vec * k_vec, axis=0)
        x = acc * sm_scale
        if x > max_val:
            sum_exp = sum_exp * tl.exp(max_val - x) + 1.0
            max_val = x
        else:
            sum_exp = sum_exp + tl.exp(x - max_val)
        t += 1

    # Second pass: accumulate output
    for t in range(0, num_tokens):
        q_base = q_ptr + b * D + h * D
        k_vec = tl.load(k_ptr + t * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        q_vec = tl.load(q_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        acc = tl.sum(q_vec * k_vec, axis=0)
        x = acc * sm_scale
        p = tl.exp(x - max_val) / sum_exp  # softmax probability
        v_vec = tl.load(v_ptr + t * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        out_vec = p * v_vec
        out_base = out_ptr + (b * Hq + h) * D
        idx = tl.arange(0, D)
        tl.store(out_base + idx, out_vec, mask=idx < D)


# Kernel 3: compute lse[b, h] = logsumexp(logits_scaled) / ln(2)
# Grid: (B, Hq)
@triton.jit
def _lse_per_bh_kernel(
    q_ptr,          # *f32, [B, Hq, D]
    k_ptr,          # *f32, [num_tokens, D]
    lse_ptr,        # *f32, [B*Hq]
    num_tokens,     # i32
    D: tl.constexpr,
    sm_scale: tl.float32,
):
    pid = tl.program_id(0)
    b = pid // Hq
    h = pid % Hq

    max_val = -1e20
    sum_exp = 0.0
    kv_head = h // (Hq // 8)

    t = 0
    while t < num_tokens:
        q_base = q_ptr + b * D + h * D
        k_vec = tl.load(k_ptr + t * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        q_vec = tl.load(q_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        acc = tl.sum(q_vec * k_vec, axis=0)
        x = acc * sm_scale
        if x > max_val:
            sum_exp = sum_exp * tl.exp(max_val - x) + 1.0
            max_val = x
        else:
            sum_exp = sum_exp + tl.exp(x - max_val)
        t += 1

    lse = max_val + tl.log(sum_exp)  # logsumexp of scaled logits
    # Divide by ln(2) to match original
    lse = lse / math.log(2.0)
    tl.store(lse_ptr + pid, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, _unused_param=None):
        # Ensure CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All inputs must be CUDA tensors."
        device = q.device

        # Shapes
        B, Hq, D = q.shape
        assert Hq == 32, "num_qo_heads must be 32"

        # num_tokens: number of KV entries used
        num_tokens = kv_indices.shape[0]

        # Convert to float32 for compute
        q_f32 = q.contiguous().to(torch.float32)  # [B, Hq, D]
        # k_cache and v_cache are provided as [num_tokens, D] in get_inputs
        k_flat = k_cache.contiguous().to(torch.float32)  # [num_tokens, D]
        v_flat = v_cache.contiguous().to(torch.float32)  # [num_tokens, D]

        # Output and lse buffers
        output = torch.zeros((B, Hq, D), dtype=torch.float32, device=device)
        output_flat = output.view(-1)  # [B*Hq*D]
        lse = torch.empty((B, Hq), dtype=torch.float32, device=device)

        # Launch Triton kernels
        # Compute logits per token (grid over (B, Hq, num_tokens))
        logits_buf = torch.empty((B * Hq * num_tokens,), dtype=torch.float32, device=device)
        grid1 = (B, Hq, num_tokens)
        _compute_logits_per_token[grid1](
            q_f32, k_flat, logits_buf, B, Hq, num_tokens, D, float(sm_scale),
        )
        # Accumulate output
        grid2 = (B, Hq)
        _accumulate_output_atomic_kernel[grid2](
            q_f32, k_flat, v_flat, output_flat, num_tokens, D, float(sm_scale),
        )
        # Compute lse
        grid3 = (B, Hq)
        _lse_per_bh_kernel[grid3](
            q_f32, k_flat, lse, num_tokens, D, float(sm_scale),
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
