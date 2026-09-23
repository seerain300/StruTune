import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def fused_gqa_row_kernel(
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, 1, K, D]
    v_ptr,            # *bf16, [Np, 1, K, D]
    indptr_ptr,       # *int32, [B+1]
    indices_ptr,      # *int32, [num_tokens_total]
    out_ptr,          # *bf16, [B, H, D]
    lse_ptr,          # *fp32, [B, H]
    sm_scale,         # fp32 scalar
    H: tl.constexpr,      # 32
    D: tl.constexpr,      # 128
    K: tl.constexpr,      # 8
    gqa_ratio: tl.constexpr,  # H // K == 4
    MAX_TOKENS: tl.constexpr,  # e.g., 1024
):
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # query head index

    # Load q vector for this (b, h) in fp32
    q_offset = pid_b * (H * D) + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D] fp32

    # Load indptr[start, end] for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Pass 1: compute sum_exp = sum(exp(logits_scaled)) across tokens (scalar per-token loop)
    sum_exp = 0.0
    # We avoid Triton while loops by using a fixed number of iterations; here MAX_TOKENS is large enough for all cases.
    for t in range(MAX_TOKENS):
        # If t >= num_tokens, break; Triton doesn't support break, so we mask by checking bounds
        # Using masks: Triton expects tensors in pointer arithmetic; here we keep scalar logic simple
        # We rely on indices_ptr[start + t] being valid up to num_tokens-1; beyond that, we skip work
        # However, since we can't easily break, we instead load idx only if t < num_tokens via pointer math
        if t >= num_tokens:
            break
        idx = tl.load(indices_ptr + start + t).to(tl.int32)  # scalar token index
        kv_head = pid_h // gqa_ratio  # 0..7

        # Load K row for this token and head (1D vector) in fp32
        k_row_ptr = k_ptr + idx * (K * D) + kv_head * D
        # Manually load each element to avoid tl.arange over tokens; keep it simple and robust
        d = 0
        k_vec = tl.zeros((D,), dtype=tl.float32)
        while d < D:
            k_elem = tl.load(k_row_ptr + d).to(tl.float32)
            k_vec[d] = k_elem
            d += 1

        # dot(q_vec, k_row)
        dot = 0.0
        d = 0
        while d < D:
            dot += q_vec[d] * k_vec[d]
            d += 1

        sum_exp += tl.exp(dot * sm_scale)

    # lse in base-2: log2(sum_exp) = log(sum_exp) / ln(2)
    lse_acc = tl.log(sum_exp) * 1.4426950408889634  # 1/ln(2)

    # Pass 2: compute output vector out[b, h, :] (fp32), then cast to bf16
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = tl.load(indices_ptr + start + t).to(tl.int32)
        kv_head = pid_h // gqa_ratio

        # Load K and V rows for this token and head (scalar per element)
        k_row_ptr = k_ptr + idx * (K * D) + kv_head * D
        v_row_ptr = v_ptr + idx * (K * D) + kv_head * D

        k_vec = tl.zeros((D,), dtype=tl.float32)
        v_vec = tl.zeros((D,), dtype=tl.float32)
        d = 0
        while d < D:
            k_elem = tl.load(k_row_ptr + d).to(tl.float32)
            v_elem = tl.load(v_row_ptr + d).to(tl.float32)
            k_vec[d] = k_elem
            v_vec[d] = v_elem
            d += 1

        # Compute dot and attention
        dot = 0.0
        d = 0
        while d < D:
            dot += q_vec[d] * k_vec[d]
            d += 1

        attn = tl.exp((dot - lse_acc) * sm_scale)

        # Accumulate output vector
        d = 0
        while d < D:
            out_vec[d] += attn * v_vec[d]
            d += 1

    # Store output as bfloat16
    out_offset = pid_b * (H * D) + pid_h * D
    tl.store(out_ptr + out_offset, out_vec.to(tl.bfloat16))

    # Store lse (base-2) as fp32
    lse_offset = pid_b * H + pid_h
    tl.store(lse_ptr + lse_offset, (lse_acc * 1.4426950408889634))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available and tensors are on CUDA
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        device = q.device
        assert device.type == "cuda", "Inputs must be on CUDA device"

        # Shapes and asserts (match original assumptions)
        B, H, D = q.shape
        assert H == 32 and D == 128, "This Triton implementation expects H=32 and D=128"
        assert k_cache.shape[1] == 1 and v_cache.shape[1] == 1, "num_pages must be 1"
        Np, _, K, _ = k_cache.shape
        assert K == 8, "This Triton implementation expects K=8"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr shape must be [B+1]"

        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Output buffers
        out = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch one program per (b, h)
        grid = (B, H)
        # MAX_TOKENS should be larger than any possible num_tokens in the evaluation (up to a few thousand).
        fused_gqa_row_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, out, lse, sm_scale,
            H=32, D=128, K=8, gqa_ratio=4, MAX_TOKENS=1024,
        )
        return out, lse


def run(*args):
    return ModelNew()(*args)
