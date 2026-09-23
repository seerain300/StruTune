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
):
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # query head index

    # Load q vector for this (b, h) in fp32: q[b, h, :]
    q_offset = pid_b * (H * D) + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D] fp32

    # Load indptr[start, end] for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Pass 1: compute sum_exp = sum(exp(logits_scaled)) across tokens
    sum_exp = 0.0
    t = 0
    while t < num_tokens:
        idx = tl.load(indices_ptr + start + t).to(tl.int32)  # scalar token index
        kv_head = pid_h // gqa_ratio  # 0..7

        # Compute dot(q_vec, k[idx, kv_head, :]) over D
        dot = 0.0
        d = 0
        while d < D:
            q_d = q_vec[d]
            k_d = tl.load(k_ptr + idx * (K * D) + kv_head * D + d).to(tl.float32)
            dot += q_d * k_d
            d += 1

        # Accumulate exp of scaled logits
        sum_exp += tl.exp(dot * sm_scale)
        t += 1

    # lse in base-2: log2(sum_exp) = log(sum_exp) / ln(2)
    lse_acc = tl.log(sum_exp) * 1.4426950408889634  # 1/ln(2) = 1.442695...

    # Pass 2: compute output vector out[b, h, :] (fp32), then cast to bf16
    out_vec = tl.zeros((D,), dtype=tl.float32)
    t = 0
    while t < num_tokens:
        idx = tl.load(indices_ptr + start + t).to(tl.int32)
        kv_head = pid_h // gqa_ratio

        # Compute dot(q_vec, k[idx, kv_head, :]) over D
        dot = 0.0
        d = 0
        while d < D:
            q_d = q_vec[d]
            k_d = tl.load(k_ptr + idx * (K * D) + kv_head * D + d).to(tl.float32)
            dot += q_d * k_d
            d += 1

        attn = tl.exp((dot - lse_acc) * sm_scale)

        # Accumulate attn * v[idx, kv_head, :] over D
        d = 0
        while d < D:
            v_d = tl.load(v_ptr + idx * (K * D) + kv_head * D + d).to(tl.float32)
            out_vec[d] += attn * v_d
            d += 1

        t += 1

    # Store output as bfloat16
    out_offset = pid_b * H * D + pid_h * D
    tl.store(out_ptr + out_offset, out_vec.to(tl.bfloat16))

    # Store lse (base-2)
    lse_offset = pid_b * H + pid_h
    tl.store(lse_ptr + lse_offset, lse_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available and tensors are on CUDA
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        device = q.device
        assert device.type == "cuda", "Inputs must be on CUDA device"

        # Shapes and asserts (same as original constraints)
        B, H, D = q.shape
        assert H == 32 and D == 128, "This Triton implementation expects H=32 and D=128"
        assert k_cache.shape[1] == 1 and v_cache.shape[1] == 1, "num_pages must be 1 (indptr length indicates batch, but cache has 1 page)"
        Np, _, K, _ = k_cache.shape
        assert K == 8, "This Triton implementation expects K=8"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr shape must be [B+1]"
        assert kv_indices.shape[0] == int(kv_indptr[-1].item()), "kv_indices total must equal last indptr"

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
        fused_gqa_row_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, out, lse,
            sm_scale,  # fp32 scalar
            H=H, D=D, K=K, gqa_ratio=H // K,
        )
        return out, lse


def run(*args):
    return ModelNew()(*args)
