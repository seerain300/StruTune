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
    B: tl.constexpr,      # (not strictly needed) left for completeness
    H: tl.constexpr,      # 32
    D: tl.constexpr,      # 128
    K: tl.constexpr,      # 8
    gqa_ratio: tl.constexpr,  # H // K == 4
):
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # query head index

    # Load q vector for this (b, h) in fp32
    q_offset = pid_b * (H * D) + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # Load indptr[start, end] for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Pass 1: compute sum_exp = sum(exp(logits_scaled)) across tokens for lse
    sum_exp = 0.0
    t = 0
    while t < num_tokens:
        idx = tl.load(indices_ptr + start + t).to(tl.int32)  # scalar token index
        kv_head = pid_h // gqa_ratio  # 0..7

        # Compute dot(q_vec, k_row) with scalar loads (avoid vectorized loads to prevent JIT issues)
        dot = 0.0
        # q_vec is [D] fp32, we multiply each dim by corresponding k_dim and accumulate
        for d in range(D):
            # k layout: [Np, 1, K, D] -> for fixed middle=1, address = idx * (K*D) + kv_head * D + d
            k_val = tl.load(k_ptr + idx * (K * D) + kv_head * D + d).to(tl.float32)
            dot += q_vec[d] * k_val

        # Accumulate sum of exp(logits_scaled) where logits_scaled = dot * sm_scale (sm_scale is passed as arg)
        sum_exp += tl.exp(dot * sm_scale)
        t += 1

    # lse in base-2: log2(sum_exp) = log(sum_exp) / ln(2)
    lse_acc = tl.log(sum_exp) * 1.4426950408889634  # 1/ln(2)

    # Pass 2: compute output vector out[b, h, :] (fp32), then cast to bf16
    out_vec = tl.zeros((D,), dtype=tl.float32)
    t = 0
    while t < num_tokens:
        idx = tl.load(indices_ptr + start + t).to(tl.int32)
        kv_head = pid_h // gqa_ratio

        # Compute dot(q_vec, k_row) again
        dot = 0.0
        for d in range(D):
            k_val = tl.load(k_ptr + idx * (K * D) + kv_head * D + d).to(tl.float32)
            dot += q_vec[d] * k_val

        attn = tl.exp((dot - lse_acc) * sm_scale)  # softmax probability for this token

        # Accumulate attn * v_row into out_vec
        for d in range(D):
            v_val = tl.load(v_ptr + idx * (K * D) + kv_head * D + d).to(tl.float32)
            out_vec[d] += attn * v_val
        t += 1

    # Store output as bfloat16
    out_offset = pid_b * (H * D) + pid_h * D
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

        # Shapes and basic checks (minimal)
        B, H, D = q.shape
        Np, _, K, _ = k_cache.shape
        assert H == 32 and D == 128, "This Triton implementation expects H=32 and D=128"
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
        fused_gqa_row_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, out, lse,
            B=B, H=H, D=D, K=K, gqa_ratio=H // K, sm_scale=float(sm_scale)
        )
        return out, lse


def run(*args):
    return ModelNew()(*args)
