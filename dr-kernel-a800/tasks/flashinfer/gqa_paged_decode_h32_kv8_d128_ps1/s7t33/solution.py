import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    lse_ptr,        # *f32,  [B, Nq]
    B: tl.int32,
    Nq: tl.int32,
    Nkv: tl.int32,
    D: tl.int32,
    stride_q_b, stride_q_h, stride_q_d,    # strides for q
    kv_head: tl.int32,                     # precomputed grouped kv head for this (b,h)
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq

    # Number of tokens for this batch element
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Accumulate m and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        # Base pointer for k at this idx and kv_head, then load k_i vector of length D
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32

        # Load q[b, h] vector and compute dot with k_i
        q_base = q_ptr + b * stride_q_b + h * stride_q_h
        q_vec = tl.load(q_base + tl.arange(0, D)).to(tl.float32)
        attn = tl.zeros((), dtype=tl.float32)
        j = 0
        while j < D:
            attn += q_vec[j] * k_i[j]
            j += 1

        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # lse = log(sum_exp) + m; divide by log(2) to match original
    log2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) + m
    tl.store(lse_ptr + b * Nq + h, lse_val / log2)


@triton.jit
def output_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D]
    v_ptr,          # *bf16, [Np, Nkv, D]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    lse_ptr,        # *f32,  [B, Nq]
    out_ptr,        # *bf16, [B, Nq, D]
    B: tl.int32,
    Nq: tl.int32,
    Nkv: tl.int32,
    D: tl.int32,
    stride_q_b, stride_q_h, stride_q_d,    # strides for q
    stride_out_b, stride_out_h, stride_out_d,
    kv_head: tl.int32,                     # precomputed grouped kv head for this (b,h)
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Recompute m and sum_exp (from lse_ptr), but if lse_ptr not used, recompute here
    # To be safe, we recompute m and sum_exp here to ensure correctness
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32

        q_base = q_ptr + b * stride_q_b + h * stride_q_h
        q_vec = tl.load(q_base + tl.arange(0, D)).to(tl.float32)
        attn = tl.zeros((), dtype=tl.float32)
        j = 0
        while j < D:
            attn += q_vec[j] * k_i[j]
            j += 1

        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # Now compute output vector out[b, h, :]
    # Initialize output vector
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    j = 0
    while j < D:
        out_val = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            # Load k_i and v_i vectors
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)
            v_base = v_ptr + idx * (Nkv * D) + kv_head * D
            v_i = tl.load(v_base + tl.arange(0, D)).to(tl.float32)

            # q_vec[j] * k_i[j]
            q_base = q_ptr + b * stride_q_b + h * stride_q_h
            q_vec = tl.load(q_base + j * stride_q_d).to(tl.float32)  # scalar
            attn_j = q_vec * k_i[j]

            scaled_j = attn_j * sm_scale
            soft_j = tl.exp(scaled_j - m) / sum_exp

            # out_val += soft_j * v_i[j]
            out_val += soft_j * v_i[j]
            i += 1
        tl.store(out_base + j * stride_out_d, out_val.to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Inputs must be on CUDA"
        q = q.contiguous()
        k_cache = k_cache.contiguous().to(torch.bfloat16)
        v_cache = v_cache.contiguous().to(torch.bfloat16)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        B, Nq, D = q.shape
        Np, _, Nkv, _ = k_cache.shape

        # Precompute grouped kv head per (b, h) as in original: kv_head = h // (Nq // Nkv)
        gqa_ratio = Nq // Nkv
        kv_heads = torch.arange(Nq, device=device).view(1, Nq) // gqa_ratio  # shape [1, Nq], we'll index by b later
        # For simplicity, since B is passed, we can expand: kv_heads = (torch.arange(Nq, device=device) // gqa_ratio).unsqueeze(0).expand(B, Nq)
        # But Triton kernels expect scalar per (b,h), so we'll pass 1* vector per launch; it's the same per b.
        # We pass kv_head as scalar per program_id; no need to build a tensor, compute on host per b is fine, but Triton expects compile-time? No, we can pass as runtime int.

        # Allocate outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Strides for q and output (in elements)
        stride_q_b = q.stride(0)
        stride_q_h = q.stride(1)
        stride_q_d = q.stride(2)

        # Strides for output (we can get strides from output tensor)
        # We need strides in elements; output is contiguous bfloat16
        # out strides: stride_out_b = output.stride(0) * element_size ? No, torch strides are in elements already.
        # For contiguous [B, Nq, D]: stride_out_b = Nq*D, stride_out_h = D, stride_out_d = 1
        # Let Triton derive by passing tensor strides, but Triton expects strides in elements. We can compute:
        # For contiguous, stride_out_b = Nq*D, stride_out_h = D, stride_out_d = 1
        # However, better to use tensor.stride() in elements directly.
        # We'll create small tensors to get strides if needed, but here output is contiguous.
        # Compute strides explicitly:
        stride_out_b = output.stride(0)
        stride_out_h = output.stride(1)
        stride_out_d = output.stride(2)

        # Launch lse kernel: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B, Nq, Nkv, D,
            stride_q_b, stride_q_h, stride_q_d,
            # kv_head: we don't have per-(b) change in grouped mapping; h // gqa_ratio is constant per h
            h // gqa_ratio,  # passing h // gqa_ratio as scalar for all b,h; this is fine because gqa_ratio is per-Nq//Nkv, and h is same for all b
            num_warps=4, num_stages=2,
        )

        # For output kernel, we need kv_head per (b,h); since grouped mapping is per-h, same scalar is fine.
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B, Nq, Nkv, D,
            stride_q_b, stride_q_h, stride_q_d,
            stride_out_b, stride_out_h, stride_out_d,
            h // gqa_ratio,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
