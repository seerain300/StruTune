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
    B, Nq, Nkv, D,  # int32 runtime args
):
    # one program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq

    # window [start, end)
    start = tl.load(indptr_ptr + b)        # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # GQA mapping
    gqa_ratio = Nq // Nkv
    kv_head = h // gqa_ratio

    # load q[b, h] as f32
    # q is assumed contiguous in D: q[b, h, d] = q_ptr + b*Nq*D + h*D + d
    q_vec = tl.load(q_ptr + b * Nq * D + h * D + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        # load k_i (length D) as f32
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # lse = log(sum_exp) + m; divide by log(2)
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
    B, Nq, Nkv, D,  # int32 runtime args
):
    # one program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq

    # window [start, end)
    start = tl.load(indptr_ptr + b)        # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # GQA mapping
    gqa_ratio = Nq // Nkv
    kv_head = h // gqa_ratio

    # load q[b, h]
    q_vec = tl.load(q_ptr + b * Nq * D + h * D + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # Recompute m and sum_exp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # Compute output vector out[b, h, :] in f32, then store as bfloat16
    out_vec = tl.zeros((D,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        v_base = v_ptr + idx * (Nkv * D) + kv_head * D
        v_i = tl.load(v_base + tl.arange(0, D)).to(tl.float32)  # [D], f32

        # Recompute attn for softmax weight
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        soft = tl.exp(scaled - m) / sum_exp
        out_vec += soft * v_i
        i += 1

    # Store output as bfloat16
    tl.store(out_ptr + b * Nq * D + h * D + tl.arange(0, D), out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and contiguity
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA."
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B, Nq, D = q.shape
        # k_cache shape is [Np, Nkv, D]
        Np, Nkv, _ = k_cache.shape

        # Output tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=q.device)

        # Launch Triton kernels: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B, Nq, Nkv, D,
            num_warps=4, num_stages=2,
        )

        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B, Nq, Nkv, D,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
