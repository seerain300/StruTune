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
    B: tl.constexpr,          # batch size
    Nq: tl.constexpr,         # num_qo_heads
    Nkv: tl.constexpr,        # num_kv_heads
    D: tl.constexpr,          # head_dim
    gqa_ratio: tl.constexpr,  # Nq // Nkv (e.g., 4)
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] vector of length D
    q_base = q_ptr + b * (Nq * D) + h * D
    q_vec = tl.load(q_base + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b).to(tl.int32)     # int32
    end = tl.load(indptr_ptr + b + 1).to(tl.int32)   # int32
    T = end - start

    # Accumulate m and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar f32
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
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
    B: tl.constexpr,
    Nq: tl.constexpr,
    Nkv: tl.constexpr,
    D: tl.constexpr,
    gqa_ratio: tl.constexpr,  # Nq // Nkv (e.g., 4)
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] vector of length D
    q_base = q_ptr + b * (Nq * D) + h * D
    q_vec = tl.load(q_base + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b).to(tl.int32)     # int32
    end = tl.load(indptr_ptr + b + 1).to(tl.int32)   # int32
    T = end - start

    # Recompute m and sum_exp for softmax normalization
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar f32
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # Compute output vector: out[b, h, j] = sum_i softmax[i] * v_i[j]
    out_base = out_ptr + b * (Nq * D) + h * D
    j = 0
    while j < D:
        acc = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            v_base = v_ptr + idx * (Nkv * D) + kv_head * D
            v_i = tl.load(v_base + j).to(tl.float32)  # scalar f32
            soft_i = tl.exp((tl.load(indices_ptr + start + i).to(tl.float32) * sm_scale - m) / sum_exp)  # softmax[i]
            # Correction: softmax[i] = exp(scaled[i] - m) / sum_exp, but we don't have scaled[i] here.
            # We need to recompute scaled[i] from k_cache:
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar
            scaled_i = attn * sm_scale
            soft_i = tl.exp(scaled_i - m) / sum_exp
            acc += soft_i * v_i
            i += 1
        tl.store(out_base + j, acc.to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on same CUDA device and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be on CUDA device"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes
        B, Nq, D = q.shape
        Np, one, Nkv, _ = k_cache.shape
        assert one == 1, "k_cache expected shape [Np, 1, Nkv, D]"
        assert v_cache.shape == (Np, 1, Nkv, D), "v_cache must have same shape as k_cache"

        # Output tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=q.device)

        # GQA ratio
        gqa_ratio = Nq // Nkv

        # Launch Triton kernels
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache,
            kv_indptr, kv_indices,
            sm_scale,
            lse,
            B, Nq, Nkv, D,
            gqa_ratio,
            num_warps=4, num_stages=2,
        )

        output_kernel[grid](
            q, k_cache, v_cache,
            kv_indptr, kv_indices,
            sm_scale,
            lse, output,
            B, Nq, Nkv, D,
            gqa_ratio,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
