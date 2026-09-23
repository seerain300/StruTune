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
    B: tl.constexpr,
    Nq: tl.constexpr,
    Nkv: tl.constexpr,
    D: tl.constexpr,
    gqa_ratio: tl.constexpr,  # Nq // Nkv (e.g., 4)
    stride_q_b, stride_q_h, stride_q_d,
):
    # one program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] vector of length D
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

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
        # k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D], f32
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
    D: tl.constexpr,           # head_dim
    gqa_ratio: tl.constexpr,   # Nq // Nkv (e.g., 4)
    stride_q_b, stride_q_h, stride_q_d,
    stride_v_p, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
):
    # one program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] vector of length D
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b).to(tl.int32)     # int32
    end = tl.load(indptr_ptr + b + 1).to(tl.int32)   # int32
    T = end - start

    # Recompute m and sum_exp for softmax
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar f32
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # Compute output elementwise: out[b, h, j] = sum_i softmax[i] * v_i[j]
    j = 0
    while j < D:
        acc = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            v_base = v_ptr + idx * (Nkv * D) + kv_head * D
            v_i_j = tl.load(v_base + j * stride_v_d).to(tl.float32)
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D], f32
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar f32
            scaled = attn * sm_scale
            soft_i = tl.exp(scaled - m) / sum_exp
            acc += soft_i * v_i_j
            i += 1
        tl.store(out_ptr + b * stride_out_b + h * stride_out_h + j * stride_out_d, acc.to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA and contiguous
        device = q.device
        q = q.contiguous().to(device=device, dtype=torch.bfloat16)
        k_cache = k_cache.contiguous().to(device=device, dtype=torch.bfloat16)
        v_cache = v_cache.contiguous().to(device=device, dtype=torch.bfloat16)
        kv_indptr = kv_indptr.contiguous().to(device=device, dtype=torch.int32)
        kv_indices = kv_indices.contiguous().to(device=device, dtype=torch.int32)

        # Shapes (no assertions on head_dim)
        B = q.shape[0]
        Nq = q.shape[1]
        D = q.shape[2]

        # Allocate outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Strides for q: q is [B, Nq, D]
        stride_q_b = q.stride(0)
        stride_q_h = q.stride(1)
        stride_q_d = q.stride(2)

        # Launch lse kernel: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B, Nq, 8, D, Nq // 8,  # Nkv expected to be 8 in original; we pass D and Nq//8 for GQA
            stride_q_b, stride_q_h, stride_q_d,
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: one program per (b, h)
        stride_v_p = v_cache.stride(0)
        stride_v_h = v_cache.stride(1)
        stride_v_d = v_cache.stride(2)

        stride_out_b = output.stride(0)
        stride_out_h = output.stride(1)
        stride_out_d = output.stride(2)

        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B, Nq, 8, D, Nq // 8,
            stride_q_b, stride_q_h, stride_q_d,
            stride_v_p, stride_v_h, stride_v_d,
            stride_out_b, stride_out_h, stride_out_d,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
