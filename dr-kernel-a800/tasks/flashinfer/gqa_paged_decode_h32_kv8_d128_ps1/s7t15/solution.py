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
    D: tl.constexpr,           # head_dim, compile-time for tl.arange
    gqa_ratio: tl.constexpr,   # Nq // Nkv (e.g., 4)
    stride_q_b, stride_q_h, stride_q_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
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
    D: tl.constexpr,           # head_dim, compile-time for tl.arange
    gqa_ratio: tl.constexpr,   # Nq // Nkv (e.g., 4)
    stride_q_b, stride_q_h, stride_q_d,
    stride_v_p, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
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

    # Recompute m and sum_exp
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

    # Compute output vector: out[b, h, :] = sum_i softmax[i] * v_i
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    out_vec = tl.zeros((D,), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index
        v_base = v_ptr + idx * (Nkv * D) + kv_head * D
        v_i = tl.load(v_base + tl.arange(0, D) * 1).to(tl.float32)  # [D], f32

        # Recompute attn for softmax scaling
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)
        attn_i = tl.sum(q_vec * k_i, axis=0)
        scaled_i = attn_i * sm_scale
        softmax_i = tl.exp(scaled_i - m) / sum_exp

        out_vec += softmax_i * v_i
        i += 1

    # Store as bfloat16
    tl.store(out_base + tl.arange(0, D) * stride_out_d, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure all inputs are on CUDA and contiguous
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        B = q.shape[0]
        Nq = q.shape[1]
        D = q.shape[2]
        Np = k_cache.shape[0]
        Nkv = k_cache.shape[2]
        T_total = kv_indices.shape[0]

        # Prepare outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch lse kernel: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, float(sm_scale), lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=(Nq // Nkv),
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: one program per (b, h)
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, float(sm_scale), lse, output,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=(Nq // Nkv),
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_v_p=v_cache.stride(0), stride_v_h=v_cache.stride(1), stride_v_d=v_cache.stride(2),
            stride_out_b=output.stride(0), stride_out_h=output.stride(1), stride_out_d=output.stride(2),
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
