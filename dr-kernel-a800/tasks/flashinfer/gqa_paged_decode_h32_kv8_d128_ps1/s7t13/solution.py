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
    sm_scale,       # f32 scalar
    lse_ptr,        # *f32,  [B, Nq]
    B: tl.constexpr,           # batch size
    Nq: tl.constexpr,          # num query heads (32)
    Nkv: tl.constexpr,         # num kv heads (8)
    head_dim: tl.constexpr,    # D (128)
    gqa_ratio: tl.constexpr,   # Nq // Nkv (4)
    stride_q_b, stride_q_h, stride_q_d,
    # strides for k_ptr: [Np, Nkv, D]
    stride_k_p, stride_k_h, stride_k_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] as f32 vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, head_dim) * stride_q_d).to(tl.float32)  # [D] f32

    # Load token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h  # base for this idx and kv_head
        k_i = tl.load(k_base + tl.arange(0, head_dim) * stride_k_d).to(tl.float32)  # [D] f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
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
    sm_scale,       # f32 scalar
    lse_ptr,        # *f32,  [B, Nq] (not used for compute, but kept for signature)
    out_ptr,        # *bf16, [B, Nq, D]
    B: tl.constexpr,           # batch size
    Nq: tl.constexpr,          # num query heads (32)
    Nkv: tl.constexpr,         # num kv heads (8)
    head_dim: tl.constexpr,    # D (128)
    gqa_ratio: tl.constexpr,   # Nq // Nkv (4)
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_p, stride_k_h, stride_k_d,
    stride_v_p, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] as f32 vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, head_dim) * stride_q_d).to(tl.float32)  # [D] f32

    # Load token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Recompute m and sum_exp to get softmax (robust without storing)
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
        k_i = tl.load(k_base + tl.arange(0, head_dim) * stride_k_d).to(tl.float32)
        attn = tl.sum(q_vec * k_i, axis=0)
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # Output vector: accumulate soft_i * v_i[j]
    out_vec = tl.zeros((head_dim,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
        k_i = tl.load(k_base + tl.arange(0, head_dim) * stride_k_d).to(tl.float32)  # [D] f32
        attn = tl.sum(q_vec * k_i, axis=0)
        scaled = attn * sm_scale
        soft_i = tl.exp(scaled - m) / sum_exp  # scalar
        v_base = v_ptr + idx * stride_v_p + kv_head * stride_v_h
        v_i = tl.load(v_base + tl.arange(0, head_dim) * stride_v_d).to(tl.float32)  # [D] f32
        # out_vec += soft_i * v_i
        out_vec += soft_i * v_i
        i += 1

    # Store output as bfloat16
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    tl.store(out_base + tl.arange(0, head_dim) * stride_out_d, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure all tensors are on CUDA and contiguous
        device = q.device
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B, Nq, D = q.shape  # dynamic sizes, do not assert fixed values
        Np, Nk, Nkv, head_dim = k_cache.shape  # squeeze middle 1 is handled on host
        # Prepare output and lse
        output = torch.empty((B, Nq, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Strides for q
        stride_q_b, stride_q_h, stride_q_d = q.stride()
        # Strides for k_cache and v_cache (both are [Np, Nkv, D])
        stride_k_p, stride_k_h, stride_k_d = k_cache.stride()
        stride_v_p, stride_v_h, stride_v_d = v_cache.stride()
        # Strides for output
        stride_out_b, stride_out_h, stride_out_d = output.stride()

        # Launch Triton kernels: one program per (b, h)
        grid = (B * Nq,)
        # For lse kernel
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B=B, Nq=Nq, Nkv=Nkv, head_dim=head_dim, gqa_ratio=Nq // Nkv,
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            stride_k_p=stride_k_p, stride_k_h=stride_k_h, stride_k_d=stride_k_d,
            num_warps=4, num_stages=2
        )
        # For output kernel
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B=B, Nq=Nq, Nkv=Nkv, head_dim=head_dim, gqa_ratio=Nq // Nkv,
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            stride_k_p=stride_k_p, stride_k_h=stride_k_h, stride_k_d=stride_k_d,
            stride_v_p=stride_v_p, stride_v_h=stride_v_h, stride_v_d=stride_v_d,
            stride_out_b=stride_out_b, stride_out_h=stride_out_h, stride_out_d=stride_out_d,
            num_warps=4, num_stages=2
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
