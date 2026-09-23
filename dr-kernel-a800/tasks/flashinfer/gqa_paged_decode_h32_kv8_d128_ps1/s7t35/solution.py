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
    gqa_ratio: tl.constexpr,  # Nq // Nkv (should be 4)
    stride_q_b, stride_q_h, stride_q_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Accumulate m and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D  # k_ptr is [Np, Nkv, D]
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
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
    B: tl.constexpr,          # batch size
    Nq: tl.constexpr,         # num_qo_heads
    Nkv: tl.constexpr,        # num_kv_heads
    D: tl.constexpr,          # head_dim
    gqa_ratio: tl.constexpr,  # Nq // Nkv
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_p, stride_k_h, stride_k_d,
    stride_v_p, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h]
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Recompute m and sum_exp (robust and simple)
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    log2 = 0.6931471805599453

    # Fill output vector out[b, h, :] in float32, then cast to bfloat16
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    for j in range(0, D):
        acc = tl.zeros((), dtype=tl.float32)
        for i in range(0, T):
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar
            scaled = attn * sm_scale
            soft = tl.exp(scaled - m) / sum_exp  # scalar
            v_base = v_ptr + idx * (Nkv * D) + kv_head * D
            v_i = tl.load(v_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
            acc += soft * v_i[j + 0]  # scalar
        tl.store(out_base + j * stride_out_d, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous
        device = q.device
        q = q.contiguous().to(torch.bfloat16).to(device)
        k_cache = k_cache.contiguous().to(torch.bfloat16).to(device)
        v_cache = v_cache.contiguous().to(torch.bfloat16).to(device)
        kv_indptr = kv_indptr.contiguous().to(torch.int32).to(device)
        kv_indices = kv_indices.contiguous().to(torch.int32).to(device)

        # Shapes (keep original assertions for correctness checks)
        B, Nq, D = q.shape
        Np, _, Nkv, _ = k_cache.shape
        assert Nq == 32, "num_qo_heads must be 32"
        assert Nkv == 8, "num_kv_heads must be 8"
        assert D == 128, "head_dim must be 128"

        # Prepare outputs
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)

        # GQA ratio
        gqa_ratio = Nq // Nkv

        # Compute strides (in elements)
        stride_q_b = q.stride(0)
        stride_q_h = q.stride(1)
        stride_q_d = q.stride(2)

        stride_k_p = k_cache.stride(0)
        stride_k_h = k_cache.stride(1)
        stride_k_d = k_cache.stride(2)

        stride_v_p = v_cache.stride(0)
        stride_v_h = v_cache.stride(1)
        stride_v_d = v_cache.stride(2)

        stride_out_b = output.stride(0)
        stride_out_h = output.stride(1)
        stride_out_d = output.stride(2)

        # Launch lse kernel: one program per (b, h)
        grid_lse = (B * Nq,)
        lse_kernel[grid_lse](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B, Nq, Nkv, D, gqa_ratio,
            stride_q_b, stride_q_h, stride_q_d,
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: one program per (b, h)
        grid_out = (B * Nq,)
        output_kernel[grid_out](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B, Nq, Nkv, D, gqa_ratio,
            stride_q_b, stride_q_h, stride_q_d,
            stride_k_p, stride_k_h, stride_k_d,
            stride_v_p, stride_v_h, stride_v_d,
            stride_out_b, stride_out_h, stride_out_d,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
