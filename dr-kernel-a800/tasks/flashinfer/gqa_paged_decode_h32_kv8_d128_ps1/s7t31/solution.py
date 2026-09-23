import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D]
    v_ptr,          # *bf16, [Np, Nkv, D] (not used in lse, kept for signature symmetry)
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    lse_ptr,        # *f32,  [B, Nq]
    B: tl.int32,    # batch size (runtime)
    Nq: tl.int32,   # num_qo_heads (runtime)
    Nkv: tl.int32,  # num_kv_heads (runtime)
    D: tl.int32,    # head_dim (runtime)
    gqa_ratio: tl.int32,  # Nq // Nkv (runtime)
    stride_q_b, stride_q_h, stride_q_d,  # strides for q
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # Accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        # Pointer to k[idx, kv_head, :]
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar f32
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
    B: tl.int32,    # batch size (runtime)
    Nq: tl.int32,   # num_qo_heads (runtime)
    Nkv: tl.int32,  # num_kv_heads (runtime)
    D: tl.int32,    # head_dim (runtime)
    gqa_ratio: tl.int32,  # Nq // Nkv (runtime)
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_p, stride_k_h, stride_k_d,
    stride_v_p, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # Recompute m and sum_exp for softmax
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # Accumulate output vector [D]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)
        scaled = attn * sm_scale
        soft_i = tl.exp(scaled - m) / sum_exp  # scalar
        v_base = v_ptr + idx * (Nkv * D) + kv_head * D
        v_i = tl.load(v_base + tl.arange(0, D)).to(tl.float32)  # [D]
        # Accumulate elementwise
        out_vec += soft_i * v_i
        i += 1

    # Store as bfloat16
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    tl.store(out_base + tl.arange(0, D) * stride_out_d, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguity
        device = q.device
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Cast compute types: Triton will load as needed, but ensure pointers are bfloat16
        # (PyTorch bfloat16 tensors are fine; Triton loads/store in their dtype.)
        # We do not cast here; we let Triton load/store as-is.

        B = q.shape[0]
        Nq = q.shape[1]
        # head_dim D
        D = q.shape[2]
        Nkv = k_cache.shape[2]  # num_kv_heads

        # Output tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch Triton kernels: one program per (b, h)
        grid = (B * Nq,)

        # Strides for q
        stride_q_b = q.stride(0)
        stride_q_h = q.stride(1)
        stride_q_d = q.stride(2)

        # Strides for k, v (they are [Np, Nkv, D])
        stride_k_p = k_cache.stride(0)
        stride_k_h = k_cache.stride(1)
        stride_k_d = k_cache.stride(2)

        stride_v_p = v_cache.stride(0)
        stride_v_h = v_cache.stride(1)
        stride_v_d = v_cache.stride(2)

        # Output strides
        stride_out_b = output.stride(0)
        stride_out_h = output.stride(1)
        stride_out_d = output.stride(2)

        gqa_ratio = Nq // Nkv

        # Kernel 1: compute lse
        lse_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, float(sm_scale),
            lse,
            B, Nq, Nkv, D, gqa_ratio,
            stride_q_b, stride_q_h, stride_q_d,
            num_warps=4, num_stages=2,
        )

        # Kernel 2: compute output
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, float(sm_scale),
            lse, output,
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
