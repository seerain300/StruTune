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
    gqa_ratio: tl.constexpr,  # Nq // Nkv (4)
    stride_q_b, stride_q_h, stride_q_d,  # strides for q
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] as float32 vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Determine token window for this batch
    start = tl.load(indptr_ptr + b)       # int32
    end = tl.load(indptr_ptr + b + 1)    # int32
    T = end - start  # dynamic

    # First pass: compute m = max((q·k_i)*sm_scale) and sum_exp = sum_i exp((q·k_i*sm_scale) - m) in chunks
    m = tl.full((), -float("inf"), tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    BLOCK_T = 128
    while i < T:
        chunk = T - i
        if chunk > BLOCK_T:
            chunk = BLOCK_T
        idxs = start + i + tl.arange(0, chunk)  # [chunk] linear token offsets
        # load indices for these offsets
        idxs = tl.load(indices_ptr + idxs).to(tl.int32)  # [chunk]
        # for each t in chunk: update m and sum_exp
        t = 0
        while t < chunk:
            idx = idxs[t]  # scalar int32
            # k_ptr points to [Np, Nkv, D]; we select kv_head and load k_i of length D
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D  # idx along Np, kv_head along Nkv, D is last dim
            k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
            attn_i = tl.sum(q_vec * k_i, axis=0)  # scalar
            attn_scaled = attn_i * sm_scale
            m = tl.maximum(m, attn_scaled)
            sum_exp += tl.exp(attn_scaled - m)
            t += 1
        i += BLOCK_T

    # lse = log(sum_exp) + m, then divide by log(2)
    lse_val = tl.log(sum_exp) + m
    log2 = 0.6931471805599453  # 1 / log(2)
    lse_val = lse_val / log2

    # Store lse[b, h]
    lse_base = lse_ptr + b * Nq + h
    tl.store(lse_base, lse_val)


@triton.jit
def output_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D]
    v_ptr,          # *bf16, [Np, Nkv, D]
    lse_ptr,        # *f32,  [B, Nq]
    out_ptr,        # *bf16, [B, Nq, D]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    B: tl.constexpr,          # batch size
    Nq: tl.constexpr,         # num_qo_heads
    Nkv: tl.constexpr,        # num_kv_heads
    D: tl.constexpr,          # head_dim
    gqa_ratio: tl.constexpr,  # Nq // Nkv (4)
    stride_q_b, stride_q_h, stride_q_d,  # strides for q
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] as float32 vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Determine token window for this batch
    start = tl.load(indptr_ptr + b)       # int32
    end = tl.load(indptr_ptr + b + 1)    # int32
    T = end - start  # dynamic

    # Recompute m and sum_exp for softmax
    m = tl.full((), -float("inf"), tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    BLOCK_T = 128
    while i < T:
        chunk = T - i
        if chunk > BLOCK_T:
            chunk = BLOCK_T
        idxs = start + i + tl.arange(0, chunk)  # [chunk] linear token offsets
        idxs = tl.load(indices_ptr + idxs).to(tl.int32)  # [chunk]
        t = 0
        while t < chunk:
            idx = idxs[t]  # scalar int32
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
            attn_i = tl.sum(q_vec * k_i, axis=0)  # scalar
            attn_scaled = attn_i * sm_scale
            m = tl.maximum(m, attn_scaled)
            sum_exp += tl.exp(attn_scaled - m)
            t += 1
        i += BLOCK_T

    # Compute output vector out[b, h, :] = softmax @ v_selected
    out_base = out_ptr + b * (Nq * D) + h * D  # treat out as [B, Nq, D] contiguous: stride along Nq is D, along D is 1
    j = 0
    while j < D:
        out_vec_j = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            chunk = T - i
            if chunk > BLOCK_T:
                chunk = BLOCK_T
            idxs = start + i + tl.arange(0, chunk)  # [chunk]
            idxs = tl.load(indices_ptr + idxs).to(tl.int32)  # [chunk]
            t = 0
            while t < chunk:
                idx = idxs[t]  # scalar int32
                k_base = k_ptr + idx * (Nkv * D) + kv_head * D
                k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
                attn_i = tl.sum(q_vec * k_i, axis=0)  # scalar
                attn_scaled = attn_i * sm_scale
                soft_i = tl.exp(attn_scaled - m) / sum_exp  # softmax[i]
                v_base = v_ptr + idx * (Nkv * D) + kv_head * D
                v_i = tl.load(v_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
                out_vec_j += soft_i * v_i[j]
                t += 1
            i += BLOCK_T
        tl.store(out_base + j, out_vec_j.to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation of the original run function.
        q: [B, Nq, D] bfloat16
        k_cache: [Np, 1, Nkv, D] bfloat16 (we'll use [Np, Nkv, D])
        v_cache: [Np, 1, Nkv, D] bfloat16
        kv_indptr: [B+1] int32
        kv_indices: [T_total] int32
        sm_scale: float32
        Returns:
        - output: [B, Nq, D] bfloat16
        - lse: [B, Nq] float32
        """
        # Ensure inputs are CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All inputs must be on CUDA for Triton kernels."

        # Make inputs contiguous to avoid any stride-related issues
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        # Extract shapes (B, Nq, D) from q; k_cache/v_cache shapes are [Np, Nkv, D]
        B, Nq, D = q.shape
        assert Nq == 32 and D == 128, "ModelNew expects Nq=32 and D=128."
        Np, Nkv, _ = k_cache.shape
        assert Nkv == 8, "ModelNew expects Nkv=8."

        # Output and lse initialization
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=q.device)

        # Strides for q (we access q as [B, Nq, D] contiguous)
        stride_q_b, stride_q_h, stride_q_d = q.stride()

        # Launch Triton kernels: one program per (b, h)
        grid = (B * Nq,)

        # Kernel 1: compute lse
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale,
            lse,
            B, Nq, Nkv, D, Nq // Nkv,
            stride_q_b, stride_q_h, stride_q_d,
            num_warps=4, num_stages=1,
        )

        # Kernel 2: compute output vectors
        output_kernel[grid](
            q, k_cache, v_cache, lse, output,
            kv_indptr, kv_indices, sm_scale,
            B, Nq, Nkv, D, Nq // Nkv,
            stride_q_b, stride_q_h, stride_q_d,
            num_warps=4, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
