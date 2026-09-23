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
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_p, stride_k_h, stride_k_d,
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

    # Compute m = max_i(q·k_i * sm_scale) and sum_exp = sum_i exp((q·k_i * sm_scale) - m)
    m = tl.full((), -float("inf"), tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
        k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D]
        attn_i = tl.sum(q_vec * k_i, axis=0)  # scalar
        attn_scaled = attn_i * sm_scale
        m = tl.maximum(m, attn_scaled)
        # sum_exp += exp(scaled - m)
        sum_exp += tl.exp(attn_scaled - m)
        i += 1

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
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_p, stride_k_h, stride_k_d,
    stride_v_p, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
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
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
        k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D]
        attn_i = tl.sum(q_vec * k_i, axis=0)  # scalar
        attn_scaled = attn_i * sm_scale
        m = tl.maximum(m, attn_scaled)
        sum_exp += tl.exp(attn_scaled - m)
        i += 1

    # Compute output vector out[b, h, :] = softmax @ v_selected
    out_base = out_ptr + b * stride_out_b + h * stride_out_h

    # We compute the entire output vector element by element; D is constexpr (128).
    j = 0
    while j < D:
        out_vec_j = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
            v_base = v_ptr + idx * stride_v_p + kv_head * stride_v_h
            k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D]
            attn_i = tl.sum(q_vec * k_i, axis=0)  # scalar
            attn_scaled = attn_i * sm_scale
            soft_i = tl.exp(attn_scaled - m) / sum_exp  # softmax[i]
            v_i = tl.load(v_base + tl.arange(0, D) * stride_v_d).to(tl.float32)  # [D]
            # out[b, h, j] += soft_i * v_i[j]
            out_vec_j += soft_i * v_i[j]
            i += 1
        tl.store(out_base + j * stride_out_d, out_vec_j.to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation of the original run function.
        q: [B, Nq, D] bfloat16
        k_cache: [Np, 1, Nkv, D] bfloat16
        v_cache: [Np, 1, Nkv, D] bfloat16
        kv_indptr: [B+1] int32
        kv_indices: [T_total] int32
        sm_scale: float32
        Returns:
        - output: [B, Nq, D] bfloat16
        - lse: [B, Nq] float32
        """
        # Ensure inputs are CUDA tensors
        assert q.is_cuda, "All inputs must be on CUDA for Triton kernels."
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        B, Nq, D = q.shape
        Np, _, Nkv, _ = k_cache.shape
        assert Nq == 32 and Nkv == 8 and D == 128, "ModelNew expects Nq=32, Nkv=8, D=128."

        # Output and lse initialization
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=q.device)

        # Strides
        stride_q_b, stride_q_h, stride_q_d = q.stride()
        stride_k_p, stride_k_h, stride_k_d = k_cache.stride()  # k_cache has shape [Np, Nkv, D]
        stride_v_p, stride_v_h, stride_v_d = v_cache.stride()
        stride_out_b, stride_out_h, stride_out_d = output.stride()

        # Launch Triton kernels: one program per (b, h)
        grid = (B * Nq,)

        # Kernel 1: compute lse
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale,
            lse,
            B, Nq, Nkv, D, Nq // Nkv,
            stride_q_b, stride_q_h, stride_q_d,
            stride_k_p, stride_k_h, stride_k_d,
            num_warps=4, num_stages=1,
        )

        # Kernel 2: compute output vectors
        output_kernel[grid](
            q, k_cache, v_cache, lse, output,
            kv_indptr, kv_indices, sm_scale,
            B, Nq, Nkv, D, Nq // Nkv,
            stride_q_b, stride_q_h, stride_q_d,
            stride_k_p, stride_k_h, stride_k_d,
            stride_v_p, stride_v_h, stride_v_d,
            stride_out_b, stride_out_h, stride_out_d,
            num_warps=4, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
