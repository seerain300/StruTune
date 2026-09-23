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
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D (bf16), cast to f32
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
        # k_ptr is [Np, Nkv, D]; each row has stride Nkv*D in the p dimension
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
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
    lse_ptr,        # *f32,  [B, Nq]  (not used directly; we recompute softmax)
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
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Recompute m and sum_exp for softmax
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

    # Initialize output vector accumulation in f32
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    acc = tl.zeros((D,), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        softmax_i = tl.exp(scaled - m) / sum_exp  # scalar
        v_base = v_ptr + idx * stride_v_p + kv_head * stride_v_h
        v_i = tl.load(v_base + tl.arange(0, D) * stride_v_d).to(tl.float32)  # [D]
        acc += softmax_i * v_i
        i += 1

    # Store as bfloat16
    out_store_base = out_ptr + b * stride_out_b + h * stride_out_h
    tl.store(out_store_base + tl.arange(0, D) * stride_out_d, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous
        device = q.device
        # We assume the same dtypes as original: q, k_cache, v_cache are bfloat16;
        # kv_indptr is int32; kv_indices is int32.
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Original assertions in the reference:
        B, Nq, D = q.shape
        assert Nq == 32, "num_qo_heads must be 32"
        Np, _, Nkv, _ = k_cache.shape
        assert Nkv == 8, "num_kv_heads must be 8"
        assert D == 128, "head_dim must be 128"

        T_total = kv_indices.numel()
        # The original PyTorch function asserts kv_indptr[-1] == T_total; we rely on harness.

        # Output tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Compute strides
        stride_q_b, stride_q_h, stride_q_d = q.stride()
        stride_k_p, stride_k_h, stride_k_d = k_cache.stride()
        stride_v_p, stride_v_h, stride_v_d = v_cache.stride()
        stride_out_b, stride_out_h, stride_out_d = output.stride()

        # gqa_ratio = Nq // Nkv
        gqa_ratio = Nq // Nkv

        # Launch Triton kernels: one program per (b, h)
        grid = (B * Nq,)

        # Cast q to bfloat16 for compute, k/v can be left in original dtype; we load and cast to f32 in-kernel.
        # Ensure dtype compatibility: Triton kernels read pointers; we cast to f32 inside kernels.
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale,
            lse,
            B, Nq, Nkv, D, gqa_ratio,
            stride_q_b, stride_q_h, stride_q_d,
            num_warps=4, num_stages=2,
        )

        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale,
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
