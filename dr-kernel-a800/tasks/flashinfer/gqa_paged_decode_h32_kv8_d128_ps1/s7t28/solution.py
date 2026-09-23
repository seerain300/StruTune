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
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Load q[b, h] vector of length D (bf16) and cast to f32 for compute
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
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

    # Load q[b, h] vector (bf16) and cast to f32 for compute
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
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # Compute output vector out[b, h, :] in f32, then store as bf16
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    j = 0
    while j < D:
        acc = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar
            scaled = attn * sm_scale
            p = tl.exp(scaled - m) / sum_exp  # scalar
            v_base = v_ptr + idx * stride_v_p + kv_head * stride_v_h
            v_j = tl.load(v_base + j * stride_v_d).to(tl.float32)  # scalar f32
            acc += p * v_j
            i += 1
        # Store acc as bf16
        tl.store(out_base + j * stride_out_d, acc.to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity for correct stride-based addressing; do NOT change dtype
        assert q.dim() == 3 and k_cache.dim() == 4 and v_cache.dim() == 4
        assert q.shape[2] == 128 and k_cache.shape[3] == 128 and v_cache.shape[3] == 128
        assert q.shape[1] == 32 and k_cache.shape[2] == 8 and v_cache.shape[2] == 8
        assert kv_indptr.shape[0] == q.shape[0] + 1 and kv_indices.numel() > 0
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B, Nq, D = q.shape
        Nkv = k_cache.shape[2]
        gqa_ratio = Nq // Nkv  # 4
        device = q.device

        # Output and lse tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch lse kernel: grid over (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale,
            lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=gqa_ratio,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: same grid
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale,
            lse, output,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=gqa_ratio,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_k_p=k_cache.stride(0), stride_k_h=k_cache.stride(2), stride_k_d=k_cache.stride(3),
            stride_v_p=v_cache.stride(0), stride_v_h=v_cache.stride(2), stride_v_d=v_cache.stride(3),
            stride_out_b=output.stride(0), stride_out_h=output.stride(1), stride_out_d=output.stride(2),
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
