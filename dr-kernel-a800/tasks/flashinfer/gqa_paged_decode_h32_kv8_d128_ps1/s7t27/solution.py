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

    # Load q[b, h] vector (bf16), cast to f32 for compute
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    CHUNK_T = 128  # chunk size for tokens
    i = 0
    while i < T:
        offs = i + tl.arange(0, CHUNK_T)
        mask = offs < T
        idxs = tl.load(indices_ptr + start + offs, mask=mask, other=0).to(tl.int32)

        # For each token in the chunk: compute attn, update m and sum_exp
        j = 0
        while j < CHUNK_T:
            if mask[j]:
                idx = idxs[j]
                k_base = k_ptr + idx * (Nkv * D) + kv_head * D
                k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D], f32
                attn = tl.sum(q_vec * k_i, axis=0)  # scalar
                scaled = attn * sm_scale
                m = tl.maximum(m, scaled)
                sum_exp += tl.exp(scaled - m)
            j += 1
        i += CHUNK_T

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
    lse_ptr,        # *f32,  [B, Nq]  (unused for softmax here; we recompute m/sum_exp)
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

    # Recompute m and sum_exp (robust recompute)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    CHUNK_T = 128
    i = 0
    while i < T:
        offs = i + tl.arange(0, CHUNK_T)
        mask = offs < T
        idxs = tl.load(indices_ptr + start + offs, mask=mask, other=0).to(tl.int32)

        j = 0
        while j < CHUNK_T:
            if mask[j]:
                idx = idxs[j]
                k_base = k_ptr + idx * (Nkv * D) + kv_head * D
                k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)
                q_base = q_ptr + b * stride_q_b + h * stride_q_h
                q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)
                attn = tl.sum(q_vec * k_i, axis=0)  # scalar
                scaled = attn * sm_scale
                m = tl.maximum(m, scaled)
                sum_exp += tl.exp(scaled - m)
            j += 1
        i += CHUNK_T

    # Compute output vector out[b, h, :] in f32, then store as bf16
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    j_out = 0
    while j_out < D:
        acc = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            offs = i + tl.arange(0, CHUNK_T)
            mask = offs < T
            idxs = tl.load(indices_ptr + start + offs, mask=mask, other=0).to(tl.int32)

            jj = 0
            while jj < CHUNK_T:
                if mask[jj]:
                    idx = idxs[jj]
                    k_base = k_ptr + idx * (Nkv * D) + kv_head * D
                    k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)
                    q_base = q_ptr + b * stride_q_b + h * stride_q_h
                    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)
                    attn = tl.sum(q_vec * k_i, axis=0)  # scalar
                    scaled = attn * sm_scale
                    p = tl.exp(scaled - m) / sum_exp  # scalar softmax
                    v_base = v_ptr + idx * stride_v_p + kv_head * stride_v_h
                    v_j = tl.load(v_base + j_out * stride_v_d).to(tl.float32)
                    acc += p * v_j
                jj += 1
            i += CHUNK_T
        tl.store(out_base + j_out * stride_out_d, acc.to(tl.bfloat16))
        j_out += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Inputs are expected to be as in the original: q [B, Nq, D], k_cache [Np, Nkv, D], v_cache [Np, Nkv, D], kv_indptr [B+1], kv_indices [T_total], sm_scale float32
        # Do not modify inputs; keep original dtypes and shapes to match harness expectations.

        device = q.device
        B, Nq, D = q.shape
        Np, Nkv, _ = k_cache.shape  # Nkv expected 8, D expected 128 in harness

        # Outputs
        out = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Compute strides (PyTorch stride is in elements)
        stride_q_b, stride_q_h, stride_q_d = q.stride()
        stride_k_p, stride_k_h, stride_k_d = k_cache.stride()
        stride_v_p, stride_v_h, stride_v_d = v_cache.stride()
        stride_out_b, stride_out_h, stride_out_d = out.stride()

        # Launch lse kernel: one program per (b, h)
        grid_lse = (B * Nq,)
        lse_kernel[grid_lse](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=Nq // Nkv,
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: one program per (b, h)
        grid_out = (B * Nq,)
        output_kernel[grid_out](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, out,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=Nq // Nkv,
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            stride_k_p=stride_k_p, stride_k_h=stride_k_h, stride_k_d=stride_k_d,
            stride_v_p=stride_v_p, stride_v_h=stride_v_h, stride_v_d=stride_v_d,
            stride_out_b=stride_out_b, stride_out_h=stride_out_h, stride_out_d=stride_out_d,
            num_warps=4, num_stages=2,
        )

        return out, lse


def run(*args):
    return ModelNew()(*args)
