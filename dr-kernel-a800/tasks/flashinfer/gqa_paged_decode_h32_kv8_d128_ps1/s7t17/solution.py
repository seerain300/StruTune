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
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] vector of length D
    # q is [B, Nq, D], contiguous => strides: (Nq*D, D, 1)
    q_base = q_ptr + b * (Nq * D) + h * D
    q_vec = tl.load(q_base + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b).to(tl.int32)     # int32
    end = tl.load(indptr_ptr + b + 1).to(tl.int32)   # int32
    T = end - start

    # Accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index
        # k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32
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
    D: tl.constexpr,
    gqa_ratio: tl.constexpr,  # Nq // Nkv (e.g., 4)
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] vector of length D
    q_base = q_ptr + b * (Nq * D) + h * D
    q_vec = tl.load(q_base + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b).to(tl.int32)     # int32
    end = tl.load(indptr_ptr + b + 1).to(tl.int32)   # int32
    T = end - start

    # Recompute m and sum_exp for softmax normalization
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar f32
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # Output vector accumulation: out[b, h, j] = sum_i softmax[i] * v_i[j]
    out_base = out_ptr + b * (Nq * D) + h * D
    for j in range(D):
        acc = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index
            v_base = v_ptr + idx * (Nkv * D) + kv_head * D
            v_j = tl.load(v_base + j).to(tl.float32)  # scalar f32
            # Recompute attn and scaled for softmax[i]
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D], f32
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar f32
            scaled_i = attn * sm_scale
            soft_i = tl.exp(scaled_i - m) / sum_exp
            acc += soft_i * v_j
            i += 1
        # Store as bfloat16
        tl.store(out_base + j, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous; use device of q
        device = q.device
        dtype = q.dtype  # typically bfloat16

        # Make sure inputs are on the same device and contiguous
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        # Shapes
        B, Nq, D = q.shape
        Np, Nkv, D_k = k_cache.shape  # D_k should match D
        assert D_k == D, "k_cache/v_cache last dim must match q's head_dim"
        assert k_cache.shape == v_cache.shape, "k_cache and v_cache must have same shape"
        assert k_cache.shape[1] == Nkv, "num_kv_heads must match k_cache's middle dimension"

        # Output allocations
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch lse kernel: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=Nq // Nkv,
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: one program per (b, h)
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=Nq // Nkv,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
