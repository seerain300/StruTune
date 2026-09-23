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
    B, Nq, Nkv, D,  # runtime ints
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)        # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # GQA mapping: kv_head = h // (Nq // Nkv)
    gqa_ratio = Nq // Nkv
    kv_head = h // gqa_ratio

    # Load q[b, h] as f32 (vector of length D)
    q_vec = tl.load(q_ptr + b * Nq * D + h * D + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # Accumulators for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        # k_i vector of length D
        k_i = tl.load(k_ptr + idx * (Nkv * D) + kv_head * D + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # lse = log(sum_exp) + m, divided by log(2)
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
    lse_ptr,        # *f32,  [B, Nq]
    out_ptr,        # *bf16, [B, Nq, D]
    B, Nq, Nkv, D,  # runtime ints
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)        # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # GQA mapping: kv_head = h // (Nq // Nkv)
    gqa_ratio = Nq // Nkv
    kv_head = h // gqa_ratio

    # Load q[b, h] as f32
    q_vec = tl.load(q_ptr + b * Nq * D + h * D + tl.arange(0, D)).to(tl.float32)  # [D], f32

    # Recompute m and sum_exp for softmax
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_i = tl.load(k_ptr + idx * (Nkv * D) + kv_head * D + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # Produce output vector: out[b, h, :] = softmax @ v
    out_vec = tl.zeros((D,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_i = tl.load(k_ptr + idx * (Nkv * D) + kv_head * D + tl.arange(0, D)).to(tl.float32)  # [D], f32
        v_i = tl.load(v_ptr + idx * (Nkv * D) + kv_head * D + tl.arange(0, D)).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        soft_i = tl.exp(scaled - m) / sum_exp  # scalar
        # out_vec += soft_i * v_i (elementwise)
        for j in range(0, D):
            out_vec[j] += soft_i * v_i[j]
        i += 1

    # Store output as bfloat16
    out_base = out_ptr + b * Nq * D + h * D
    tl.store(out_base + tl.arange(0, D), out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        device = q.device
        assert device.type == "cuda", "Inputs must be on CUDA device"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes (runtime)
        B = q.shape[0]
        Nq = q.shape[1]
        D = q.shape[2]  # typically 128, but we keep it generic

        # Output tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch kernels: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, float(sm_scale), lse,
            B, Nq, 8, D,  # Nkv is 8 in the original; keep as runtime arg
            num_warps=4, num_stages=1,
        )
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, float(sm_scale), lse, output,
            B, Nq, 8, D,
            num_warps=4, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
