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
    gqa_ratio: tl.constexpr,  # Nq // Nkv
    stride_q_b, stride_q_h, stride_q_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D (float32 compute)
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Accumulate max and sum_exp for logsumexp over tokens
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        # k_base points to [Nkv, D] for this token; we select kv_head-th head
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
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D (float32 compute)
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Recompute m and sum_exp (same as lse kernel)
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

    # Output vector: out[b, h, :] = softmax @ v_selected, accumulate elementwise
    out_vec = tl.zeros((D,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        v_base = v_ptr + idx * (Nkv * D) + kv_head * D
        v_i = tl.load(v_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        # Recompute attn for token i (cheap)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        p_i = tl.exp(scaled - m) / sum_exp  # scalar probability for token i
        # Accumulate out_vec += p_i * v_i
        out_vec += p_i * v_i
        i += 1

    # Store as bfloat16
    tl.store(out_ptr + b * stride_out_b + h * stride_out_h + tl.arange(0, D) * stride_out_d, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous
        device = q.device
        q = q.to(device).contiguous()
        k_cache = k_cache.to(device).contiguous()
        v_cache = v_cache.to(device).contiguous()
        kv_indptr = kv_indptr.to(device).contiguous()
        kv_indices = kv_indices.to(device).contiguous()

        # Shapes as compile-time constants for Triton (B, Nq, Nkv, D from q/k_cache)
        B = q.shape[0]
        Nq = q.shape[1]
        Np = k_cache.shape[0]
        Nkv = k_cache.shape[2]
        D = q.shape[2]

        # Output and lse buffers
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Strides for q
        stride_q_b, stride_q_h, stride_q_d = q.stride()

        # Strides for k/v
        stride_k_p, stride_k_h, stride_k_d = k_cache.stride()
        stride_v_p, stride_v_h, stride_v_d = v_cache.stride()

        # Strides for output
        stride_out_b, stride_out_h, stride_out_d = output.stride()

        # Launch Triton kernels: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B, Nq, Nkv, D, Nq // Nkv,
            stride_q_b, stride_q_h, stride_q_d,
            num_warps=4, num_stages=2,
        )

        # Second kernel to compute output vectors
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B, Nq, Nkv, D, Nq // Nkv,
            stride_q_b, stride_q_h, stride_q_d,
            stride_k_p, stride_k_h, stride_k_d,
            stride_v_p, stride_v_h, stride_v_d,
            stride_out_b, stride_out_h, stride_out_d,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
