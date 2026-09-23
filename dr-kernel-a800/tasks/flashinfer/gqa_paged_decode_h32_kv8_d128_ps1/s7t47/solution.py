import math
import torch
import triton
import triton.language as tl


@triton.jit
def fused_lse_output_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D] but we will pass float32 tensors
    v_ptr,          # *bf16, [Np, Nkv, D] but we will pass float32 tensors
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32 scalar
    lse_ptr,        # *f32,  [B, Nq]
    out_ptr,        # *bf16, [B, Nq, D]
    B: tl.constexpr,          # batch size
    Nq: tl.constexpr,         # num_qo_heads
    Nkv: tl.constexpr,        # num_kv_heads
    D: tl.constexpr,          # head_dim
    gqa_ratio: tl.constexpr,  # Nq // Nkv (4)
    # strides for q
    stride_q_b, stride_q_h, stride_q_d,
    # strides for k/v (float32 tensors)
    stride_k_p, stride_k_h, stride_k_d,
    stride_v_p, stride_v_h, stride_v_d,
    # strides for out (bfloat16)
    stride_out_b, stride_out_h, stride_out_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D as float32
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D], f32
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

    # Compute output vector: out[b, h, :] = sum_i softmax[i] * v_i[j]
    inv_sum = 1.0 / sum_exp
    for j in range(D):
        out_vec_j = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D], f32
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar
            scaled = attn * sm_scale
            prob = tl.exp(scaled - m) * inv_sum  # softmax probability
            v_base = v_ptr + idx * (Nkv * D) + kv_head * D
            v_j = tl.load(v_base + j * stride_v_d).to(tl.float32)  # scalar
            out_vec_j += prob * v_j
            i += 1
        # Store as bfloat16
        out_base = out_ptr + b * stride_out_b + h * stride_out_h
        tl.store(out_base + j * stride_out_d, out_vec_j.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguity
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be on CUDA device."
        device = q.device

        # Cast k_cache and v_cache to float32 and squeeze middle dim to [Np, Nkv, D]
        k_cache = k_cache.contiguous().to(torch.float32).squeeze(1)  # [Np, Nkv, D], float32
        v_cache = v_cache.contiguous().to(torch.float32).squeeze(1)  # [Np, Nkv, D], float32

        # Ensure indices are int32 and contiguous
        kv_indices = kv_indices.contiguous().to(torch.int32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)

        B = q.shape[0]
        Nq = q.shape[1]
        D = q.shape[2]  # head_dim; must match k_cache/v_cache last dim
        assert k_cache.shape[-1] == D and v_cache.shape[-1] == D, "head_dim mismatch"
        Nkv = k_cache.shape[1]

        # Allocate outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch fused Triton kernel
        grid = (B * Nq,)
        fused_lse_output_kernel[grid](
            q, k_cache, v_cache,
            kv_indptr, kv_indices,
            sm_scale,
            lse, output,
            B, Nq, Nkv, D, Nq // Nkv,
            # q strides
            q.stride(0), q.stride(1), q.stride(2),
            # k/v strides (float32 tensors)
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            # output strides (bfloat16)
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
