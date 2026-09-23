import math
import torch
import triton
import triton.language as tl


@triton.jit
def fused_lse_output_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *f32,  [Np, Nkv, D]
    v_ptr,          # *f32,  [Np, Nkv, D]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    out_ptr,        # *bf16, [B, Nq, D]
    lse_ptr,        # *f32,  [B, Nq]
    # shapes
    B: tl.constexpr,          # batch size
    Nq: tl.constexpr,         # num_qo_heads
    Nkv: tl.constexpr,        # num_kv_heads
    D: tl.constexpr,          # head_dim
    gqa_ratio: tl.constexpr,  # Nq // Nkv (4)
    # strides
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_k, stride_k_d,
    stride_v_b, stride_v_k, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D (bfloat16), convert to f32 for compute
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
        # Base pointers for k[idx, kv_head, :] and v[idx, kv_head, :]
        k_base = k_ptr + idx * stride_k_b + kv_head * stride_k_k
        v_base = v_ptr + idx * stride_v_b + kv_head * stride_v_k

        # Load k_i and v_i as vectors of length D (f32)
        k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D], f32
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

    # Now compute output vector: out[b, h, :] = sum_i softmax[i] * v_i
    inv_sum = 1.0 / sum_exp
    for j in range(D):
        out_vec_j = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            k_base = k_ptr + idx * stride_k_b + kv_head * stride_k_k
            v_base = v_ptr + idx * stride_v_b + kv_head * stride_v_k
            k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D], f32
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar f32
            scaled = attn * sm_scale
            prob = tl.exp(scaled - m) * inv_sum  # softmax probability
            v_j = tl.load(v_base + j * stride_v_d).to(tl.float32)  # scalar f32
            out_vec_j += prob * v_j
            i += 1
        # Store as bfloat16
        out_base = out_ptr + b * stride_out_b + h * stride_out_h
        tl.store(out_base + j * stride_out_d, out_vec_j.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous
        device = q.device
        # q is [B, Nq, D], bfloat16
        q = q.contiguous()
        # k_cache, v_cache are [Np, 1, Nkv, D]; squeeze middle dim and cast to float32
        k_cache = k_cache.contiguous().squeeze(1).to(torch.float32)  # [Np, Nkv, D], f32
        v_cache = v_cache.contiguous().squeeze(1).to(torch.float32)  # [Np, Nkv, D], f32
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        B, Nq, D = q.shape
        Np, Nkv, D2 = k_cache.shape
        assert D2 == D, "head_dim mismatch between q and k_cache"
        assert k_cache.shape == v_cache.shape, "k_cache and v_cache shapes must match"
        assert kv_indptr.shape[0] == B + 1
        # Ensure kv_indices length is consistent with indptr (not used directly for T here)

        # Allocate outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * Nq,)
        fused_lse_output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse,
            B, Nq, Nkv, D, Nq // Nkv,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
