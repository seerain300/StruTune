import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_lse_and_output_kernel(
    q_ptr,           # *bf16, [B, Nq, D]
    k_ptr,           # *f32,  [Np, Nkv, D]
    v_ptr,           # *f32,  [Np, Nkv, D]
    indptr_ptr,      # *i32,  [B+1]
    indices_ptr,     # *i32,  [T_total]
    sm_scale,        # f32
    lse_ptr,         # *f32,  [B, Nq]
    out_ptr,         # *bf16, [B, Nq, D]
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
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] as float32 vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Pass 1: compute m = max(scaled attn) and sum_exp = sum(exp(scaled - m))
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        # Base pointer for this token and kv_head
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h  # [D]
        attn = tl.sum(q_vec * tl.load(k_base + tl.arange(0, D) * stride_k_d), axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # lse = log(sum_exp) + m, then divide by log(2) to match original
    log2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) + m
    tl.store(lse_ptr + b * Nq + h, lse_val / log2)

    # Pass 2: compute output vector out[b, h, :] = sum_i softmax[i] * v_i[j]
    inv_sum = 1.0 / sum_exp
    for j in range(D):
        out_vec_j = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
            attn = tl.sum(q_vec * tl.load(k_base + tl.arange(0, D) * stride_k_d), axis=0)  # scalar
            scaled = attn * sm_scale
            prob = tl.exp(scaled - m) * inv_sum  # softmax probability
            v_base = v_ptr + idx * stride_v_p + kv_head * stride_v_h
            v_j = tl.load(v_base + j * stride_v_d).to(tl.float32)  # scalar
            out_vec_j += prob * v_j
            i += 1
        # Store as bfloat16
        out_base = out_ptr + b * stride_out_b + h * stride_out_h
        tl.store(out_base + j * stride_out_d, out_vec_j.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguity; make inputs ready
        device = q.device
        B, Nq, D = q.shape
        # k_cache, v_cache are [Np, 1, Nkv, D]; squeeze middle dim and cast to float32
        k_cache = k_cache.contiguous().to(torch.float32).squeeze(1)  # [Np, Nkv, D], float32
        v_cache = v_cache.contiguous().to(torch.float32).squeeze(1)  # [Np, Nkv, D], float32
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        # Allocate outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * Nq,)

        # Compute strides
        stride_q_b, stride_q_h, stride_q_d = q.stride()
        stride_k_p, stride_k_h, stride_k_d = k_cache.stride()
        stride_v_p, stride_v_h, stride_v_d = v_cache.stride()
        stride_out_b, stride_out_h, stride_out_d = output.stride()

        compute_lse_and_output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B=B, Nq=Nq, Nkv=k_cache.shape[1], D=D, gqa_ratio=Nq // k_cache.shape[1],
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            stride_k_p=stride_k_p, stride_k_h=stride_k_h, stride_k_d=stride_k_d,
            stride_v_p=stride_v_p, stride_v_h=stride_v_h, stride_v_d=stride_v_d,
            stride_out_b=stride_out_b, stride_out_h=stride_out_h, stride_out_d=stride_out_d,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
