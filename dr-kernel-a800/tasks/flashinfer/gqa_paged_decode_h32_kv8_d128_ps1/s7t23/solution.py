import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D] (this is squeeze(1) result)
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
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # Accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * D  # squeezed: [Np, Nkv, D]
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        # Load q[b, h] vector and compute dot
        q_base = q_ptr + b * stride_q_b + h * stride_q_h
        q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D]
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
    k_ptr,          # *bf16, [Np, Nkv, D] (squeezed)
    v_ptr,          # *bf16, [Np, Nkv, D] (squeezed)
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
    stride_k_p, stride_k_h, stride_k_d,  # for squeezed k_ptr, stride_k_p=D, stride_k_h=D, stride_k_d=1
    stride_v_p, stride_v_h, stride_v_d,  # for squeezed v_ptr, stride_v_p=D, stride_v_h=D, stride_v_d=1
    stride_out_b, stride_out_h, stride_out_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D (float32 for compute)
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # Recompute m and sum_exp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * D  # squeezed: [Np, Nkv, D]
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # Output vector: out[b, h, :] = sum_i (exp(scaled[i]-m)/sum_exp) * v_selected[i]
    out_row_base = out_ptr + b * stride_out_b + h * stride_out_h
    j = 0
    while j < D:
        acc = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            # v_ptr is squeezed [Np, Nkv, D]
            v_base = v_ptr + idx * D
            v_i = tl.load(v_base + j * stride_v_d).to(tl.float32)  # scalar
            # prob = exp(scaled[i] - m) / sum_exp
            k_base = k_ptr + idx * D
            k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar
            scaled = attn * sm_scale
            prob = tl.exp(scaled - m) / sum_exp
            acc += prob * v_i
            i += 1
        tl.store(out_row_base + j * stride_out_d, acc.to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous; squeeze k_cache/v_cache along dim=1 to match original behavior
        device = q.device
        assert q.is_cuda, "Input q must be on CUDA"
        # Make contiguous
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Squeeze middle dim to [Np, Nkv, D] exactly like original
        k_cache = k_cache.squeeze(1)
        v_cache = v_cache.squeeze(1)

        # Shapes: derive Nq, Nkv, D, B
        B = q.shape[0]
        Nq = q.shape[1]
        D = q.shape[2]
        Nkv = k_cache.shape[1]
        # gqa_ratio = Nq // Nkv, expected 4 in given tests
        gqa_ratio = Nq // Nkv

        # Allocate outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Strides for q
        stride_q_b, stride_q_h, stride_q_d = q.stride()
        # Strides for squeezed k_cache/v_cache: [Np, Nkv, D]
        stride_k_p, stride_k_h, stride_k_d = k_cache.stride()  # expected (D, D, 1)
        stride_v_p, stride_v_h, stride_v_d = v_cache.stride()  # expected (D, D, 1)
        # Strides for output
        stride_out_b, stride_out_h, stride_out_d = output.stride()

        # Launch lse kernel: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=gqa_ratio,
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: one program per (b, h)
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=gqa_ratio,
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            stride_k_p=stride_k_p, stride_k_h=stride_k_h, stride_k_d=stride_k_d,
            stride_v_p=stride_v_p, stride_v_h=stride_v_h, stride_v_d=stride_v_d,
            stride_out_b=stride_out_b, stride_out_h=stride_out_h, stride_out_d=stride_out_d,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
