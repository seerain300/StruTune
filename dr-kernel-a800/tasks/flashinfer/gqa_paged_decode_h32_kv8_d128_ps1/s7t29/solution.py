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
    B: tl.constexpr,          # batch size
    Nq: tl.constexpr,         # num_qo_heads
    Nkv: tl.constexpr,        # num_kv_heads
    D: tl.constexpr,          # head_dim
    gqa_ratio: tl.constexpr,  # Nq // Nkv (4)
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Compute token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Load q[b, h] as bf16 vector, keep as is (bf16) for computation
    q_base = q_ptr + b * (Nq * D) + h * D
    q_vec_bf16 = tl.load(q_base + tl.arange(0, D))  # [D], bf16

    # Compute m = max(scaled) and sum_exp in bf16
    m = tl.full((), -float("inf"), dtype=tl.bfloat16)
    sum_exp = tl.zeros((), dtype=tl.bfloat16)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i_bf16 = tl.load(k_base + tl.arange(0, D))  # [D], bf16
        attn_bf16 = tl.sum(q_vec_bf16 * k_i_bf16, axis=0)  # scalar bf16
        scaled_bf16 = attn_bf16 * sm_scale  # scalar bf16
        # max and sum in bf16
        m = tl.maximum(m, scaled_bf16)
        exp_term = tl.exp(scaled_bf16 - m)  # scalar bf16
        sum_exp += exp_term
        i += 1

    # lse = log(sum_exp) + m, then divide by log(2.0) to match original
    log2 = 0.6931471805599453  # f32
    lse_val = tl.log(sum_exp.to(tl.float32)) + m.to(tl.float32)  # f32
    lse_val = lse_val / log2
    tl.store(lse_ptr + b * Nq + h, lse_val)  # store as f32


@triton.jit
def output_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D]
    v_ptr,          # *bf16, [Np, Nkv, D]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32 scalar
    lse_ptr,        # *f32,  [B, Nq] (we won't use it here, but kept for signature symmetry)
    out_ptr,        # *bf16, [B, Nq, D]
    B: tl.constexpr,          # batch size
    Nq: tl.constexpr,         # num_qo_heads
    Nkv: tl.constexpr,        # num_kv_heads
    D: tl.constexpr,          # head_dim
    gqa_ratio: tl.constexpr,  # Nq // Nkv (4)
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] as bf16 vector
    q_base = q_ptr + b * (Nq * D) + h * D
    q_vec_bf16 = tl.load(q_base + tl.arange(0, D))  # [D], bf16

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Recompute m and sum_exp in bf16 for softmax
    m = tl.full((), -float("inf"), dtype=tl.bfloat16)
    sum_exp = tl.zeros((), dtype=tl.bfloat16)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i_bf16 = tl.load(k_base + tl.arange(0, D))  # [D], bf16
        attn_bf16 = tl.sum(q_vec_bf16 * k_i_bf16, axis=0)  # scalar bf16
        scaled_bf16 = attn_bf16 * sm_scale  # scalar bf16
        m = tl.maximum(m, scaled_bf16)
        exp_term = tl.exp(scaled_bf16 - m)  # scalar bf16
        sum_exp += exp_term
        i += 1

    # Produce output vector out[b, h, :] in bf16
    out_base = out_ptr + b * (Nq * D) + h * D
    j = 0
    while j < D:
        acc_bf16 = tl.zeros((), dtype=tl.bfloat16)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i_bf16 = tl.load(k_base + tl.arange(0, D))  # [D], bf16
            attn_bf16 = tl.sum(q_vec_bf16 * k_i_bf16, axis=0)  # scalar bf16
            scaled_bf16 = attn_bf16 * sm_scale  # scalar bf16
            p_bf16 = tl.exp(scaled_bf16 - m) / sum_exp  # scalar bf16
            v_base = v_ptr + idx * (Nkv * D) + kv_head * D
            v_j_bf16 = tl.load(v_base + j)  # scalar bf16
            acc_bf16 += p_bf16 * v_j_bf16
            i += 1
        # store as bf16
        tl.store(out_base + j, acc_bf16)
        j += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are contiguous for correct pointer arithmetic
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes: B, Nq, Nkv, D are fixed in original assumptions
        B, Nq, D = q.shape
        Np, _, Nkv, _ = k_cache.shape
        assert Nq == 32, "num_qo_heads must be 32"
        assert Nkv == 8, "num_kv_heads must be 8"
        assert D == 128, "head_dim must be 128"

        # Output buffers
        out = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=q.device)

        # Launch Triton kernels: one program per (b, h)
        grid = (B * Nq,)

        # Compute lse
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, float(sm_scale), lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=(Nq // Nkv),
            num_warps=4, num_stages=2,
        )

        # Compute output
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, float(sm_scale), lse, out,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=(Nq // Nkv),
            num_warps=4, num_stages=2,
        )

        return out, lse


def run(*args):
    return ModelNew()(*args)
