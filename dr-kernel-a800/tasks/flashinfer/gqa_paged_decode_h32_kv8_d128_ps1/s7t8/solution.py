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
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Vector q[b, h, :]
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

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
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # lse = log(sum_exp) + m; divide by log(2)
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
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # q[b, h, :]
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # Recompute m and sum_exp for softmax
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

    # Output vector out[b, h, :] = softmax @ v_selected
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    for j in range(D):
        dot_j = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
            v_base = v_ptr + idx * (Nkv * D) + kv_head * D
            v_i = tl.load(v_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar
            scaled = attn * sm_scale
            soft = tl.exp(scaled - m) / sum_exp
            dot_j += soft * v_i[j]
            i += 1
        tl.store(out_base + j * stride_out_d, dot_j.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and contiguity; do NOT assume fixed shapes or assert them
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes
        B = q.shape[0]
        Nq = q.shape[1]
        D = q.shape[2]

        Np = k_cache.shape[0]
        Nkv = k_cache.shape[1]
        # k_cache/v_cache must be [Np, Nkv, D] like original; handle generically
        assert k_cache.shape == v_cache.shape, "k_cache and v_cache must have the same shape"
        assert k_cache.shape[1] == Nkv and k_cache.shape[2] == D, "k_cache/v_cache must be [Np, Nkv, D]"

        # Output and LSE buffers
        lse = torch.empty((B, Nq), dtype=torch.float32, device=q.device)
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=q.device)

        # Strides for q
        stride_q_b, stride_q_h, stride_q_d = q.stride()

        # Launch LSE kernel: one program per (b, h)
        grid_lse = (B * Nq,)
        lse_kernel[grid_lse](
            q, k_cache, kv_indptr, kv_indices, sm_scale,
            lse,
            B, Nq, Nkv, D, Nq // Nkv,
            stride_q_b, stride_q_h, stride_q_d,
            num_warps=4, num_stages=2,
        )

        # Strides for v/output (output strides mirror q's because it's [B, Nq, D])
        stride_out_b, stride_out_h, stride_out_d = output.stride()

        # Launch output kernel: one program per (b, h)
        grid_out = (B * Nq,)
        output_kernel[grid_out](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale,
            lse, output,
            B, Nq, Nkv, D, Nq // Nkv,
            stride_q_b, stride_q_h, stride_q_d,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            stride_out_b, stride_out_h, stride_out_d,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
