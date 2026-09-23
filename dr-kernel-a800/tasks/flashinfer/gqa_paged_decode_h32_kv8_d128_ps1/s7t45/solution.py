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
    BLOCK_T: tl.constexpr,                   # chunk size for tokens
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D (bf16 -> f32 for compute)
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
        offs = i + tl.arange(0, BLOCK_T)
        mask = offs < T
        idx = tl.load(indices_ptr + start + offs, mask=mask, other=0).to(tl.int32)

        # Loop over this chunk to update m and sum_exp
        j = 0
        while j < BLOCK_T:
            cur = i + j
            valid = cur < T
            # if not valid, skip
            if valid:
                idx_j = tl.load(indices_ptr + start + cur).to(tl.int32)
                k_base = k_ptr + idx_j * (Nkv * D) + kv_head * D
                k_i = tl.load(k_base + tl.arange(0, D) * stride_q_d, mask=mask, other=0.0).to(tl.float32)  # [D]
                attn = tl.sum(q_vec * k_i, axis=0)  # scalar
                scaled = attn * sm_scale
                m = tl.maximum(m, scaled)
                exp_term = tl.exp(scaled - m)
                sum_exp += exp_term
            j += 1
        i += BLOCK_T

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
    gqa_ratio: tl.constexpr,  # Nq // Nkv (4)
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_p, stride_k_h, stride_k_d,
    stride_v_p, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
    BLOCK_T: tl.constexpr,                   # chunk size for tokens
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D (bf16 -> f32 for compute)
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Recompute m and sum_exp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        offs = i + tl.arange(0, BLOCK_T)
        mask = offs < T
        idx = tl.load(indices_ptr + start + offs, mask=mask, other=0).to(tl.int32)

        # Loop over chunk to update m and sum_exp
        j = 0
        while j < BLOCK_T:
            cur = i + j
            valid = cur < T
            if valid:
                idx_j = tl.load(indices_ptr + start + cur).to(tl.int32)
                k_base = k_ptr + idx_j * (Nkv * D) + kv_head * D
                k_i = tl.load(k_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D]
                attn = tl.sum(q_vec * k_i, axis=0)  # scalar
                scaled = attn * sm_scale
                m = tl.maximum(m, scaled)
                exp_term = tl.exp(scaled - m)
                sum_exp += exp_term
            j += 1
        i += BLOCK_T

    # Prepare output vector
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    out_vec = tl.zeros((D,), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        soft_i = tl.exp(scaled - m) / sum_exp  # scalar
        v_base = v_ptr + idx * stride_v_p + kv_head * stride_v_h
        v_i = tl.load(v_base + tl.arange(0, D) * stride_v_d).to(tl.float32)  # [D]
        out_vec += soft_i * v_i
        i += 1

    # Store output as bfloat16
    tl.store(out_base + tl.arange(0, D) * stride_out_d, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and contiguity
        device = q.device
        assert q.is_cuda, "q must be on CUDA"
        assert k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA"

        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B, Nq, D = q.shape
        Np, _, Nkv, _ = k_cache.shape
        assert k_cache.shape == (Np, 1, Nkv, D), "k_cache must have shape [num_pages, 1, num_kv_heads, head_dim]"
        assert v_cache.shape == (Np, 1, Nkv, D), "v_cache must have shape [num_pages, 1, num_kv_heads, head_dim]"

        # Output tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Compute strides (in elements)
        stride_q_b = q.stride(0)
        stride_q_h = q.stride(1)
        stride_q_d = q.stride(2)

        stride_k_p = k_cache.stride(0)
        stride_k_h = k_cache.stride(1)
        stride_k_d = k_cache.stride(2)

        stride_v_p = v_cache.stride(0)
        stride_v_h = v_cache.stride(1)
        stride_v_d = v_cache.stride(2)

        stride_out_b = output.stride(0)
        stride_out_h = output.stride(1)
        stride_out_d = output.stride(2)

        # Launch kernels: one program per (b, h)
        grid = (B * Nq,)
        # Use a chunk size for tokens; 128 is fine for typical D=128 workloads.
        BLOCK_T = 128

        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B, Nq, Nkv, D, (Nq // Nkv),
            stride_q_b, stride_q_h, stride_q_d,
            BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B, Nq, Nkv, D, (Nq // Nkv),
            stride_q_b, stride_q_h, stride_q_d,
            stride_k_p, stride_k_h, stride_k_d,
            stride_v_p, stride_v_h, stride_v_d,
            stride_out_b, stride_out_h, stride_out_d,
            BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
