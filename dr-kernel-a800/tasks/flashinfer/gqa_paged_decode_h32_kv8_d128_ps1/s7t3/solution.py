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
    stride_q_b, stride_q_h, stride_q_d,  # strides for q (B, Nq, D)
    stride_k_p, stride_k_h, stride_k_d,  # strides for k (Np, Nkv, D)
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] as f32 vector of length D
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Determine token window [start, end)
    start = tl.load(indptr_ptr + b)       # int32
    end = tl.load(indptr_ptr + b + 1)     # int32
    T = end - start

    # Compute m = max(scaled attn) and sum_exp = sum(exp(scaled - m)) in chunks
    m = tl.full((), -1.0e30, tl.float32)
    sum_exp = tl.zeros((), tl.float32)

    i = 0
    BLOCK_T = 128
    while i < T:
        j = 0
        while j < BLOCK_T:
            idx = i + j
            if idx >= T:
                break
            # Load idx-th token index within this batch's window
            idx_i = tl.load(indices_ptr + start + idx).to(tl.int32)
            # k_i vector [D] for kv_head
            k_base = k_ptr + idx_i * stride_k_p + kv_head * stride_k_h
            k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D], f32
            # attn = q·k_i
            attn = tl.sum(q_vec * k_i, axis=0)  # scalar
            scaled = attn * sm_scale
            # Update m and sum_exp
            m_new = tl.maximum(m, scaled)
            sum_exp = sum_exp * tl.exp(m - m_new) + tl.exp(scaled - m_new)
            m = m_new
            j += 1
        i += BLOCK_T

    # lse = log(sum_exp) + m; original divides by log(2)
    log2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) + m
    lse_val = lse_val / log2

    # Store lse[b, h]
    tl.store(lse_ptr + b * Nq + h, lse_val)


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
    stride_q_b, stride_q_h, stride_q_d,     # strides for q (B, Nq, D)
    stride_out_b, stride_out_h, stride_out_d,  # strides for out (B, Nq, D)
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] as f32 vector of length D
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Determine token window [start, end)
    start = tl.load(indptr_ptr + b)       # int32
    end = tl.load(indptr_ptr + b + 1)     # int32
    T = end - start

    # Recompute m and sum_exp
    m = tl.full((), -1.0e30, tl.float32)
    sum_exp = tl.zeros((), tl.float32)
    i = 0
    BLOCK_T = 128
    while i < T:
        j = 0
        while j < BLOCK_T:
            idx = i + j
            if idx >= T:
                break
            idx_i = tl.load(indices_ptr + start + idx).to(tl.int32)
            k_base = k_ptr + idx_i * stride_k_p + kv_head * stride_k_h
            k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D]
            attn = tl.sum(q_vec * k_i, axis=0)
            scaled = attn * sm_scale
            m_new = tl.maximum(m, scaled)
            sum_exp = sum_exp * tl.exp(m - m_new) + tl.exp(scaled - m_new)
            m = m_new
            j += 1
        i += BLOCK_T

    # Compute output vector per j
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    j = 0
    while j < D:
        acc = tl.zeros((), tl.float32)
        i = 0
        while i < T:
            idx = i
            idx_i = tl.load(indices_ptr + start + idx).to(tl.int32)
            k_base = k_ptr + idx_i * stride_k_p + kv_head * stride_k_h
            k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D]
            attn = tl.sum(q_vec * k_i, axis=0)
            scaled = attn * sm_scale
            soft_i = tl.exp(scaled - m) / sum_exp
            v_base = v_ptr + idx_i * stride_k_p + kv_head * stride_k_h  # same layout as k/v
            v_i_j = tl.load(v_base + j * stride_k_d).to(tl.float32)     # scalar
            acc += soft_i * v_i_j
            i += 1
        tl.store(out_base + j * stride_out_d, acc.to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguity; keep dtype for output
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        B, Nq, D = q.shape
        Np, Nkv, Dk = k_cache.shape
        assert Dk == D and Nkv == 8 and Nq == 32 and kv_indptr.shape[0] == B + 1

        # Compute in float32, output in bfloat16
        q_f32 = q.to(torch.float32)
        k_f32 = k_cache.to(torch.float32)
        v_f32 = v_cache.to(torch.float32)

        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=q.device)

        grid = (B * Nq,)

        # Strides
        stride_q_b, stride_q_h, stride_q_d = q_f32.stride()
        stride_k_p, stride_k_h, stride_k_d = k_f32.stride()
        # out uses same layout as q
        stride_out_b, stride_out_h, stride_out_d = output.stride()

        # Kernel 1: lse
        lse_kernel[grid](
            q_f32, k_f32, kv_indptr, kv_indices, sm_scale, lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=(Nq // Nkv),
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            stride_k_p=stride_k_p, stride_k_h=stride_k_h, stride_k_d=stride_k_d,
            num_warps=4, num_stages=2,
        )

        # Kernel 2: output
        output_kernel[grid](
            q_f32, k_f32, v_f32, kv_indptr, kv_indices, sm_scale, lse, output,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=(Nq // Nkv),
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            stride_out_b=stride_out_b, stride_out_h=stride_out_h, stride_out_d=stride_out_d,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
