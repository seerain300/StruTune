import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D]
    v_ptr,          # *bf16, [Np, Nkv, D]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    lse_ptr,        # *f32,  [B, Nq]
    B,              # int32
    Nq,             # int32
    Nkv,            # int32
    D,              # int32 (head_dim, runtime)
    stride_q_b, stride_q_h, stride_q_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    gqa_ratio = Nq // Nkv
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector
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
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index in [0, Np)

        # Build pointers to k[idx, kv_head, :] and v[idx, kv_head, :]
        # Address for k/v row: base = idx * (Nkv * D) + kv_head * D
        k_row_base = k_ptr + idx * (Nkv * D) + kv_head * D
        v_row_base = v_ptr + idx * (Nkv * D) + kv_head * D

        # Load k_i and v_i vectors
        k_i = tl.load(k_row_base + tl.arange(0, D) * 1).to(tl.float32)  # [D], f32
        v_i = tl.load(v_row_base + tl.arange(0, D) * 1).to(tl.float32)  # [D], f32

        # Compute dot and scaled
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale

        # Update max and sum_exp
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
    B,              # int32
    Nq,             # int32
    Nkv,            # int32
    D,              # int32 (head_dim, runtime)
    stride_q_b, stride_q_h, stride_q_d,
    stride_out_b, stride_out_h, stride_out_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    gqa_ratio = Nq // Nkv
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D
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
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index in [0, Np]

        k_row_base = k_ptr + idx * (Nkv * D) + kv_head * D
        v_row_base = v_ptr + idx * (Nkv * D) + kv_head * D

        k_i = tl.load(k_row_base + tl.arange(0, D) * 1).to(tl.float32)
        attn = tl.sum(q_vec * k_i, axis=0)
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # Compute output vector out[b, h, :] in bfloat16
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    j = 0
    while j < D:
        # Reinitialize output element accumulator
        out_val = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            k_row_base = k_ptr + idx * (Nkv * D) + kv_head * D
            v_row_base = v_ptr + idx * (Nkv * D) + kv_head * D

            k_i = tl.load(k_row_base + tl.arange(0, D) * 1).to(tl.float32)
            attn = tl.sum(q_vec * k_i, axis=0)
            scaled = attn * sm_scale
            soft = tl.exp(scaled - m) / sum_exp
            v_i = tl.load(v_row_base + tl.arange(0, D) * 1).to(tl.float32)
            out_val += soft * v_i[j]
            i += 1
        # store as bfloat16
        tl.store(out_base + j * stride_out_d, out_val.to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous; convert dtypes
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Inputs must be on CUDA"
        q = q.contiguous().to(torch.bfloat16)
        k_cache = k_cache.contiguous().to(torch.bfloat16)
        v_cache = v_cache.contiguous().to(torch.bfloat16)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        B = q.shape[0]
        Nq = q.shape[1]
        Np = k_cache.shape[0]  # number of cache pages (should match kv_indices length in typical usage)
        Nkv = k_cache.shape[1]  # number of kv heads (8 in original, but we handle generically)
        D = q.shape[2]  # head_dim, dynamic

        # Allocate outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Strides for q and output (for address computation in Triton)
        stride_q_b, stride_q_h, stride_q_d = q.stride()
        stride_out_b, stride_out_h, stride_out_d = output.stride()

        # Launch kernels: one program per (b, h)
        grid = (B * Nq,)

        # LSE kernel
        lse_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale,
            lse,
            B, Nq, Nkv, D,
            stride_q_b, stride_q_h, stride_q_d,
            num_warps=4, num_stages=1
        )

        # Output kernel
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B, Nq, Nkv, D,
            stride_q_b, stride_q_h, stride_q_d,
            stride_out_b, stride_out_h, stride_out_d,
            num_warps=4, num_stages=1
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
