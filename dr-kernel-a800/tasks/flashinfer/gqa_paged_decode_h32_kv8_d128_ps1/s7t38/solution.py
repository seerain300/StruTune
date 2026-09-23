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
    gqa_ratio: tl.constexpr,  # Nq // Nkv (should be 4)
    stride_q_b, stride_q_h, stride_q_d,  # strides for q
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio

    # Load q[b, h] vector of length D (contiguous last dim)
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
        # Base for k[idx, kv_head, :] as a 1D contiguous vector over D
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        # update m and sum_exp
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
    kv_head = h // gqa_ratio

    # Load q[b, h] vector
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # Recompute m and sum_exp for softmax
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D)).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # Compute output vector out[b, h, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        v_base = v_ptr + idx * (Nkv * D) + kv_head * D
        v_i = tl.load(v_base + tl.arange(0, D)).to(tl.float32)  # [D]
        softmax_i = tl.exp((tl.load(indices_ptr + start + i).to(tl.float32) * 0 + scaled) - m) / sum_exp  # placeholder: compute as exp(scaled - m) / sum_exp
        # softmax_i = exp(scaled - m) / sum_exp
        # We don't recompute scaled; use (scaled - m) computed in last loop:
        # We need the scaled value here. Recompute attn for i, then scaled, then softmax_i.
        # However, we cannot read sm_scale here. Instead, since scaled is per i, we recompute:
        # But we don't have sm_scale? We do have sm_scale in kernel signature; use it.
        # We can compute scaled by reusing the variable. The previous loop doesn't save scaled; we must recompute.
        # But to avoid complexity, we recompute attn here:
        k_base_i = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i_i = tl.load(k_base_i + tl.arange(0, D)).to(tl.float32)
        attn_i = tl.sum(q_vec * k_i_i, axis=0)
        scaled_i = attn_i * sm_scale
        softmax_i = tl.exp(scaled_i - m) / sum_exp
        # Accumulate out_vec = sum over i of softmax_i * v_i
        out_vec += softmax_i * v_i
        i += 1

    # Store output as bfloat16
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    tl.store(out_base + tl.arange(0, D) * stride_out_d, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous; avoid any shape-unpacking on non-3D tensors
        device = q.device
        q = q.contiguous().to(device=device, dtype=torch.bfloat16)
        k_cache = k_cache.contiguous().to(device=device, dtype=torch.bfloat16)
        v_cache = v_cache.contiguous().to(device=device, dtype=torch.bfloat16)
        kv_indptr = kv_indptr.contiguous().to(device=device, dtype=torch.int32)
        kv_indices = kv_indices.contiguous().to(device=device, dtype=torch.int32)

        B, Nq, D = q.shape
        Np, Nkv, D_k, D_v = k_cache.shape
        assert D_k == D_v == D, "k/v head dim mismatch"
        assert Nkv == 8, "num_kv_heads must be 8 per original"
        T_total = kv_indices.numel()

        # Output tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch lse kernel: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale,
            lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=Nq // Nkv,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: one program per (b, h)
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=Nq // Nkv,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_k_p=k_cache.stride(0), stride_k_h=k_cache.stride(1), stride_k_d=k_cache.stride(2),
            stride_v_p=v_cache.stride(0), stride_v_h=v_cache.stride(1), stride_v_d=v_cache.stride(2),
            stride_out_b=output.stride(0), stride_out_h=output.stride(1), stride_out_d=output.stride(2),
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
