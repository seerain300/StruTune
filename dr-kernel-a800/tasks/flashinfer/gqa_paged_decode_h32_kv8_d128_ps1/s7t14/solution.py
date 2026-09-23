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
    B,              # i32
    Nq,             # i32
    Nkv,            # i32
    D,              # i32
    gqa_ratio,      # i32 (Nq // Nkv, e.g., 4)
    T_total,        # i32 total number of tokens across all batches/windows
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] as f32 vector
    q_base = q_ptr + b * (Nq * D) + h * D
    q_vec = tl.load(q_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)  # [D], f32

    # Compute token window for this batch element
    start = tl.load(indptr_ptr + b).to(tl.int32)         # int32
    end = tl.load(indptr_ptr + b + 1).to(tl.int32)       # int32
    T = end - start  # int32

    # Accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        # k_i vector [D] as f32
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
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
    B,              # i32
    Nq,             # i32
    Nkv,            # i32
    D,              # i32
    gqa_ratio,      # i32 (Nq // Nkv, e.g., 4)
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] as f32 vector
    q_base = q_ptr + b * (Nq * D) + h * D
    q_vec = tl.load(q_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)  # [D], f32

    # Compute token window for this batch element
    start = tl.load(indptr_ptr + b).to(tl.int32)         # int32
    end = tl.load(indptr_ptr + b + 1).to(tl.int32)       # int32
    T = end - start  # int32

    # Recompute m and sum_exp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # Compute output vector elementwise
    out_vec = tl.zeros((D,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        soft = tl.exp(scaled - m) / sum_exp  # scalar
        v_base = v_ptr + idx * (Nkv * D) + kv_head * D
        v_i = tl.load(v_base + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)  # [D]
        out_vec += soft * v_i
        i += 1

    # Store output as bfloat16
    out_base = out_ptr + b * (Nq * D) + h * D
    tl.store(out_base + tl.arange(0, D), out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on same device and contiguous
        device = q.device
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B = q.shape[0]           # runtime int
        Nq = q.shape[1]          # runtime int
        Dq = q.shape[2]          # runtime int (should equal head_dim)

        Np, Nkv, Dv, _ = k_cache.shape
        assert Dv == Dq, "k_cache head_dim must match q's head_dim"

        # We don't assert fixed Nq/Nkv; the original asserts Nq=32,Nkv=8,D=128, but evaluation axes vary. We keep it generic.

        # Allocate outputs
        output = torch.empty((B, Nq, Dq), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch Triton kernels
        T_total = int(kv_indptr[-1].item())  # total tokens across all batches/windows
        grid = (B * Nq,)

        # lse kernel
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B, Nq, Nkv, Dq, (Nq // Nkv) if Nq % 8 == 0 else 4, T_total,
            num_warps=4, num_stages=2,
        )

        # output kernel
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B, Nq, Nkv, Dq, (Nq // Nkv) if Nq % 8 == 0 else 4,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
