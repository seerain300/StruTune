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
    gqa_ratio: tl.constexpr,  # Nq // Nkv (e.g., 4)
    stride_q_b, stride_q_h, stride_q_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Token window [start, end)
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # Accumulate m and sum_exp for logsumexp
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

    # Store lse[b, h] = log(sum_exp) + m (note: original may divide by log(2); we match PyTorch run behavior without that division)
    lse_val = tl.log(sum_exp) + m
    tl.store(lse_ptr + b * Nq + h, lse_val)


@triton.jit
def output_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D]
    v_ptr,          # *bf16, [Np, Nkv, D]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    lse_ptr,        # *f32,  [B, Nq]  (unused for output, but could be recomputed if needed)
    out_ptr,        # *bf16, [B, Nq, D]
    B: tl.constexpr,          # batch size
    Nq: tl.constexpr,         # num_qo_heads
    Nkv: tl.constexpr,        # num_kv_heads
    D: tl.constexpr,          # head_dim
    gqa_ratio: tl.constexpr,  # Nq // Nkv (e.g., 4)
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_p, stride_k_h, stride_k_d,
    stride_v_p, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # Load q[b, h] vector of length D
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

    # Compute output vector elementwise: out[b, h, j] = sum_i (exp(scaled[i]-m)/sum_exp) * v_selected[i, j]
    # We will write the entire vector by looping j.
    for j in range(D):
        acc = tl.zeros((), dtype=tl.float32)
        i = 0
        while i < T:
            idx = tl.load(indices_ptr + start + i).to(tl.int32)
            v_base = v_ptr + idx * (Nkv * D) + kv_head * D
            v_i = tl.load(v_base + j * stride_v_d).to(tl.float32)  # scalar
            prob = tl.exp((tl.load(indices_ptr + start + i).to(tl.float32) * sm_scale - m) / sum_exp)  # placeholder to avoid "use of loop variable" issues; correct prob = exp(scaled[i] - m) / sum_exp
            # Note: We need scaled[i], not indices_ptr. Recompute scaled[i] here:
            k_base = k_ptr + idx * (Nkv * D) + kv_head * D
            k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)
            attn = tl.sum(q_vec * k_i, axis=0)
            scaled_i = attn * sm_scale
            prob = tl.exp(scaled_i - m) / sum_exp
            acc += prob * v_i
            i += 1
        # Store as bfloat16
        out_row_base = out_ptr + b * stride_out_b + h * stride_out_h
        tl.store(out_row_base + j * stride_out_d, acc.to(tl.bfloat16))

# Note: The placeholder prob computation used tl.load(indices_ptr) which is incorrect.
# We fix this by recomputing attn and scaled_i inside the inner loop. This adds some overhead
# but ensures correctness. For better performance, one could cache scaled values in a vector,
# but Triton does not support dynamic-length Python lists inside kernels; hence recomputation
# is the simplest correct approach.


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        device = q.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors."
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Squeeze middle dim to match original reference behavior: [Np, Nkv, D]
        # get_inputs creates k_cache/v_cache with shape [num_pages, 1, Nkv, D]
        k_cache = k_cache.squeeze(1)
        v_cache = v_cache.squeeze(1)

        # Derive shapes (constexpr for Triton)
        B = q.shape[0]
        Nq = q.shape[1]  # num_qo_heads
        D = q.shape[2]   # head_dim
        Np = k_cache.shape[0]
        Nkv = k_cache.shape[1]  # num_kv_heads
        gqa_ratio = Nq // Nkv   # expected 4

        # Allocate outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch lse kernel: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale, lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=gqa_ratio,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            num_warps=4, num_stages=2
        )

        # Launch output kernel: one program per (b, h)
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, lse, output,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=gqa_ratio,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_k_p=k_cache.stride(0), stride_k_h=k_cache.stride(1), stride_k_d=k_cache.stride(2),
            stride_v_p=v_cache.stride(0), stride_v_h=v_cache.stride(1), stride_v_d=v_cache.stride(2),
            stride_out_b=output.stride(0), stride_out_h=output.stride(1), stride_out_d=output.stride(2),
            num_warps=4, num_stages=2
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
