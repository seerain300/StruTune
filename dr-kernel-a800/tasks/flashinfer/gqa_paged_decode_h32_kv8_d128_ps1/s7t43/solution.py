import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_kernel(
    q_ptr,          # *bf16, [B, Nq, 128] (we'll enforce 128)
    k_ptr,          # *bf16, [Np, 8, 128]  (we'll enforce 8 kv_heads, 128 dim)
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    lse_ptr,        # *f32,  [B, Nq]
    B, Nq,          # runtime ints
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq

    # Window [start, end)
    start = tl.load(indptr_ptr + b)        # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    # GQA mapping: kv_head = h // (Nq // 8)
    gqa_ratio = Nq // 8
    kv_head = h // gqa_ratio

    # Initialize m and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    # Iterate over tokens
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index
        # Load q[b, h, :] vector as f32 (scalar loads over D=128)
        q_base = q_ptr + b * Nq * 128 + h * 128
        dot = tl.zeros((), dtype=tl.float32)
        j = 0
        while j < 128:
            q_val = tl.load(q_base + j).to(tl.float32)
            k_val = tl.load(k_ptr + idx * (8 * 128) + kv_head * 128 + j).to(tl.float32)
            dot += q_val * k_val
            j += 1
        scaled = dot * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # lse = log(sum_exp) + m; divide by log(2)
    log2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) + m
    tl.store(lse_ptr + b * Nq + h, lse_val / log2)


@triton.jit
def output_kernel(
    q_ptr,          # *bf16, [B, Nq, 128]
    k_ptr,          # *bf16, [Np, 8, 128]
    v_ptr,          # *bf16, [Np, 8, 128]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    lse_ptr,        # *f32,  [B, Nq]
    out_ptr,        # *bf16, [B, Nq, 128]
    B, Nq,          # runtime ints
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq

    start = tl.load(indptr_ptr + b)        # int32
    end = tl.load(indptr_ptr + b + 1)      # int32
    T = end - start

    gqa_ratio = Nq // 8
    kv_head = h // gqa_ratio

    # Recompute m and sum_exp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    # First pass to compute m and sum_exp
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        q_base = q_ptr + b * Nq * 128 + h * 128
        dot = tl.zeros((), dtype=tl.float32)
        j = 0
        while j < 128:
            q_val = tl.load(q_base + j).to(tl.float32)
            k_val = tl.load(k_ptr + idx * (8 * 128) + kv_head * 128 + j).to(tl.float32)
            dot += q_val * k_val
            j += 1
        scaled = dot * sm_scale
        m = tl.maximum(m, scaled)
        sum_exp += tl.exp(scaled - m)
        i += 1

    # Second pass: compute softmax and output
    out_vec = tl.zeros((128,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        q_base = q_ptr + b * Nq * 128 + h * 128
        dot = tl.zeros((), dtype=tl.float32)
        j = 0
        while j < 128:
            q_val = tl.load(q_base + j).to(tl.float32)
            k_val = tl.load(k_ptr + idx * (8 * 128) + kv_head * 128 + j).to(tl.float32)
            dot += q_val * k_val
            j += 1
        scaled = dot * sm_scale
        soft_i = tl.exp(scaled - m) / sum_exp
        v_base = v_ptr + idx * (8 * 128) + kv_head * 128
        j = 0
        while j < 128:
            v_val = tl.load(v_base + j).to(tl.float32)
            out_vec[j] += soft_i * v_val
            j += 1
        i += 1

    # Store output as bfloat16
    out_base = out_ptr + b * Nq * 128 + h * 128
    j = 0
    while j < 128:
        tl.store(out_base + j, out_vec[j].to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and contiguity
        device = q.device
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B = q.shape[0]
        Nq = q.shape[1]
        # Enforce D=128 by layout; we assume head_dim=128 as per original get_inputs
        D = q.shape[2]
        assert D == 128, "This Triton implementation expects head_dim=128."

        # Outputs
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch kernels: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, float(sm_scale), lse,
            B, Nq,
            num_warps=4, num_stages=1,
        )
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, float(sm_scale), lse, output,
            B, Nq,
            num_warps=4, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
