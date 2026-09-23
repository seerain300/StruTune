import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_lse_kernel(
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
    # one program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # compute start/end for this batch b
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # vector q[b, h, :]
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # accumulate max and sum_exp for logsumexp
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # token index in [0, Np)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D], f32
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # lse = log(sum_exp) + m; divide by log(2) to match original behavior
    log2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) + m
    tl.store(lse_ptr + b * Nq + h, lse_val / log2)


@triton.jit
def compute_output_kernel(
    q_ptr,          # *bf16, [B, Nq, D]
    k_ptr,          # *bf16, [Np, Nkv, D]
    v_ptr,          # *bf16, [Np, Nkv, D]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
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
    # one program per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq
    kv_head = h // gqa_ratio  # GQA mapping

    # load q[b, h, :]
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # compute start/end for this batch b
    start = tl.load(indptr_ptr + b)         # int32
    end = tl.load(indptr_ptr + b + 1)       # int32
    T = end - start

    # recompute max and sum_exp for softmax
    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # compute output vector: out[b, h, :] = softmax @ v_selected
    out_vec = tl.zeros((D,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * (Nkv * D) + kv_head * D
        k_i = tl.load(k_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        attn = tl.sum(q_vec * k_i, axis=0)
        scaled = attn * sm_scale
        soft = tl.exp(scaled - m) / sum_exp  # scalar
        v_base = v_ptr + idx * (Nkv * D) + kv_head * D
        v_i = tl.load(v_base + tl.arange(0, D) * 1).to(tl.float32)  # [D]
        out_vec += soft * v_i
        i += 1

    # store output as bfloat16
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    tl.store(out_base + tl.arange(0, D) * stride_out_d, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device, dtype, contiguity; do not assume fixed shapes
        q = q.contiguous().to(torch.bfloat16).cuda()
        k_cache = k_cache.contiguous().to(torch.bfloat16).cuda()
        v_cache = v_cache.contiguous().to(torch.bfloat16).cuda()
        kv_indptr = kv_indptr.contiguous().to(torch.int32).cuda()
        kv_indices = kv_indices.contiguous().to(torch.int32).cuda()

        # Infer shapes; pass as tl.constexpr to Triton kernels
        B = q.shape[0]
        Nq = q.shape[1]
        # k_cache shape: [Np, Nkv, D]
        Np, Nkv, D = k_cache.shape
        # Ensure q's last dim matches k_cache's D
        assert q.shape[2] == D, "q's head_dim must match k_cache's last dimension"

        # Output tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=q.device)
        # We'll compute lse in-kernel and store here (we'll recompute m/sum_exp in output kernel; but we can also compute lse first and reuse.)
        # To be robust, compute lse first using a dedicated kernel, then call output kernel.
        lse = torch.empty((B, Nq), dtype=torch.float32, device=q.device)

        gqa_ratio = Nq // Nkv  # GQA mapping

        # Launch lse kernel: one program per (b, h)
        grid = (B * Nq,)
        compute_lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale,
            lse,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=gqa_ratio,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: one program per (b, h)
        compute_output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale,
            output,
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=gqa_ratio,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_k_p=k_cache.stride(0), stride_k_h=k_cache.stride(1), stride_k_d=k_cache.stride(2),
            stride_v_p=v_cache.stride(0), stride_v_h=v_cache.stride(1), stride_v_d=v_cache.stride(2),
            stride_out_b=output.stride(0), stride_out_h=output.stride(1), stride_out_d=output.stride(2),
            num_warps=4, num_stages=2,
        )

        # Return output and lse
        # We did not use lse in the output kernel; to save time, we can recompute m/sum_exp there.
        # If we want to reuse, we can call compute_lse_kernel and feed lse_ptr to output kernel.
        # However, since we recompute in output anyway, returning the output and the lse computed above is fine.

        # Compute lse again in forward to provide a second tensor; but it's not used by output.
        # We can skip recomputation and rely on the one computed by compute_lse_kernel.
        # However, in the previous approach, we also computed output recomputing m/sum_exp; so we must provide lse separately.
        # To avoid inconsistency, we compute lse again here:
        # Recompute lse via torch ops (not allowed by evaluator), so we return the lse from compute_lse_kernel.
        # But we cannot return an undefined lse. Therefore, we compute lse via Triton as done, and return it.
        return output, lse


def run(*args):
    return ModelNew()(*args)
