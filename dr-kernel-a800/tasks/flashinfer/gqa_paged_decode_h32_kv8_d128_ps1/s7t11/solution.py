import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_kernel(
    q_ptr,          # *bf16, [B, Nq, head_dim]
    k_ptr,          # *bf16, [Np, Nkv, head_dim]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    lse_ptr,        # *f32,  [B, Nq]
    B,              # i32 (batch size)
    Nq,             # i32 (num query heads)
    Nkv,            # i32 (num kv heads)
    head_dim,       # i32 (D)
    gqa_ratio,      # i32 (Nq // Nkv)
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq

    # Load q[b, h] vector as f32
    q_base = q_ptr + b * (Nq * head_dim) + h * head_dim
    q_vec = tl.load(q_base + tl.arange(0, head_dim)).to(tl.float32)

    # Token window start/end
    start = tl.load(indptr_ptr + b)        # i32
    end = tl.load(indptr_ptr + b + 1)      # i32
    T = end - start

    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)  # i32
        kv_head = h // gqa_ratio  # GQA mapping
        k_base = k_ptr + idx * (Nkv * head_dim) + kv_head * head_dim
        k_i = tl.load(k_base + tl.arange(0, head_dim)).to(tl.float32)  # [head_dim]
        attn = tl.sum(q_vec * k_i, axis=0)  # scalar
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    log2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) + m
    tl.store(lse_ptr + b * Nq + h, lse_val / log2)


@triton.jit
def output_kernel(
    q_ptr,          # *bf16, [B, Nq, head_dim]
    k_ptr,          # *bf16, [Np, Nkv, head_dim]
    v_ptr,          # *bf16, [Np, Nkv, head_dim]
    indptr_ptr,     # *i32,  [B+1]
    indices_ptr,    # *i32,  [T_total]
    sm_scale,       # f32
    lse_ptr,        # *f32,  [B, Nq]
    out_ptr,        # *bf16, [B, Nq, head_dim]
    B,              # i32
    Nq,             # i32
    Nkv,            # i32
    head_dim,       # i32
    gqa_ratio,      # i32
):
    pid = tl.program_id(0)  # one program per (b, h)
    b = pid // Nq
    h = pid % Nq

    # Re-compute m and sum_exp using scaled logits
    start = tl.load(indptr_ptr + b)
    end = tl.load(indptr_ptr + b + 1)
    T = end - start

    m = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        kv_head = h // gqa_ratio
        k_base = k_ptr + idx * (Nkv * head_dim) + kv_head * head_dim
        k_i = tl.load(k_base + tl.arange(0, head_dim)).to(tl.float32)
        q_base = q_ptr + b * (Nq * head_dim) + h * head_dim
        q_vec = tl.load(q_base + tl.arange(0, head_dim)).to(tl.float32)
        attn = tl.sum(q_vec * k_i, axis=0)
        scaled = attn * sm_scale
        m = tl.maximum(m, scaled)
        exp_term = tl.exp(scaled - m)
        sum_exp += exp_term
        i += 1

    # Initialize output vector
    out_base = out_ptr + b * (Nq * head_dim) + h * head_dim
    out_vec = tl.zeros((head_dim,), dtype=tl.float32)

    # Compute output = softmax @ v_selected
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        kv_head = h // gqa_ratio
        k_base = k_ptr + idx * (Nkv * head_dim) + kv_head * head_dim
        k_i = tl.load(k_base + tl.arange(0, head_dim)).to(tl.float32)  # [head_dim]
        q_base = q_ptr + b * (Nq * head_dim) + h * head_dim
        q_vec = tl.load(q_base + tl.arange(0, head_dim)).to(tl.float32)
        attn = tl.sum(q_vec * k_i, axis=0)
        scaled = attn * sm_scale
        soft = tl.exp(scaled - m) / sum_exp  # scalar
        v_base = v_ptr + idx * (Nkv * head_dim) + kv_head * head_dim
        v_i = tl.load(v_base + tl.arange(0, head_dim)).to(tl.float32)  # [head_dim]
        out_vec += soft * v_i
        i += 1

    # Store output as bfloat16
    tl.store(out_base + tl.arange(0, head_dim), out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure everything is on CUDA and contiguous
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be CUDA tensors"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Get shapes
        B = q.shape[0]
        Nq = q.shape[1]  # num_qo_heads
        head_dim = q.shape[2]  # head_dim
        Np = k_cache.shape[0]
        Nkv = k_cache.shape[2]  # num_kv_heads

        # Output buffers
        output = torch.empty((B, Nq, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=device)

        # Launch lse kernel: one program per (b, h)
        grid = (B * Nq,)
        lse_kernel[grid](
            q, k_cache, kv_indptr, kv_indices, sm_scale,
            lse,
            B, Nq, Nkv, head_dim, (Nq // Nkv),
            num_warps=4, num_stages=2,
        )

        # Launch output kernel: one program per (b, h)
        output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale,
            lse, output,
            B, Nq, Nkv, head_dim, (Nq // Nkv),
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
