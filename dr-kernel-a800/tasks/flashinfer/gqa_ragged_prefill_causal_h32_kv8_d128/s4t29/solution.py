import torch
import math
import triton
import triton.language as tl

# Kernel 1: compute lse for each (b, q_token, qo_head)
@triton.jit
def compute_lse_kernel(
    q_ptr, k_exp_ptr, lse_ptr,
    total_q: tl.int32, num_qo_heads: tl.int32, head_dim: tl.int32,
    sm_scale: tl.float32, out_len: tl.constexpr, gqa_ratio: tl.int32
):
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    if (b >= 1 or q_token >= total_q or qo_head >= num_qo_heads):
        return

    # Load Q vector for this (q_token, qo_head)
    q_base = (q_token * num_qo_heads + qo_head) * head_dim
    offs = tl.arange(0, head_dim)
    q_vec = tl.load(q_ptr + q_base + offs)  # shape: [head_dim], vector

    # First pass: compute max over valid logits (j <= q_token)
    max_val = -float('inf')
    for kv_pos in tl.static_range(0, out_len):
        j = kv_pos // gqa_ratio
        r = kv_pos % gqa_ratio
        k_base = kv_pos * head_dim + qo_head * head_dim
        k_vec = tl.load(k_exp_ptr + k_base + offs)  # [head_dim]
        dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
        if j <= q_token:
            if dot > max_val:
                max_val = dot

    # Second pass: compute sum(exp(dot - max_val)) over valid kv positions
    sum_exp = 0.0
    for kv_pos in tl.static_range(0, out_len):
        j = kv_pos // gqa_ratio
        r = kv_pos % gqa_ratio
        k_base = kv_pos * head_dim + qo_head * head_dim
        k_vec = tl.load(k_exp_ptr + k_base + offs)
        dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
        if j <= q_token:
            sum_exp += tl.exp(dot - max_val)

    # lse = max + log(sum) / log(2)
    lse_val = max_val + tl.log(sum_exp) / math.log(2.0)

    # Store lse
    lse_offset = b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head
    tl.store(lse_ptr + lse_offset, lse_val)

# Kernel 2: compute output for each (b, q_token, qo_head) using lse
@triton.jit
def compute_output_kernel(
    q_ptr, k_exp_ptr, v_exp_ptr, lse_ptr, output_ptr,
    total_q: tl.int32, num_qo_heads: tl.int32, head_dim: tl.int32,
    sm_scale: tl.float32, out_len: tl.constexpr, gqa_ratio: tl.int32
):
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    if (b >= 1 or q_token >= total_q or qo_head >= num_qo_heads):
        return

    # Load Q vector
    q_base = (q_token * num_qo_heads + qo_head) * head_dim
    offs = tl.arange(0, head_dim)
    q_vec = tl.load(q_ptr + q_base + offs)  # [head_dim]

    # Load lse
    lse_offset = b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head
    lse_val = tl.load(lse_ptr + lse_offset)

    # Output vector
    out_vec = tl.zeros([head_dim], dtype=tl.float32)

    # Compute normalization sum over valid kv positions
    sum_exp = 0.0
    for kv_pos in tl.static_range(0, out_len):
        j = kv_pos // gqa_ratio
        r = kv_pos % gqa_ratio
        k_base = kv_pos * head_dim + qo_head * head_dim
        k_vec = tl.load(k_exp_ptr + k_base + offs)
        dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
        if j <= q_token:
            sum_exp += tl.exp(dot - lse_val)

    # Now accumulate output for each kv_pos
    for kv_pos in tl.static_range(0, out_len):
        j = kv_pos // gqa_ratio
        r = kv_pos % gqa_ratio
        k_base = kv_pos * head_dim + qo_head * head_dim
        k_vec = tl.load(k_exp_ptr + k_base + offs)
        dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
        if j <= q_token:
            attn = tl.exp(dot - lse_val) / sum_exp
            v_base = kv_pos * (num_qo_heads * head_dim) + qo_head * head_dim
            v_vec = tl.load(v_exp_ptr + v_base + offs)  # [head_dim]
            out_vec += attn * v_vec

    # Store output: [b, q_token, qo_head, :]
    out_offset = b * (total_q * num_qo_heads * head_dim) + q_token * (num_qo_heads * head_dim) + qo_head * head_dim
    tl.store(output_ptr + out_offset + offs, out_vec)

class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors and dtypes; convert to float32 for computation
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors."
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)

        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        len_indptr = qo_indptr.shape[0]
        assert kv_indptr.shape[0] == len_indptr, "qo_indptr and kv_indptr must have the same length"
        assert qo_indptr[-1].item() == total_q, "qo_indptr[-1] must equal total_q"
        assert kv_indptr[-1].item() == total_kv, "kv_indptr[-1] must equal total_kv"

        # Expand K and V by GQA ratio along head dim: shape [out_len, head_dim]
        out_len = total_kv * gqa_ratio
        k_exp = k.repeat_interleave(gqa_ratio, dim=1).contiguous()
        v_exp = v.repeat_interleave(gqa_ratio, dim=1).contiguous()

        # Allocate lse and output
        lse = torch.empty((len_indptr, total_q, num_qo_heads), dtype=torch.float32, device=q.device)
        output = torch.empty((len_indptr, total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)

        # Launch compute_lse_kernel: grid over (len_indptr, total_q, num_qo_heads)
        grid_lse = (len_indptr, total_q, num_qo_heads)
        compute_lse_kernel[grid_lse](
            q, k_exp, lse,
            total_q, num_qo_heads, head_dim,
            float(sm_scale), out_len, gqa_ratio,
            num_warps=4
        )

        # Launch compute_output_kernel
        grid_out = (len_indptr, total_q, num_qo_heads)
        compute_output_kernel[grid_out](
            q, k_exp, v_exp, lse, output,
            total_q, num_qo_heads, head_dim,
            float(sm_scale), out_len, gqa_ratio,
            num_warps=4
        )

        # Return output as bfloat16, lse as float32 (mirroring original behavior of returning both)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
