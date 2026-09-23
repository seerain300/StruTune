import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: compute logsumexp (base-2) over all expanded KV positions per (b, q_token, qo_head), masked and scaled.
if TRITON_AVAILABLE:
    @triton.jit
    def compute_lse_kernel(
        q_ptr, k_ptr, v_ptr,           # inputs (float32)
        qo_indptr_ptr, kv_indptr_ptr,  # int32 arrays, length len_indptr+1
        lse_ptr,                       # output lse: [len_indptr, total_q, 32], float32
        sm_scale,                      # float32 scalar
        total_q, total_kv, len_indptr,  # ints
        num_qo_heads, num_kv_heads,     # ints (32, 8)
        head_dim,                       # int (128)
        gqa_ratio,                      # int (4, compile-time for loop)
    ):
        b = tl.program_id(0)
        q_token = tl.program_id(1)
        qo_head = tl.program_id(2)

        qo_start = tl.load(qo_indptr_ptr + b)   # int32
        qo_end = tl.load(qo_indptr_ptr + b + 1) # int32
        kv_start = tl.load(kv_indptr_ptr + b)   # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1) # int32

        if qo_start >= qo_end or kv_start >= kv_end:
            return

        num_q_tokens = qo_end - qo_start
        num_kv_tokens = kv_end - kv_start

        # Load q vector for this (b, q_token, qo_head)
        q_off = (b * (num_q_tokens * num_qo_heads) + q_token) * head_dim + qo_head * head_dim
        q_vec = tl.load(q_ptr + qo_start * (num_qo_heads * head_dim) + q_off)  # [head_dim]

        sum_exp = 0.0
        # Loop over original KV heads (8) and repeats (4); num_kv_heads and gqa_ratio are compile-time constants here
        for j in range(0, 8):  # num_kv_heads
            for r in range(0, 4):  # gqa_ratio
                kv_pos = j * 4 + r  # expanded KV position 0..31
                # Only consider positions within num_kv_tokens * 4 (always <= total_kv * 32)
                # No need explicit guard: we iterate 32, but expanded length is 32 for num_kv_tokens=8, gqa=4 => 32 <= 32
                k_off = kv_start * (8 * head_dim) + (j * head_dim) + kv_pos * head_dim
                k_vec = tl.load(k_ptr + k_off)  # [head_dim]
                # dot product: q_vec @ k_vec
                idx = tl.arange(0, head_dim)
                dot = tl.sum(q_vec[idx] * k_vec[idx], axis=0)
                # causal mask: q can attend up to (q_token + 1 + (num_kv_tokens - num_q_tokens))
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = q_token + 1 + delta
                if j < max_kv_idx:
                    logits = dot * sm_scale
                    sum_exp += tl.exp(logits)

        inv_log2 = 1.0 / math.log(2.0)
        lse_scalar = tl.log(sum_exp) * inv_log2

        # Store lse[b, q_token, qo_head]
        lse_index = b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head
        tl.store(lse_ptr + lse_index, lse_scalar)


    # Kernel 2: compute output per (b, q_token, qo_head): softmax over masked logits, then dot with expanded V
    @triton.jit
    def compute_output_kernel(
        q_ptr, k_ptr, v_ptr,           # inputs (float32)
        qo_indptr_ptr, kv_indptr_ptr,  # int32 arrays, length len_indptr+1
        output_ptr,                    # output: [total_q, 32, 128], float32
        sm_scale,                      # float32 scalar
        total_q, total_kv, len_indptr,  # ints
        num_qo_heads, num_kv_heads,     # ints (32, 8)
        head_dim,                       # int (128)
        gqa_ratio,                      # int (4, compile-time for loop)
    ):
        b = tl.program_id(0)
        q_token = tl.program_id(1)
        qo_head = tl.program_id(2)

        qo_start = tl.load(qo_indptr_ptr + b)   # int32
        qo_end = tl.load(qo_indptr_ptr + b + 1) # int32
        kv_start = tl.load(kv_indptr_ptr + b)   # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1) # int32

        if qo_start >= qo_end or kv_start >= kv_end:
            return

        num_q_tokens = qo_end - qo_start
        num_kv_tokens = kv_end - kv_start

        # Load q vector for this (b, q_token, qo_head)
        q_off = (b * (num_q_tokens * num_qo_heads) + q_token) * head_dim + qo_head * head_dim
        q_vec = tl.load(q_ptr + qo_start * (num_qo_heads * head_dim) + q_off)  # [head_dim]

        # Compute max for numerical stability over masked logits
        max_log = -float("inf")
        for j in range(0, 8):
            for r in range(0, 4):
                kv_pos = j * 4 + r
                k_off = kv_start * (8 * head_dim) + (j * head_dim) + kv_pos * head_dim
                k_vec = tl.load(k_ptr + k_off)  # [head_dim]
                dot = tl.sum(q_vec * k_vec, axis=0)
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = q_token + 1 + delta
                if j < max_kv_idx:
                    logits = dot * sm_scale
                    max_log = tl.maximum(max_log, logits)

        # Compute sum of exp(logits - max_log)
        sum_exp = 0.0
        for j in range(0, 8):
            for r in range(0, 4):
                kv_pos = j * 4 + r
                k_off = kv_start * (8 * head_dim) + (j * head_dim) + kv_pos * head_dim
                k_vec = tl.load(k_ptr + k_off)  # [head_dim]
                dot = tl.sum(q_vec * k_vec, axis=0)
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = q_token + 1 + delta
                if j < max_kv_idx:
                    logits = dot * sm_scale
                    sum_exp += tl.exp(logits - max_log)

        # Compute and store output vector
        out_vec = tl.zeros([head_dim], dtype=tl.float32)
        for j in range(0, 8):
            for r in range(0, 4):
                kv_pos = j * 4 + r
                k_off = kv_start * (8 * head_dim) + (j * head_dim) + kv_pos * head_dim
                k_vec = tl.load(k_ptr + k_off)  # [head_dim]
                dot = tl.sum(q_vec * k_vec, axis=0)
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = q_token + 1 + delta
                if j < max_kv_idx:
                    logits = dot * sm_scale
                    alpha = tl.exp(logits - max_log) / sum_exp
                    # expanded V: v[j, r, :] for this batch slice starts at kv_start
                    v_off = kv_start * (8 * head_dim) + (j * head_dim) + (r * head_dim)
                    v_vec = tl.load(v_ptr + v_off)  # [head_dim]
                    out_vec += alpha * v_vec

        # Store output[b, q_token, qo_head, :]
        out_linear = ((b * total_q + q_token) * num_qo_heads + qo_head) * head_dim
        tl.store(output_ptr + out_linear, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Triton must be available and inputs on CUDA
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"

        # Ensure contiguity and dtype
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)

        # Shapes
        total_q = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        total_kv = k.shape[0]
        num_kv_heads = k.shape[1]
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"
        assert v.shape == (total_kv, num_kv_heads, head_dim), "v shape must match k"
        assert qo_indptr.shape[0] == kv_indptr.shape[0], "len_indptr must match for qo and kv"
        len_indptr = qo_indptr.shape[0]
        assert total_q == int(qo_indptr[-1].item()), "total_q must equal last qo_indptr"
        assert total_kv == int(kv_indptr[-1].item()), "total_kv must equal last kv_indptr"

        # Allocate outputs
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.empty((len_indptr, total_q, 32), dtype=torch.float32, device=q.device)

        # Launch Triton kernels
        grid = (len_indptr, total_q, 32)
        compute_lse_kernel[grid](
            q, k, v,
            qo_indptr, kv_indptr,
            lse,
            sm_scale,
            total_q, total_kv, len_indptr,
            num_qo_heads, num_kv_heads,
            head_dim,
            4,  # gqa_ratio
            num_warps=1, num_stages=1
        )

        compute_output_kernel[grid](
            q, k, v,
            qo_indptr, kv_indptr,
            output,
            sm_scale,
            total_q, total_kv, len_indptr,
            num_qo_heads, num_kv_heads,
            head_dim,
            4,  # gqa_ratio
            num_warps=1, num_stages=1
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
