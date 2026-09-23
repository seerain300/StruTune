import torch
import triton
import triton.language as tl


@triton.jit
def compute_lse_kernel(
    q_ptr,          # *float32, shape: [num_q_tokens, num_qo_heads, head_dim], contiguous in last dim
    k_ptr,          # *float32, shape: [num_kv_tokens, num_kv_heads, head_dim], contiguous
    lse_max_ptr,    # *float32, shape: [num_q_tokens, num_qo_heads]
    lse_sum_ptr,    # *float32, shape: [num_q_tokens, num_qo_heads]
    # meta-parameters
    num_qo_heads: tl.constexpr,   # 32
    num_kv_heads: tl.constexpr,   # 8
    head_dim: tl.constexpr,       # 128
    gqa_ratio: tl.constexpr,      # 4
    # runtime parameters
    q_start,                      # int32: start token index for this batch
    num_q_tokens,                 # int32: number of tokens in this batch
    kv_start,                     # int32: start kv index for this batch
    num_kv_tokens,                # int32: number of kv tokens in this batch
    kv_indices_ptr,               # *int32: [num_kv_tokens]
    q_batch_stride,               # int32: stride between tokens in q_ptr (num_qo_heads * head_dim)
    kv_head_stride,               # int32: stride between heads in k_ptr/v_ptr (head_dim)
):
    # One program per token in this batch
    t = tl.program_id(0)
    if t >= num_q_tokens:
        return

    # Base offset for this token in q_ptr
    q_token_base = (t - q_start) * q_batch_stride  # t is 0..num_q_tokens-1

    # Initialize accumulators for LSE (logsumexp)
    max_logits = tl.full((), -float("inf"), tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    # Process each query head
    for h in range(num_qo_heads):
        kv_head = h // gqa_ratio  # 0..7

        # Pass 1: compute max and sum_exp across kv tokens
        for i in range(num_kv_tokens):
            # Gather kv_index for this batch element
            kv_index = tl.load(kv_indices_ptr + i)  # int32
            # Compute offset in k_ptr/v_ptr: ((kv_index * num_kv_heads + kv_head) * head_dim)
            k_offset = (kv_index * num_kv_heads + kv_head) * kv_head_stride

            # Load q_vec[h] (128 elements)
            q_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                q_elem = tl.load(q_ptr + q_token_base + h * kv_head_stride + j)
                q_vec[j] = q_elem

            # Load k_vec (128 elements)
            k_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                k_elem = tl.load(k_ptr + kv_start * kv_head_stride + k_offset + j)
                k_vec[j] = k_elem

            # Dot product
            dot = tl.zeros((), dtype=tl.float32)
            for j in range(head_dim):
                dot += q_vec[j] * k_vec[j]

            logits_scaled = dot * sm_scale  # sm_scale will be passed as runtime scalar later
            max_logits = tl.maximum(max_logits, logits_scaled)
            exp_term = tl.exp(logits_scaled - max_logits)
            sum_exp += exp_term

        # Store max and sum for this (token, head)
        lse_max_base = t * num_qo_heads + h
        tl.store(lse_max_ptr + lse_max_base, max_logits)
        tl.store(lse_sum_ptr + lse_max_base, sum_exp)


@triton.jit
def compute_output_kernel(
    q_ptr,          # *float32, shape: [num_q_tokens, num_qo_heads, head_dim]
    k_ptr,          # *float32, shape: [num_kv_tokens, num_kv_heads, head_dim]
    lse_max_ptr,    # *float32, shape: [num_q_tokens, num_qo_heads]
    lse_sum_ptr,    # *float32, shape: [num_q_tokens, num_qo_heads]
    v_ptr,          # *float32, shape: [num_kv_tokens, num_kv_heads, head_dim]
    out_ptr,        # *bfloat16, shape: [num_q_tokens, num_qo_heads, head_dim]
    # meta-parameters
    num_qo_heads: tl.constexpr,   # 32
    num_kv_heads: tl.constexpr,   # 8
    head_dim: tl.constexpr,       # 128
    gqa_ratio: tl.constexpr,      # 4
    sm_scale,                     # float32 scalar
    # runtime parameters
    q_start,                      # int32: start token index for this batch
    num_q_tokens,                 # int32: number of tokens in this batch
    kv_start,                     # int32: start kv index for this batch
    num_kv_tokens,                # int32: number of kv tokens in this batch
    kv_indices_ptr,               # *int32: [num_kv_tokens]
    q_batch_stride,               # int32: stride between tokens in q_ptr (num_qo_heads * head_dim)
    kv_head_stride,               # int32: stride between heads in k_ptr/v_ptr (head_dim)
):
    # One program per token in this batch
    t = tl.program_id(0)
    if t >= num_q_tokens:
        return

    # Base offset for this token in q_ptr
    q_token_base = (t - q_start) * q_batch_stride

    # Initialize output vector for each head
    for h in range(num_qo_heads):
        out_vec = tl.zeros([head_dim], dtype=tl.float32)

        kv_head = h // gqa_ratio  # 0..7

        max_logits = tl.load(lse_max_ptr + (t * num_qo_heads + h))
        sum_exp = tl.load(lse_sum_ptr + (t * num_qo_heads + h))

        # Pass 2: recompute logits, compute softmax, accumulate output
        for i in range(num_kv_tokens):
            kv_index = tl.load(kv_indices_ptr + i)
            k_offset = (kv_index * num_kv_heads + kv_head) * kv_head_stride

            q_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                q_elem = tl.load(q_ptr + q_token_base + h * kv_head_stride + j)
                q_vec[j] = q_elem

            k_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                k_elem = tl.load(k_ptr + kv_start * kv_head_stride + k_offset + j)
                k_vec[j] = k_elem

            dot = tl.zeros((), dtype=tl.float32)
            for j in range(head_dim):
                dot += q_vec[j] * k_vec[j]

            logits_scaled = dot * sm_scale
            soft = tl.exp(logits_scaled - max_logits) / sum_exp  # scalar

            v_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                v_elem = tl.load(v_ptr + kv_start * kv_head_stride + k_offset + j)
                v_vec[j] = v_elem

            # Accumulate output: out_vec += soft * v_vec
            for j in range(head_dim):
                out_vec[j] += soft * v_vec[j]

        # Store output vector as bfloat16
        for j in range(head_dim):
            tl.store(out_ptr + t * (num_qo_heads * head_dim) + h * head_dim + j,
                     out_vec[j].to(tl.bfloat16))


@triton.jit
def add_log_kernel(
    lse_sum_ptr,    # *float32, shape: [num_q_tokens, num_qo_heads]
    lse_max_ptr,    # *float32, shape: [num_q_tokens, num_qo_heads]
    out_lse_ptr,    # *float32, shape: [num_q_tokens, num_qo_heads]
    # meta-parameters
    num_qo_heads: tl.constexpr,   # 32
    # runtime parameters
    total_q,                # int32
):
    # One program per token and head
    t = tl.program_id(0)  # tokens
    h = tl.program_id(1)  # heads
    if (t >= total_q) or (h >= num_qo_heads):
        return
    lse_max = tl.load(lse_max_ptr + t * num_qo_heads + h)
    lse_sum = tl.load(lse_sum_ptr + t * num_qo_heads + h)
    lse = lse_max + tl.log(lse_sum)
    tl.store(out_lse_ptr + t * num_qo_heads + h, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA
        device = q.device
        assert device.type == "cuda", "ModelNew.forward requires CUDA tensors for Triton kernels"

        total_q, num_qo_heads, head_dim = q.shape
        # Cast q to float32 for stable math
        q_f32 = q.to(torch.float32)

        # Compute k_cache_flat, v_cache_flat: squeeze the 1 dimension to [num_pages, 8, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]

        len_indptr = qo_indptr.shape[0]
        # Allocate outputs
        output = torch.empty(
            (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        # We will compute lse via Triton: first store max and sum, then combine
        lse_max = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)
        lse_sum = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)
        # Final lse
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute gqa_ratio (num_qo_heads // 8)
        gqa_ratio = num_qo_heads // 8

        # Process batches defined by qo_indptr and kv_indptr
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            # Skip empty batches
            if (q_end - q_start) <= 0 or (kv_end - kv_start) <= 0:
                continue

            # Slice q for this batch
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]
            num_q_tokens = q_batch.shape[0]

            # Gather k and v based on kv_indices for this batch
            num_kv_tokens = kv_end - kv_start
            kv_indices_batch = kv_indices[kv_start:kv_end].to(torch.int64)  # [num_kv_tokens]
            k_batch = k_cache_flat.index_select(0, kv_indices_batch)  # [num_kv_tokens, 8, 128]
            v_batch = v_cache_flat.index_select(0, kv_indices_batch)  # [num_kv_tokens, 8, 128]

            # Launch Triton kernel to compute lse_max and lse_sum: one program per token
            grid = (num_q_tokens,)
            compute_lse_kernel[grid](
                q_batch, k_batch,
                lse_max, lse_sum,
                num_qo_heads=32, num_kv_heads=8, head_dim=128, gqa_ratio=4, sm_scale=float(sm_scale),
                q_start=q_start, num_q_tokens=num_q_tokens,
                kv_start=kv_start, num_kv_tokens=num_kv_tokens,
                kv_indices_ptr=kv_indices_batch,
                q_batch_stride=num_qo_heads * head_dim, kv_head_stride=head_dim,
                num_warps=1, num_stages=2
            )

            # Launch Triton kernel to compute final output
            compute_output_kernel[grid](
                q_batch, k_batch, lse_max, lse_sum, v_batch, output,
                num_qo_heads=32, num_kv_heads=8, head_dim=128, gqa_ratio=4, sm_scale=float(sm_scale),
                q_start=q_start, num_q_tokens=num_q_tokens,
                kv_start=kv_start, num_kv_tokens=num_kv_tokens,
                kv_indices_ptr=kv_indices_batch,
                q_batch_stride=num_qo_heads * head_dim, kv_head_stride=head_dim,
                num_warps=1, num_stages=2
            )

            # Combine lse_max and lse_sum into final lse using Triton (avoid host torch ops)
            grid_lse = (num_q_tokens, 32)
            add_log_kernel[grid_lse](
                lse_sum, lse_max, lse,
                num_qo_heads=32, total_q=total_q,
                num_warps=1, num_stages=1
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
