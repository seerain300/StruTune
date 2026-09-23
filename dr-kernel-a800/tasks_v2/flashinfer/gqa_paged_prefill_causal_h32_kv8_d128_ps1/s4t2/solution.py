import torch
import triton
import triton.language as tl


@triton.jit
def attn_gqa_token_kernel(
    q_ptr,          # *float32, shape: [num_q_tokens, num_qo_heads, head_dim], contiguous in last dim
    k_ptr,          # *float32, shape: [num_kv_tokens, num_kv_heads, head_dim], contiguous
    v_ptr,          # *float32, shape: [num_kv_tokens, num_kv_heads, head_dim], contiguous
    out_ptr,        # *bfloat16, shape: [num_q_tokens, num_qo_heads, head_dim]
    lse_ptr,        # *float32, shape: [num_q_tokens, num_qo_heads]
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
    q_token_base = (t - q_start) * q_batch_stride  # since t is 0..num_q_tokens-1

    # Process each query head
    for h in range(num_qo_heads):
        kv_head = h // gqa_ratio  # 0..7

        # Initialize accumulators for LSE (logsumexp) and output vector
        max_logits = tl.full((), -float("inf"), tl.float32)
        sum_exp = tl.zeros((), dtype=tl.float32)
        out_vec = tl.zeros([head_dim], dtype=tl.float32)

        # Loop over kv tokens to compute logsumexp
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

            logits_scaled = dot * sm_scale
            # Update max and sum for logsumexp
            max_logits = tl.maximum(max_logits, logits_scaled)
            exp_term = tl.exp(logits_scaled - max_logits)
            sum_exp += exp_term

        # Now compute softmax per token and accumulate output
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

            # Load v_vec (128 elements)
            v_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                v_elem = tl.load(v_ptr + kv_start * kv_head_stride + k_offset + j)
                v_vec[j] = v_elem

            # Accumulate output: out_vec += soft * v_vec
            for j in range(head_dim):
                out_vec[j] += soft * v_vec[j]

        # Store LSE and output
        lse_base = t * num_qo_heads + h
        tl.store(lse_ptr + lse_base, max_logits + tl.log(sum_exp))
        # Store output vector as bfloat16
        for j in range(head_dim):
            tl.store(out_ptr + t * (num_qo_heads * head_dim) + h * head_dim + j,
                     out_vec[j].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q.device
        # Cast to float32 for math
        q_f32 = q.to(torch.float32)
        # Flatten k_cache/v_cache along page dimension
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, 8, 128]

        total_q = int(qo_indptr[-1].item())
        num_qo_heads = q_f32.shape[1]
        assert num_qo_heads == 32
        head_dim = q_f32.shape[2]
        assert head_dim == 128
        assert k_cache_flat.shape[2] == head_dim
        assert v_cache_flat.shape[2] == head_dim
        num_kv_heads = 8
        gqa_ratio = 4

        output = torch.empty(
            (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]
        # For each batch defined by indptr
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if (q_end - q_start) <= 0 or (kv_end - kv_start) <= 0:
                continue

            # Slice q for this batch
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]
            num_q_tokens = q_batch.shape[0]

            # Gather k and v based on kv_indices for this batch using torch (data movement, not compute)
            num_kv_tokens = kv_end - kv_start
            kv_indices_batch = kv_indices[kv_start:kv_end].to(torch.int64)  # indices are int32 already
            k_batch = k_cache_flat.index_select(0, kv_indices_batch)  # [num_kv_tokens, 8, 128]
            v_batch = v_cache_flat.index_select(0, kv_indices_batch)  # [num_kv_tokens, 8, 128]

            # Launch Triton kernel: one program per token in this batch
            grid = (num_q_tokens,)
            attn_gqa_token_kernel[grid](
                q_batch, k_batch, v_batch,
                output, lse,
                num_qo_heads=32, num_kv_heads=8, head_dim=128, gqa_ratio=4, sm_scale=float(sm_scale),
                q_start=q_start, num_q_tokens=num_q_tokens,
                kv_start=kv_start, num_kv_tokens=num_kv_tokens,
                kv_indices_ptr=kv_indices_batch,
                q_batch_stride=num_qo_heads * head_dim, kv_head_stride=head_dim,
                num_warps=1, num_stages=2
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
