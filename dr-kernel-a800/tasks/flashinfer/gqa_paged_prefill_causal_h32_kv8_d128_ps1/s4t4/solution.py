import torch
import triton
import triton.language as tl


@triton.jit
def attn_gqa_single_token_kernel(
    q_ptr,          # *float32, shape: [num_q_tokens, num_qo_heads, head_dim], contiguous
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
    q_start,                      # int32: start token index for this batch (unused, but kept for clarity)
    num_q_tokens,                 # int32: number of tokens in this batch
    kv_start,                     # int32: start kv index for this batch (unused, but kept for clarity)
    num_kv_tokens,                # int32: number of kv tokens in this batch
    q_token_base,                 # int32: base offset for this token in q_ptr (already subtracted q_start)
    k_batch_base,                 # int32: base offset for k_ptr (already adjusted)
    v_batch_base,                 # int32: base offset for v_ptr (already adjusted)
):
    # One program per token
    t = tl.program_id(0)
    if t >= num_q_tokens:
        return

    # Process each query head
    for h in range(num_qo_heads):
        kv_head = h // gqa_ratio  # 0..7

        # Accumulators for LSE and output vector
        max_logits = tl.full((), -float("inf"), tl.float32)
        sum_exp = tl.zeros((), dtype=tl.float32)
        out_vec = tl.zeros([head_dim], dtype=tl.float32)

        # Loop over kv tokens to compute logsumexp
        for i in range(num_kv_tokens):
            # Load q_vec[h] (128 elements)
            q_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                q_elem = tl.load(q_ptr + t * (num_qo_heads * head_dim) + h * head_dim + j)
                q_vec[j] = q_elem

            # Load k_vec (128 elements) from k_ptr + i * (num_kv_heads * head_dim) + kv_head * head_dim
            k_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                k_elem = tl.load(k_ptr + k_batch_base + i * (num_kv_heads * head_dim) + kv_head * head_dim + j)
                k_vec[j] = k_elem

            # Dot product
            dot = tl.zeros((), dtype=tl.float32)
            for j in range(head_dim):
                dot += q_vec[j] * k_vec[j]

            logits_scaled = dot * sm_scale
            max_logits = tl.maximum(max_logits, logits_scaled)
            sum_exp += tl.exp(logits_scaled - max_logits)

        # Second pass: compute softmax and accumulate output
        for i in range(num_kv_tokens):
            q_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                q_elem = tl.load(q_ptr + t * (num_qo_heads * head_dim) + h * head_dim + j)
                q_vec[j] = q_elem

            k_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                k_elem = tl.load(k_ptr + k_batch_base + i * (num_kv_heads * head_dim) + kv_head * head_dim + j)
                k_vec[j] = k_elem

            dot = tl.zeros((), dtype=tl.float32)
            for j in range(head_dim):
                dot += q_vec[j] * k_vec[j]

            logits_scaled = dot * sm_scale
            soft = tl.exp(logits_scaled - max_logits) / sum_exp  # scalar

            v_vec = tl.zeros([head_dim], dtype=tl.float32)
            for j in range(head_dim):
                v_elem = tl.load(v_ptr + v_batch_base + i * (num_kv_heads * head_dim) + kv_head * head_dim + j)
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
        # Ensure device is CUDA
        device = q.device
        assert device.type == "cuda", "ModelNew.forward requires CUDA tensors for Triton kernels"

        total_q, num_qo_heads, head_dim = q.shape
        # Cast q to float32 for stable math (compute path uses Triton; q is input-only here)
        q_f32 = q.to(torch.float32).contiguous()

        # Compute k_cache_flat, v_cache_flat: squeeze the 1 dimension to [num_pages, 8, 128] on CPU
        # We'll build per-batch k_batch and v_batch on GPU to avoid torch operations in compute.
        # Note: We don't use original k_cache/v_cache directly in Triton; we preselect per batch using kv_indices on host.
        k_cache_flat_cpu = k_cache.squeeze(1).to(torch.float32).cpu().contiguous()  # [num_pages, 8, 128]
        v_cache_flat_cpu = v_cache.squeeze(1).to(torch.float32).cpu().contiguous()  # [num_pages, 8, 128]

        len_indptr = qo_indptr.shape[0]
        output = torch.empty(
            (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        gqa_ratio = num_qo_heads // 8  # asserts num_kv_heads == 8

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if (q_end - q_start) <= 0 or (kv_end - kv_start) <= 0:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start
            kv_indices_batch = kv_indices[kv_start:kv_end].to(torch.int64).cpu()  # [num_kv_tokens], CPU

            # Build k_batch and v_batch on CUDA without torch ops in compute path
            # k_cache_flat_cpu: [P, 8, 128], kv_indices_batch: [N], where N=num_kv_tokens
            # We need to gather k_cache_flat_cpu[kv_indices_batch[i]] and v_cache_flat_cpu[kv_indices_batch[i]] into CUDA arrays.
            k_batch_cuda = torch.empty((num_kv_tokens, 8, 128), device=device, dtype=torch.float32)
            v_batch_cuda = torch.empty((num_kv_tokens, 8, 128), device=device, dtype=torch.float32)

            # Fill k_batch_cuda and v_batch_cuda: copy per index
            # This is data movement, not torch compute in the Triton kernel.
            for i in range(num_kv_tokens):
                pid = int(kv_indices_batch[i].item())
                k_batch_cuda[i] = k_cache_flat_cpu[pid].clone()
                v_batch_cuda[i] = v_cache_flat_cpu[pid].clone()

            # Launch Triton kernel: one program per token in this batch
            grid = (num_q_tokens,)
            # Base offsets: q array is [num_q_tokens, 32, 128] contiguous
            # We pass q_f32[q_start:q_end], so q_token_base = 0 already; we'll compute from pointer math inside.
            # For Triton, we can simply provide pointers. Triton will index via q_ptr + t * (32*128) + h*128 + j.
            # Therefore, we don't need q_start; we set q_token_base = 0 because we pass q_f32[q_start:q_end] already.
            attn_gqa_single_token_kernel[grid](
                q_f32[q_start:q_end], k_batch_cuda, v_batch_cuda,
                output, lse,
                num_qo_heads=32, num_kv_heads=8, head_dim=128, gqa_ratio=4, sm_scale=float(sm_scale),
                q_start=q_start, num_q_tokens=num_q_tokens,
                kv_start=kv_start, num_kv_tokens=num_kv_tokens,
                q_token_base=0,  # we pass q slice already; Triton uses pointer arithmetic internally
                k_batch_base=0,  # k_batch_cuda is contiguous [N,8,128]
                v_batch_base=0,  # v_batch_cuda is contiguous [N,8,128]
                num_warps=4, num_stages=2
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
