import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: process one batch b, compute output[total_q, 32, 128] and lse[total_q, 32] for that batch.
# It uses fixed tile BLOCK_T=128 for q tokens and masks to handle actual num_q_tokens and num_kv_tokens.
if TRITON_AVAILABLE:
    @triton.jit
    def _batch_compute_kernel(
        q_ptr, k_ptr, v_ptr,
        qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr,
        output_ptr, lse_ptr,
        sm_scale: tl.float32,
        total_q: tl.int32, num_qo_heads: tl.int32, head_dim: tl.int32,
        gqa_ratio: tl.int32,
        b: tl.int32,  # batch index in [0, len_indptr - 2]
        len_indptr: tl.int32,
        BLOCK_T: tl.constexpr,  # number of q tokens to process per batch (tile), set to 128
    ):
        # Load qo_indptr[b] and qo_indptr[b+1], and kv_indptr[b], kv_indptr[b+1]
        q_start = tl.load(qo_indptr_ptr + b)
        q_end = tl.load(qo_indptr_ptr + b + 1)
        kv_start = tl.load(kv_indptr_ptr + b)
        kv_end = tl.load(kv_indptr_ptr + b)

        num_q_tokens = q_end - q_start
        num_kv_tokens = kv_end - kv_start

        # If nothing to do, return (output/lse are already initialized)
        if (num_q_tokens <= 0) or (num_kv_tokens <= 0) or (b >= len_indptr - 1):
            return

        # Build idx_vec for this batch: kv_indices[kv_start:kv_end]
        # We'll form a vector idx_vec with length BLOCK_T, then mask by actual num_kv_tokens
        idx_vec = []
        i = 0
        while i < num_kv_tokens:
            idx_val = tl.load(kv_indices_ptr + kv_start + i)
            idx_vec.append(idx_val)
            i += 1
        # For indices beyond num_kv_tokens, fill with 0 (masked later)
        while len(idx_vec) < BLOCK_T:
            idx_vec.append(0)

        # Process tokens in tiles of size BLOCK_T
        q_idx_base = 0
        while q_idx_base < BLOCK_T:
            # Vector of q indices in this tile
            q_idx_vec = q_idx_base + tl.arange(0, BLOCK_T)
            # Active mask for q: q_idx < num_q_tokens
            q_active = q_idx_vec < num_q_tokens
            # Convert to global q indices
            global_q_idx_vec = q_start + q_idx_vec

            # For each head h
            h = 0
            while h < num_qo_heads:
                kv_head = h // gqa_ratio  # GQA mapping: 32 heads -> 8 kv_heads, ratio=4

                # q_vec[h] for each token in tile: [BLOCK_T, head_dim]
                # q layout: [total_q, num_qo_heads, head_dim] contiguous
                q_off_vec = global_q_idx_vec * (num_qo_heads * head_dim) + h * head_dim
                q_vec = tl.load(q_ptr + q_off_vec + tl.arange(0, head_dim),
                                mask=q_active, other=0.0)  # [BLOCK_T, head_dim], fp32
                q_vec = q_vec.to(tl.float32)  # ensure fp32

                # Initialize lse accumulators for this head
                running_max = tl.full((), -float("inf"), dtype=tl.float32)
                lse_sum = tl.full((), 0.0, dtype=tl.float32)

                # Loop over KV indices i in tile: i = 0..BLOCK_T-1
                i = 0
                while i < BLOCK_T:
                    # i may exceed num_kv_tokens; mask by i < num_kv_tokens
                    active_i = i < num_kv_tokens
                    idx = idx_vec[i]  # int32
                    # Compute linear offsets for K and V rows
                    # k_ptr layout: [num_pages, num_kv_heads, head_dim] flattened
                    k_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                    # Load k_row as vector [head_dim]; masked by active_i
                    k_row = tl.load(k_ptr + k_off + tl.arange(0, head_dim),
                                    mask=active_i, other=0.0)
                    k_row = k_row.to(tl.float32)

                    # Apply causal mask: only consider i where q_idx < i
                    causal_mask = q_idx_vec < i
                    q_vec_eff = tl.where(q_active & causal_mask, q_vec, 0.0)
                    # Dot product q_vec_eff @ k_row across j
                    dot_vals = tl.zeros((BLOCK_T,), dtype=tl.float32)
                    for d in range(0, head_dim, 16):
                        q_chunk = q_vec_eff[:, d : d + 16]
                        k_chunk = k_row[d : d + 16]
                        dot_vals += tl.sum(q_chunk * k_chunk, axis=1)

                    # Scale
                    logits_vec = dot_vals * sm_scale  # [BLOCK_T]

                    # Update logsumexp over i where active_i is True; causal_mask zeros others
                    new_max = tl.maximum(running_max, logits_vec)
                    sum_term = lse_sum * tl.exp(running_max - logits_vec) + tl.exp(new_max - logits_vec)
                    running_max = new_max
                    lse_sum = sum_term

                    i += 1

                lse_value = (running_max + tl.log(lse_sum)) * (1.0 / math.log(2.0))  # float32

                # Compute output vector: out_vec[h] = sum_i attn_i * v_row[i]
                out_vec = tl.zeros((head_dim,), dtype=tl.float32)
                i = 0
                while i < BLOCK_T:
                    active_i = i < num_kv_tokens
                    idx = idx_vec[i]
                    k_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                    k_row = tl.load(k_ptr + k_off + tl.arange(0, head_dim),
                                    mask=active_i, other=0.0)
                    k_row = k_row.to(tl.float32)
                    # Compute dot for each token in tile
                    dot_eff = tl.zeros((BLOCK_T,), dtype=tl.float32)
                    for d in range(0, head_dim, 16):
                        q_chunk = q_vec[:, d : d + 16]
                        k_chunk = k_row[d : d + 16]
                        dot_eff += tl.sum(q_chunk * k_chunk, axis=1)
                    logits_scalar = dot_eff * sm_scale  # [BLOCK_T]
                    attn = tl.exp(logits_scalar - running_max)  # softmax over i with causal
                    v_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                    v_row = tl.load(v_ptr + v_off + tl.arange(0, head_dim),
                                    mask=active_i, other=0.0)
                    v_row = v_row.to(tl.float32)
                    # Only sum contributions for i where q_idx < i; other positions contribute 0
                    out_vec += tl.sum(tl.where(q_active[:, None] & (q_idx_vec[None, :] < i), attn[None, :] * v_row[None, :], 0.0), axis=0)

                    i += 1

                # Store output: output[global_q_idx, h, :] in float32 (host will cast to bfloat16)
                # output_ptr layout is [total_q, num_qo_heads, head_dim] contiguous
                for j in range(0, BLOCK_T):
                    if q_idx_vec[j] < num_q_tokens:
                        gq_idx = q_start + q_idx_vec[j]
                        out_addr = output_ptr + (gq_idx * num_qo_heads + h) * head_dim
                        tl.store(out_addr + tl.arange(0, head_dim), out_vec.to(tl.float32))

                # Update lse[global_q_idx, h] += lse_value / ln(2) for each token in this tile
                for j in range(0, BLOCK_T):
                    if q_idx_vec[j] < num_q_tokens:
                        gq_idx = q_start + q_idx_vec[j]
                        lse_addr = lse_ptr + (gq_idx * num_qo_heads + h)
                        curr_lse = tl.load(lse_addr)
                        new_lse = curr_lse + lse_value
                        tl.store(lse_addr, new_lse)

                h += 1

            q_idx_base += BLOCK_T


def _run_triton(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    """
    q: [total_q, 32, 128], bfloat16
    k_cache: [num_pages, 1, 8, 128], bfloat16
    v_cache: [num_pages, 1, 8, 128], bfloat16
    qo_indptr, kv_indptr: int32
    kv_indices: int32
    sm_scale: float32 scalar
    Returns: (output, lse) where output is bfloat16 [total_q, 32, 128], lse is float32 [total_q, 32]
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA for Triton"
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, _, num_kv_heads, _ = k_cache.shape
    assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Shape assertions must hold."
    gqa_ratio = num_qo_heads // num_kv_heads  # 4

    # Make inputs contiguous and flatten k_cache_flat, v_cache_flat by squeezing time dim (1)
    q = q.contiguous()
    k_cache_flat = k_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]
    v_cache_flat = v_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]

    len_indptr = qo_indptr.shape[0]
    B = len_indptr - 1  # number of batches

    # Allocate output and lse buffers: float32 for computation; host will cast output to bfloat16
    output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

    # Launch one Triton program per batch
    grid = (B,)
    _batch_compute_kernel[grid](
        q, k_cache_flat, v_cache_flat,
        qo_indptr, kv_indptr, kv_indices,
        output, lse,
        sm_scale,
        total_q, num_qo_heads, head_dim,
        gqa_ratio,
        B, len_indptr,
        BLOCK_T=128,
        num_warps=4, num_stages=2
    )

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)
    return output, lse


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Triton-only computation: no torch ops in host
        if q.device.type != "cuda":
            q = q.cuda()
        if k_cache.device.type != "cuda":
            k_cache = k_cache.cuda()
        if v_cache.device.type != "cuda":
            v_cache = v_cache.cuda()
        if qo_indptr.device.type != "cuda":
            qo_indptr = qo_indptr.cuda()
        if kv_indptr.device.type != "cuda":
            kv_indptr = kv_indptr.cuda()
        if kv_indices.device.type != "cuda":
            kv_indices = kv_indices.cuda()

        output, lse = _run_triton(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)
        return output, lse


def run(*args):
    return ModelNew()(*args)
