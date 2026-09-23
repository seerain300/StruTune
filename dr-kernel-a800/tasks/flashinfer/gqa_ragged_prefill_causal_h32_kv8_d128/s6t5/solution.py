import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_block_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    q_stride_b, q_stride_h, q_stride_d,
    k_stride_b, k_stride_h, k_stride_d,
    logits_stride_q, logits_stride_h, logits_stride_j,
    scale: tl.float32,
    BLOCK_J: tl.constexpr,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Load q[i, h] scalar (q is [num_q_tokens, 32, 128], contiguous)
    q_offset = i * q_stride_b + h * q_stride_h
    q_val = tl.load(q_ptr + q_offset)

    # Loop over kv positions in tiles
    for j0 in range(0, num_kv_tokens, BLOCK_J):
        j_vec = j0 + tl.arange(0, BLOCK_J)
        mask_j = j_vec < num_kv_tokens

        # Load k_expanded[j_vec, h] as vector
        k_offset_vec = j_vec * k_stride_b + h * k_stride_h
        k_vals = tl.load(k_ptr + k_offset_vec, mask=mask_j, other=0.0)

        # Compute scores
        scores = q_val * k_vals * scale

        # Apply causal mask: i can only attend to j < i + 1 + delta
        delta = num_kv_tokens - num_q_tokens
        causal_mask = j_vec < (i + 1 + delta)
        scores = tl.where(causal_mask & mask_j, scores, -float("inf"))

        # Store scores into logits[i, h, j_vec]
        logits_offsets = i * logits_stride_q + h * logits_stride_h + j_vec * logits_stride_j
        tl.store(logits_ptr + logits_offsets, scores, mask=mask_j)


@triton.jit
def _lse_reduce_block_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    lse_stride_b, lse_stride_h,
    scale: tl.float32,  # not used here but kept for signature symmetry
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Two-pass reduction over tiles: compute m (max) and s (sum exp shifted)
    m_global = -float("inf")
    s_global = 0.0

    for j0 in range(0, num_kv_tokens, BLOCK_J):
        j_vec = j0 + tl.arange(0, BLOCK_J)
        mask_j = j_vec < num_kv_tokens

        # Load scores for this tile
        offsets = i * lse_stride_b + h * lse_stride_h + j_vec * 1  # j_stride is 1 in contiguous layout
        scores = tl.load(logits_ptr + offsets, mask=mask_j, other=-float("inf"))

        # Pass 1: max in tile
        tile_max = -float("inf")
        for jj in range(0, BLOCK_J):
            score = scores[jj]
            tile_max = tl.maximum(tile_max, score)
        m_global = tl.maximum(m_global, tile_max)

        # Pass 2: sum exp(score - m_global) in tile
        tile_sum = 0.0
        for jj in range(0, BLOCK_J):
            score = scores[jj]
            y = tl.exp(score - m_global)
            tile_sum += y
        s_global += tile_sum

    # lse = logsumexp with max-shift: log(s_global) + m_global
    lse_val = tl.log(s_global) + m_global
    # Original code divides by log(2). Implement that here.
    inv_log2 = 1.0 / math.log(2.0)
    lse_val = lse_val * inv_log2

    lse_offset = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, lse_val)


@triton.jit
def _softmax_accum_block_kernel(
    logits_ptr, lse_ptr, v_expanded_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    out_stride_b, out_stride_h, out_stride_d,
    lse_stride_b, lse_stride_h,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Load lse[i, h]
    lse_offset = i * lse_stride_b + h * lse_stride_h
    m = tl.load(lse_ptr + lse_offset)

    # Output accumulator
    out_base = i * out_stride_b + h * out_stride_h
    d = tl.arange(0, 128)
    out_ptr_vec = out_ptr + out_base + d

    # Loop over kv tiles and accumulate
    for j0 in range(0, num_kv_tokens, BLOCK_J):
        j_vec = j0 + tl.arange(0, BLOCK_J)
        mask_j = j_vec < num_kv_tokens

        # Load scores for this tile
        offsets = i * lse_stride_b + h * lse_stride_h + j_vec  # contiguous layout along j
        scores = tl.load(logits_ptr + offsets, mask=mask_j, other=-float("inf"))

        # Compute y = exp(score - m)
        y = tl.exp(scores - m)
        # For masked j, y should be 0; we can multiply by mask to zero out, but Triton doesn't have tl.where on vectors; compute naturally via -inf => 0 after exp.
        # Load v_expanded[j_vec, h, :] as vector (shape [BLOCK_J, 128])
        v_stride_b = v_expanded_ptr.shape[1] * v_expanded_ptr.shape[2]  # 32 * 128 if contiguous across h and d
        # Note: Triton expects simple pointer arithmetic; since v_expanded is [N, 32, 128], we can treat strides as:
        v_stride_b = 128 * num_qo_heads  # each row (fixed j) is 32*128 elements; but we access j-th row's h-th slice directly
        v_stride_h = 128
        v_stride_d = 1

        v_base = j_vec * v_stride_b + h * v_stride_h
        v_ptr_mat = v_expanded_ptr + v_base[:, None] + d[None, :]  # [BLOCK_J, 128]

        # y[:, None] * v_ptr_mat -> [BLOCK_J, 128]
        contrib = y[:, None] * tl.load(v_ptr_mat)

        # Reduce across j to accumulate into out[i, h, :]
        # Sum along axis=0 to get [128] vector
        contrib_sum = tl.sum(contrib, axis=0)

        # Accumulate
        out_vals = tl.load(out_ptr_vec)
        out_vals = out_vals + contrib_sum
        tl.store(out_ptr_vec, out_vals)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block_j = 128  # tile size over kv dimension

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton requires CUDA tensors"

        # Constants from original code
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = num_qo_heads // num_kv_heads

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        device = q.device

        # Output and LSE (full tensors)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch segment (b-slice)
        for b in range(0, qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice tensors and make them contiguous
            q_slice = q[q_start:q_end].to(torch.float32).contiguous()          # [num_q_tokens, 32, 128]
            k_slice = k[kv_start:kv_end].to(torch.float32).contiguous()        # [num_kv_tokens, 8, 128]
            v_slice = v[kv_start:kv_end].to(torch.float32).contiguous()        # [num_kv_tokens, 8, 128]

            # Expand K/V to 32 heads (GQA)
            k_expanded = k_slice.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
            v_expanded = v_slice.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

            num_q_tokens = q_slice.shape[0]
            num_kv_tokens = k_expanded.shape[0]

            # Allocate logits buffer [num_q_tokens, 32, num_kv_tokens], float32, contiguous
            logits = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Strides for Triton (contiguous tensors)
            # q_slice: [num_q_tokens, 32, 128]
            q_stride_b = q_slice.shape[1] * head_dim  # 32 * 128 = 4096
            q_stride_h = head_dim                      # 128
            q_stride_d = 1

            # k_expanded: [num_kv_tokens, 32, 128]
            k_stride_b = k_expanded.shape[1] * head_dim  # 32 * 128 = 4096
            k_stride_h = head_dim                        # 128
            k_stride_d = 1

            # logits: [num_q_tokens, 32, num_kv_tokens], contiguous across j
            logits_stride_q = num_qo_heads * num_kv_tokens  # not directly used; we index by j
            logits_stride_h = num_kv_tokens
            logits_stride_j = 1

            # Launch Triton kernel to compute logits in tiles
            grid = (num_q_tokens * num_qo_heads,)
            _compute_logits_block_kernel[grid](
                q_slice, k_expanded, logits,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h, k_stride_d,
                logits_stride_q, logits_stride_h, logits_stride_j,
                sm_scale,
                BLOCK_J=self.block_j,
            )

            # Compute lse per (i, h) using Triton reduction in tiles
            lse_slice = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device)
            _lse_reduce_block_kernel[grid](
                logits, lse_slice,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                lse_slice.stride(0), lse_slice.stride(1),
                sm_scale,
                BLOCK_J=self.block_j,
            )

            # Accumulate output per (i, h) using Triton softmax*attention accumulation in tiles
            out_acc = torch.zeros((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)

            # Output strides (out_acc is [num_q_tokens, 32, 128], contiguous)
            out_stride_b = head_dim * num_qo_heads  # 128 * 32 = 4096
            out_stride_h = head_dim                 # 128
            out_stride_d = 1

            _softmax_accum_block_kernel[grid](
                logits, lse_slice, v_expanded, out_acc,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                out_stride_b, out_stride_h, out_stride_d,
                lse_slice.stride(0), lse_slice.stride(1),
                BLOCK_J=self.block_j,
            )

            # Store results into output for this b-slice (cast to bfloat16)
            output[q_start:q_end] = out_acc.to(torch.bfloat16)
            # Store lse for this b-slice (float32)
            lse[q_start:q_end] = lse_slice

        return output, lse


def run(*args):
    return ModelNew()(*args)
