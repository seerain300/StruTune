import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_attention_kernel(
    q_ptr,           # *float32, [M, G, D], here M == len(q_block)
    k_ptr,           # *float32, [N, GH, D]
    v_ptr,           # *float32, [N, GH, D]
    out_ptr,         # *float32, [M, G, D] (will cast to bfloat16 on host)
    lse_ptr,         # *float32, [M, G]
    qo_indptr_ptr,   # *int32, [q_start, q_end]
    kv_indptr_ptr,   # *int32, [kv_start, kv_end]
    G: tl.constexpr,      # num_qo_heads (32)
    GH: tl.constexpr,     # num_kv_heads (8)
    D: tl.constexpr,      # head_dim (128)
    SM_SCALE: tl.constexpr  # scaling factor (float32)
):
    # Load block indices
    q_start = tl.load(qo_indptr_ptr + 0)  # int32
    q_end = tl.load(qo_indptr_ptr + 1)    # int32
    kv_start = tl.load(kv_indptr_ptr + 0) # int32
    kv_end = tl.load(kv_indptr_ptr + 1)   # int32

    M = q_end - q_start                    # number of queries in this block
    N = kv_end - kv_start                  # number of key/value tokens in this block
    delta = N - M                          # extra KV tokens beyond Q tokens

    # Process each query index q_idx in the block
    for q_idx in range(0, M):
        # Row-wise output accumulator for this q_idx
        out_row = tl.zeros((G, D), dtype=tl.float32)
        # LSE accumulator for this q_idx
        lse_row = tl.full((G,), -float('inf'), tl.float32)

        # For each qo_head g, compute attention
        for g in range(0, G):
            # Pointer to q[q_idx, g, :]
            qg_ptr = q_ptr + q_idx * G * D + g * D
            q_vec_g = tl.zeros((D,), dtype=tl.float32)
            # Load q vector
            for d in range(0, D):
                q_vec_g[d] = tl.load(qg_ptr + d)

            # Accumulate logits over all KV tokens for this (q_idx, g)
            logits_g = tl.zeros((N,), dtype=tl.float32)

            # Loop over KV tokens
            for kv_n in range(0, N):
                # Expand kv across GH groups (GQA mapping)
                for gh in range(0, GH):
                    # Load k[kv_n, gh, :] and v[kv_n, gh, :]
                    k_row_ptr = k_ptr + kv_start * GH * D + gh * D + kv_n * D
                    v_row_ptr = v_ptr + kv_start * GH * D + gh * D + kv_n * D

                    k_vec_g = tl.zeros((D,), dtype=tl.float32)
                    v_vec_g = tl.zeros((D,), dtype=tl.float32)
                    # Load k and v vectors
                    for d in range(0, D):
                        k_vec_g[d] = tl.load(k_row_ptr + d)
                        v_vec_g[d] = tl.load(v_row_ptr + d)

                    # Score contribution: dot(q_vec_g, k_vec_g) * SM_SCALE
                    score = 0.0
                    for dd in range(0, D):
                        score += q_vec_g[dd] * k_vec_g[dd]
                    score *= SM_SCALE

                    # Apply causal mask: j < q_idx + 1 + delta
                    allowed = kv_n < (q_idx + 1 + delta)
                    logits_g[kv_n] = tl.where(allowed, score, -float('inf'))

            # Compute LSE (base-2) for this row
            max_score = tl.max(logits_g, axis=0)
            exp_scores = tl.exp(logits_g - max_score)
            sum_scores = tl.sum(exp_scores, axis=0)
            lse_row[g] = tl.log(sum_scores) / math.log(2.0)

            # Softmax and accumulate output
            exp_scores = tl.exp(logits_g - max_score)
            denom = tl.sum(exp_scores, axis=0)
            for kv_n in range(0, N):
                attn = tl.where(kv_n < (q_idx + 1 + delta), exp_scores[kv_n] / denom, 0.0)
                # Accumulate output: out_row[g, d] += attn * v[kv_n, gh, d]
                # We need to use the expanded kv group, which maps to 4 v vectors for qo_head g.
                # Since qo_indptr and kv_indptr slices are consistent with GQA, we can just
                # use the v_vec_g for the current (gh, kv_n) to accumulate contributions.
                # Note: GH loop above already computed k for all groups; here we use v_vec_g
                # aligned with the same (gh, kv_n).
                # We will use v_vec_g as the expanded contribution (GQA semantics).
                # For each d, out_row[g, d] += attn * v_vec_g[d]
                # out_row is [G, D]; store as flat vectors
                for d in range(0, D):
                    out_row[g, d] += attn * v_vec_g[d]

        # Store outputs for this q_idx
        for g in range(0, G):
            out_base = out_ptr + q_idx * G * D + g * D
            for d in range(0, D):
                tl.store(out_base + d, out_row[g, d])
            lse_base = lse_ptr + q_idx * G + g
            tl.store(lse_base, lse_row[g])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Fallback to PyTorch if Triton not available or not on CUDA
        if (not TRITON_AVAILABLE) or (q.device.type != 'cuda'):
            total_q, num_qo_heads, head_dim = q.shape
            total_kv, num_kv_heads, _ = k.shape
            len_indptr = qo_indptr.shape[0]
            assert num_qo_heads == 32
            assert num_kv_heads == 8
            assert head_dim == 128
            assert total_q == int(qo_indptr[-1].item())
            assert total_kv == int(kv_indptr[-1].item())
            output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
            lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)
            gqa_ratio = num_qo_heads // num_kv_heads
            q_f32 = q.to(torch.float32)
            k_f32 = k.to(torch.float32)
            v_f32 = v.to(torch.float32)
            for b in range(len_indptr - 1):
                q_start = int(qo_indptr[b].item())
                q_end = int(qo_indptr[b + 1].item())
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())
                q_batch = q_f32[q_start:q_end]   # [num_q_tokens, 32, 128]
                k_batch = k_f32[kv_start:kv_end] # [num_kv_tokens, 8, 128]
                v_batch = v_f32[kv_start:kv_end] # [num_kv_tokens, 8, 128]
                num_q_tokens = q_batch.shape[0]
                num_kv_tokens = k_batch.shape[0]
                k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1) # [num_kv_tokens, 32, 128]
                v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1) # [num_kv_tokens, 32, 128]
                logits = torch.einsum('qhd,khd->qhk', q_batch, k_expanded) * sm_scale  # [num_q_tokens, 32, num_kv_tokens]
                q_positions = torch.arange(num_q_tokens, device=q.device)
                kv_positions = torch.arange(num_kv_tokens, device=q.device)
                delta = num_kv_tokens - num_q_tokens
                causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
                logits = logits.masked_fill(~causal_mask[:, None, :], float('-inf'))
                lse[q_start:q_end] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
                attn_weights = torch.softmax(logits, dim=-1)
                output_batch = torch.einsum('qhk,khd->qhd', attn_weights, v_expanded)  # [num_q_tokens, 32, 128]
                output[q_start:q_end] = output_batch.to(torch.bfloat16)
            return output, lse

        # Triton path
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        total_kv, num_kv_heads, _ = k_f32.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each block
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            q_block = q_f32[q_start:q_end]   # [M, 32, 128]
            k_block = k_f32[kv_start:kv_end] # [N, 8, 128]
            v_block = v_f32[kv_start:kv_end] # [N, 8, 128]

            # Create device int32 pointers for block ranges
            qo_block = torch.tensor([q_start, q_end], dtype=torch.int32, device=device)
            kv_block = torch.tensor([kv_start, kv_end], dtype=torch.int32, device=device)

            # Launch Triton kernel per block
            _block_attention_kernel[(1,)](
                q_block, k_block, v_block,
                output, lse,
                qo_block, kv_block,
                G=32, GH=8, D=128, SM_SCALE=float(sm_scale)
            )

        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
