import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel_full(
    q_ptr,        # *float32, [Q, 32, 128]
    k_ptr,        # *float32, [K, 32, 128]
    v_ptr,        # *float32, [K, 32, 128]
    output_ptr,   # *float32, [Q, 32, 128]
    lse_ptr,      # *float32, [Q, 32]
    sm_scale,     # float32
    Q, K,         # int32
    delta,        # int32 = K - Q
    H: tl.constexpr,               # 32
    head_dim: tl.constexpr,        # 128
    BLOCK_K: tl.constexpr,         # tile size over K, e.g., 128
):
    # For each query position i and head h, compute attention output and lse in tiles over K.
    for i in range(0, Q):
        # Initialize per-(i,h) accumulators
        for h in range(0, H):
            # Running max and sum_exp for lse
            max_val = -float("inf")
            sum_exp = 0.0

            # Process K in tiles
            t = 0
            while t * BLOCK_K < K:
                # First pass: compute max over this tile
                tile_max = -float("inf")
                for j in tl.static_range(0, BLOCK_K):
                    k_idx = t * BLOCK_K + j
                    # Mask: j < (i + 1 + delta)
                    if k_idx < (i + 1 + delta):
                        q_base = q_ptr + i * H * head_dim + h * head_dim
                        k_base = k_ptr + k_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                    else:
                        logits_ij = -float("inf")
                    tile_max = tl.maximum(tile_max, logits_ij)
                # Second pass: compute sum_exp for this tile
                tile_sum_exp = 0.0
                for j in tl.static_range(0, BLOCK_K):
                    k_idx = t * BLOCK_K + j
                    if k_idx < (i + 1 + delta):
                        q_base = q_ptr + i * H * head_dim + h * head_dim
                        k_base = k_ptr + k_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                    else:
                        logits_ij = -float("inf")
                    tile_sum_exp += tl.exp(logits_ij - tile_max)
                # Combine with global max and sum_exp
                # Since tile_max is the max in this tile, update max_val and sum_exp across tiles
                # We need to compute per-(i,h) logsumexp over all K; do this by recomputing per tile.
                # A more efficient approach would store all logits, but Triton doesn't support dynamic 2D storage across tiles easily.
                # Instead, we recompute soft for each tile and accumulate output. We'll compute soft_j per tile below.
                max_val = tl.maximum(max_val, tile_max)
                t += 1
            # After processing all tiles, compute final lse[i,h] = max_val + log(sum_exp)
            # Note: sum_exp above is accumulated per tile. We need total sum over all tiles:
            # We cannot carry sum_exp across tiles because tiles may not cover all j; instead, we recompute soft per tile and update output.
            # To compute final lse, we need to traverse tiles again to accumulate total sum_exp. We'll do that in the next loop when computing output.
            pass  # placeholder; actual sum_exp accumulation will be done when computing output in tiles.

            # Initialize output vector
            out_vec = [0.0] * head_dim

            # Third pass: compute softmax and accumulate output over tiles
            for t in tl.static_range(0, 1):  # we'll use while loop instead
                # This needs a while-like loop; Triton supports while. Replace placeholder.
                pass

            # Instead, we implement the output accumulation using while loop with tile processing:
            t = 0
            while t * BLOCK_K < K:
                # Compute tile max and sum_exp for this tile
                tile_max = -float("inf")
                for j in tl.static_range(0, BLOCK_K):
                    k_idx = t * BLOCK_K + j
                    if k_idx < (i + 1 + delta):
                        q_base = q_ptr + i * H * head_dim + h * head_dim
                        k_base = k_ptr + k_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                    else:
                        logits_ij = -float("inf")
                    tile_max = tl.maximum(tile_max, logits_ij)

                tile_sum_exp = 0.0
                for j in tl.static_range(0, BLOCK_K):
                    k_idx = t * BLOCK_K + j
                    if k_idx < (i + 1 + delta):
                        q_base = q_ptr + i * H * head_dim + h * head_dim
                        k_base = k_ptr + k_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                    else:
                        logits_ij = -float("inf")
                    tile_sum_exp += tl.exp(logits_ij - tile_max)

                # Now compute soft_j for each j in tile and accumulate output
                for j in tl.static_range(0, BLOCK_K):
                    k_idx = t * BLOCK_K + j
                    if k_idx < (i + 1 + delta):
                        q_base = q_ptr + i * H * head_dim + h * head_dim
                        k_base = k_ptr + k_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                    else:
                        logits_ij = -float("inf")
                    soft_j = tl.exp(logits_ij - tile_max) / tile_sum_exp
                    # v_expanded[k_idx, h, :] base = k_idx * (H * head_dim) + h * head_dim
                    v_base = v_ptr + k_idx * (H * head_dim) + h * head_dim
                    for d in tl.static_range(0, head_dim):
                        vd = tl.load(v_base + d)
                        out_vec[d] += soft_j * vd

                t += 1

            # Store output[i,h,:]
            out_base = output_ptr + i * H * head_dim + h * head_dim
            for d in tl.static_range(0, head_dim):
                tl.store(out_base + d, out_vec[d])

            # Store lse[i,h] = logsumexp across all K; we can compute it by traversing all tiles again.
            # But to avoid recomputation, we compute lse per tile and combine. Implement final lse:
            # We need sum of exp(logits - max) across all tiles. We'll compute it by traversing tiles once more and accumulating.
            total_sum_exp = 0.0
            t = 0
            while t * BLOCK_K < K:
                tile_max = -float("inf")
                for j in tl.static_range(0, BLOCK_K):
                    k_idx = t * BLOCK_K + j
                    if k_idx < (i + 1 + delta):
                        q_base = q_ptr + i * H * head_dim + h * head_dim
                        k_base = k_ptr + k_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                    else:
                        logits_ij = -float("inf")
                    tile_max = tl.maximum(tile_max, logits_ij)
                tile_sum_exp = 0.0
                for j in tl.static_range(0, BLOCK_K):
                    k_idx = t * BLOCK_K + j
                    if k_idx < (i + 1 + delta):
                        q_base = q_ptr + i * H * head_dim + h * head_dim
                        k_base = k_ptr + k_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                    else:
                        logits_ij = -float("inf")
                    tile_sum_exp += tl.exp(logits_ij - tile_max)
                total_sum_exp += tile_sum_exp
                t += 1
            max_val_total = -float("inf")
            for t in tl.static_range(0, 1):  # similar placeholder; actual max from tiles via while
                pass

            # Since we don't have per-element logits to find global max, we can approximate max as 0 if all masked; but that's incorrect.
            # A robust approach would be to store all logits; Triton doesn't support dynamic 2D storage. Therefore, we recompute max by scanning tiles again:
            max_val_total = -float("inf")
            t = 0
            while t * BLOCK_K < K:
                tile_max = -float("inf")
                for j in tl.static_range(0, BLOCK_K):
                    k_idx = t * BLOCK_K + j
                    if k_idx < (i + 1 + delta):
                        q_base = q_ptr + i * H * head_dim + h * head_dim
                        k_base = k_ptr + k_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                    else:
                        logits_ij = -float("inf")
                    tile_max = tl.maximum(tile_max, logits_ij)
                max_val_total = tl.maximum(max_val_total, tile_max)
                t += 1

            lse_entry = lse_ptr + i * H + h
            tl.store(lse_entry, max_val_total + tl.log(total_sum_exp))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors and cast to float32 for compute
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels require CUDA tensors"
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Shapes
        assert q_f32.shape[1:] == (32, 128), "q must have shape [*, 32, 128]"
        assert k_f32.shape[1:] == (8, 128), "k must have shape [*, 8, 128]"
        assert v_f32.shape[1:] == (8, 128), "v must have shape [*, 8, 128]"

        total_q = q_f32.shape[0]
        device = q_f32.device

        # Output buffers (float32 compute; we'll return bfloat16)
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        # lse buffer (float32), we will compute in kernel
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.numel()
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No tokens for this segment; skip
                continue

            q_batch = q_f32[q_start:q_end]      # [Q, 32, 128]
            k_batch = k_f32[kv_start:kv_end]    # [K, 8, 128]
            v_batch = v_f32[kv_start:kv_end]    # [K, 8, 128]

            Q = q_batch.shape[0]
            K = k_batch.shape[0]

            # Expand k and v along heads (GQA mapping: 8 -> 32)
            gqa_ratio = 4
            k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)   # [K, 32, 128]
            v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)   # [K, 32, 128]

            # delta for this segment
            delta = K - Q

            # Launch Triton kernel for this segment: one program per segment
            segment_attention_kernel_full[(1,)](
                q_batch, k_expanded, v_expanded,
                output[q_start:q_end], lse[q_start:q_end],
                float(sm_scale),
                Q, K, delta,
                H=32, head_dim=128, BLOCK_K=128,
                num_warps=4, num_stages=2
            )

        # Return output as bfloat16 (original behavior) and lse as float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
