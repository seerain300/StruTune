import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel(
    q_ptr,       # *float32, shape [Q, 32, 128]
    k_ptr,       # *float32, shape [K, 32, 128]
    v_ptr,       # *float32, shape [K, 32, 128]
    out_ptr,     # *float32, shape [Q, 32, 128]
    lse_ptr,     # *float32, shape [Q, 32]
    sm_scale,    # float32
    Q, K,        # int32
    delta,       # int32 = K - Q
    H: tl.constexpr,               # 32
    ln2,          # float32 = log(2)
    head_dim: tl.constexpr,        # 128
    BLOCK_Q: tl.constexpr,         # e.g., 64
    BLOCK_K: tl.constexpr,         # e.g., 64
):
    # We will loop over heads h in static_range, and tile over Q and K.
    # Note: We assume per-segment Q, K are passed via Q, K. We iterate i in tiles of BLOCK_Q.
    for h in tl.static_range(0, H):
        # Initialize per-(i,h) lse and denom for this head h
        lse_vals = tl.full((Q,), -float("inf"), tl.float32)
        denom_vals = tl.full((Q,), 0.0, tl.float32)

        # Process Q in tiles
        for q0 in tl.static_range(0, Q, BLOCK_Q):
            i_vec = q0 + tl.arange(0, BLOCK_Q)      # [BLOCK_Q]
            valid_i = i_vec < Q

            # Compute lse per i for this head h by scanning K in chunks
            for k0 in tl.static_range(0, K, BLOCK_K):
                j_vec = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                valid_j = j_vec < K

                # For masked j, mask[i,j] = 1 if j < (i + 1 + delta) else 0
                # We implement mask as 1 for valid_j, 0 otherwise, then adjust per j < i + 1 + delta
                # But we need to exclude invalid j (out-of-range) by setting to 0. We'll load mask and set -inf for invalid j after.
                # Prepare mask_sub: shape [BLOCK_Q, BLOCK_K], default 1
                mask_sub = tl.full((BLOCK_Q, BLOCK_K), 1, tl.int32)
                # Apply j < (i + 1 + delta) condition
                # Note: i_vec is [BLOCK_Q], j_vec is [BLOCK_K]
                # Compute broadcasted condition
                cond = (j_vec[None, :] < (i_vec[:, None] + 1 + delta))
                # mask_sub is int8, tl.full returns int32; convert and mask
                mask_sub = mask_sub.to(tl.int8)
                # Broadcast valid_j across rows and valid_i across cols
                # Convert to int8 where invalid j -> 0, invalid i will be ignored by stores later via valid_i
                # Triton requires int8 for mask pointer; keep it as int8
                # For invalid j (j >= K), set mask to 0 so we ignore
                # We will pass mask pointer from host with correct values, but to keep it simple, we precompute mask_sub here.
                # However, since we cannot compute tl.load of mask here (we need mask_ptr), we'll instead pass mask as int8 tensor in host code.
                # To keep kernel simple, we assume mask has been passed correctly from host. We'll skip computing mask here and rely on mask_ptr.
                # But since Triton doesn't let us load from mask_ptr in this abstract context, we will compute mask within kernel using the formula above.
                # Convert cond to int8
                mask_sub = cond.to(tl.int8)
                # Ensure invalid j get 0 (j >= K), invalid i get 0 (i >= Q) — invalid i won't be stored due to valid_i anyway.
                # We need to pass mask_ptr from host; to reflect that, we instead load mask from memory: we'll define mask_ptr usage in host launch.
                # Since we can't create mask_ptr here, we'll restructure: we will load mask in host and pass mask_ptr; however Triton kernel signature does not include mask_ptr in the given problem.
                # Therefore, we'll instead compute mask within kernel using i_vec and j_vec. Triton supports elementwise comparison and broadcasting.

                # Compute logits chunk: q_sub @ k_sub_chunk^T
                # Load q_sub: shape [BLOCK_Q, 128]
                q_sub_ptrs = q_ptr + i_vec[:, None] * (H * head_dim) + h * head_dim
                q_sub = tl.load(q_sub_ptrs, mask=valid_i[:, None], other=0.0)  # [BLOCK_Q, 128]
                # Load k_sub_chunk: shape [BLOCK_K, 128]
                k_sub_ptrs = k_ptr + j_vec[:, None] * (H * head_dim) + h * head_dim
                k_sub = tl.load(k_sub_ptrs, mask=valid_j[:, None], other=0.0)  # [BLOCK_K, 128]

                # logits_sub_chunk: [BLOCK_Q, BLOCK_K] = sum_d q_sub[i,d] * k_sub[j,d]
                # We'll implement this by reducing over head_dim=128
                # Initialize
                logits_sub = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
                # Unroll over head_dim (constexpr 128)
                for d in tl.static_range(0, head_dim):
                    q_col = q_sub[:, d]  # [BLOCK_Q]
                    k_col = k_sub[:, d]  # [BLOCK_K]
                    logits_sub += q_col[:, None] * k_col[None, :]

                logits_sub = logits_sub * sm_scale

                # Apply mask: set -inf where mask_sub == 0. Since we created mask_sub as 1s and modified cond, we can directly use it.
                # Triton requires mask as tl.int8; convert cond to int8 and use as mask
                # We set logits_sub where mask_sub == 0 to -inf
                # Triton doesn't have tl.where(cond, x, y) but we can emulate: multiply by mask and keep others as is. Simpler: set -inf by where logic.
                # To set -inf, we can use mask_sub as boolean: convert to int1-like via tl.where, but Triton uses int8. We'll cast to bool via comparison.
                # However, Triton doesn't allow direct where with int8; we'll subtract a large value where invalid. Simpler: we keep mask_sub as 1 for valid and 0 for invalid, and set logits_sub where mask_sub==0 to -inf by multiplying with mask_sub (0 will zero out logits; but we need -inf). Therefore, we will explicitly set via masked load/store? Not possible. Instead, we will set logits_sub = tl.where(mask_sub == 1, logits_sub, -float("inf")) using tl.where with int1.
                # But tl.where expects int1. We can cast int8 to int1 via mask_sub != 0. Let's do that.
                mask_int1 = (mask_sub != 0)  # [BLOCK_Q, BLOCK_K], int1
                logits_sub = tl.where(mask_int1, logits_sub, -float("inf"))

                # Update per-(i,h) lse
                # max over this chunk: need to ignore invalid j by setting -inf for invalid j. mask_sub already did that.
                # Compute per i max
                lse_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
                for kk in tl.static_range(0, BLOCK_K):
                    # For invalid j, logits_sub[:,kk] was set to -inf, so it won't affect max
                    lse_i = tl.maximum(lse_i, logits_sub[:, kk])
                lse_vals = tl.maximum(lse_vals, lse_i)

                # Compute denom for this tile: sum_j exp(logits - lse_i)
                # We need to ignore invalid j as well. For invalid j, logits_sub = -inf -> exp(-inf) = 0. Good.
                for kk in tl.static_range(0, BLOCK_K):
                    # For each i in this tile, accumulate
                    # denom_vals = denom_vals + exp(logits_sub[i, kk] - lse_vals[i])
                    # Note: We need lse per i, not per tile. So use lse_i here? Actually, lse is per-(i,h) over all K. We need the max across entire K per i.
                    # However, we cannot access lse over entire K here; we compute a local lse_i and update denom for the positions in this chunk.
                    # That is acceptable: denom is computed per (i,h) across all K, but we update only with this chunk's max and sum. This gives us the correct denom.
                    # For now, we compute denom for this chunk using the maximum computed above (lse_i), but we need the true max across all K per i. Since we can't do that here without scanning all K, we will do a second scan to compute denom.
                    # To get true denom, we need the max over all K. We will recompute denom after we have lse_vals.
                    pass
                # We need to compute denom after finishing scanning all K. So we'll store logits_sub and lse_vals; then do a second pass to compute denom. This requires more memory. To keep things simple and correct, we'll restructure: compute logits_sub and then update lse_vals, and then a second pass to compute denom.

        # Now compute denom across all K for each i in this tile using lse_vals (per-(i,h) max)
        for q0 in tl.static_range(0, Q, BLOCK_Q):
            i_vec = q0 + tl.arange(0, BLOCK_Q)
            valid_i = i_vec < Q
            # denom_vals already computed? Not yet. We need to compute it now. But we can't. So we need to compute denom during the first pass. Let's integrate it.
            # We'll keep a running sum over all K chunks: accumulate denom for each (i,h) as we scan K.
            # To do that, we need to know lse per (i,h) after scanning all K. Triton doesn't support returning intermediate values across chunks like that. Therefore, we will:
            # - In the first loop over K chunks, compute lse_i for this chunk, update lse_vals (per i), and also compute the sum exp(logits - lse_vals[i]) for each j in this chunk, scaled by whether j is valid. However, without knowing the true lse across all K, computing denom here is incorrect.
            # Conclusion: We need two-phase: first pass compute lse_vals across all K, second pass compute denom using lse_vals, third pass compute output. But Triton does not allow returning lse_vals to a second pass easily. Therefore, we will keep lse_vals updated during first pass and then compute denom in a dedicated pass using those lse_vals.

        # Implement second pass: compute denom per (i,h) using lse_vals
        # But we cannot re-run loads; we need to keep logits_sub; instead, we can recompute logits_sub per (i,h) and update denom. However, recomputing q@k per chunk would be costly. Better: maintain a vector of logits for each i over K? Triton doesn't allow dynamic arrays. So we will not do this; we will instead restructure: for each K chunk, we store the chunk lse_i and chunk denom_i, and merge into lse_vals and denom_vals. This can be done because lse is a max, denom is a sum. For each chunk:
        #  - new_lse = max(lse, lse_i_chunk)
        #  - denom_i = sum(exp(logits_chunk - new_lse)) for valid j
        #  - denom = denom + denom_i * exp(new_lse - old_lse) / ln(2) (adjust for change in scale). But changing base requires log(2). To keep it simple, we will compute denom in two passes, but Triton does not support returning intermediate results; thus, we need to recompute.

        # Given the complexity, we will simplify: we will compute per-(i,h) lse in a first loop over K, then compute denom in a second loop over K using lse_vals, then compute output in a third loop. Triton allows multiple loops; we cannot return values, but we can update scalar accumulators.

        # Therefore, we will: in first scan, compute lse_vals across all K. Store lse_vals. In second scan, compute denom_vals using lse_vals. In third scan, compute output. For simplicity, we will do this by restructuring the kernel loops.

        # Re-implement: First loop: compute lse_vals across all K chunks
        # Initialize lse_vals
        lse_vals = tl.full((Q,), -float("inf"), tl.float32)
        # Pass 1: compute lse_vals
        for q0 in tl.static_range(0, Q, BLOCK_Q):
            i_vec = q0 + tl.arange(0, BLOCK_Q)
            valid_i = i_vec < Q
            for k0 in tl.static_range(0, K, BLOCK_K):
                j_vec = k0 + tl.arange(0, BLOCK_K)
                valid_j = j_vec < K
                mask_sub = tl.full((BLOCK_Q, BLOCK_K), 1, tl.int8)
                cond = (j_vec[None, :] < (i_vec[:, None] + 1 + delta))
                mask_sub = cond.to(tl.int8)
                # Load q_sub and k_sub
                q_sub_ptrs = q_ptr + i_vec[:, None] * (H * head_dim) + h * head_dim
                q_sub = tl.load(q_sub_ptrs, mask=valid_i[:, None], other=0.0)  # [BLOCK_Q, 128]
                k_sub_ptrs = k_ptr + j_vec[:, None] * (H * head_dim) + h * head_dim
                k_sub = tl.load(k_sub_ptrs, mask=valid_j[:, None], other=0.0)  # [BLOCK_K, 128]
                # Compute logits_sub
                logits_sub = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
                for d in tl.static_range(0, head_dim):
                    q_col = q_sub[:, d]
                    k_col = k_sub[:, d]
                    logits_sub += q_col[:, None] * k_col[None, :]
                logits_sub = logits_sub * sm_scale
                # Apply mask: set -inf where mask_sub == 0
                mask_int1 = (mask_sub != 0)
                logits_sub = tl.where(mask_int1, logits_sub, -float("inf"))
                # Compute per i max
                lse_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
                for kk in tl.static_range(0, BLOCK_K):
                    lse_i = tl.maximum(lse_i, logits_sub[:, kk])
                lse_vals = tl.maximum(lse_vals, lse_i)

        # Pass 2: compute denom_vals using lse_vals
        denom_vals = tl.full((Q,), 0.0, tl.float32)
        for q0 in tl.static_range(0, Q, BLOCK_Q):
            i_vec = q0 + tl.arange(0, BLOCK_Q)
            valid_i = i_vec < Q
            for k0 in tl.static_range(0, K, BLOCK_K):
                j_vec = k0 + tl.arange(0, BLOCK_K)
                valid_j = j_vec < K
                mask_sub = tl.full((BLOCK_Q, BLOCK_K), 1, tl.int8)
                cond = (j_vec[None, :] < (i_vec[:, None] + 1 + delta))
                mask_sub = cond.to(tl.int8)
                q_sub_ptrs = q_ptr + i_vec[:, None] * (H * head_dim) + h * head_dim
                q_sub = tl.load(q_sub_ptrs, mask=valid_i[:, None], other=0.0)  # [BLOCK_Q, 128]
                k_sub_ptrs = k_ptr + j_vec[:, None] * (H * head_dim) + h * head_dim
                k_sub = tl.load(k_sub_ptrs, mask=valid_j[:, None], other=0.0)  # [BLOCK_K, 128]
                logits_sub = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
                for d in tl.static_range(0, head_dim):
                    q_col = q_sub[:, d]
                    k_col = k_sub[:, d]
                    logits_sub += q_col[:, None] * k_col[None, :]
                logits_sub = logits_sub * sm_scale
                mask_int1 = (mask_sub != 0)
                logits_sub = tl.where(mask_int1, logits_sub, -float("inf"))
                # Compute sum_j exp(logits - lse_vals[i])
                # Note: We need to map lse per i from lse_vals vector. Triton supports vector ops.
                # Broadcast lse_vals to [BLOCK_Q, 1] and subtract
                lse_vec = lse_vals[i_vec]  # [BLOCK_Q]
                lse_vec = lse_vec[:, None]  # [BLOCK_Q, 1]
                exp_sum = 0.0
                for kk in tl.static_range(0, BLOCK_K):
                    # Accumulate only for valid j
                    if valid_j[kk]:
                        exp_sum += tl.exp(logits_sub[:, kk] - lse_vec[:, 0])
                denom_vals += exp_sum

        # Convert denom to base-2: denom2 = denom / ln(2)
        denom_vals = denom_vals / ln2

        # Pass 3: compute output
        for q0 in tl.static_range(0, Q, BLOCK_Q):
            i_vec = q0 + tl.arange(0, BLOCK_Q)
            valid_i = i_vec < Q
            # For each i, compute output over all K chunks
            for k0 in tl.static_range(0, K, BLOCK_K):
                j_vec = k0 + tl.arange(0, BLOCK_K)
                valid_j = j_vec < K
                mask_sub = tl.full((BLOCK_Q, BLOCK_K), 1, tl.int8)
                cond = (j_vec[None, :] < (i_vec[:, None] + 1 + delta))
                mask_sub = cond.to(tl.int8)
                # Load q_sub and k_sub
                q_sub_ptrs = q_ptr + i_vec[:, None] * (H * head_dim) + h * head_dim
                q_sub = tl.load(q_sub_ptrs, mask=valid_i[:, None], other=0.0)  # [BLOCK_Q, 128]
                k_sub_ptrs = k_ptr + j_vec[:, None] * (H * head_dim) + h * head_dim
                k_sub = tl.load(k_sub_ptrs, mask=valid_j[:, None], other=0.0)  # [BLOCK_K, 128]
                # Compute logits_sub
                logits_sub = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
                for d in tl.static_range(0, head_dim):
                    q_col = q_sub[:, d]
                    k_col = k_sub[:, d]
                    logits_sub += q_col[:, None] * k_col[None, :]
                logits_sub = logits_sub * sm_scale
                mask_int1 = (mask_sub != 0)
                logits_sub = tl.where(mask_int1, logits_sub, -float("inf"))
                # Broadcast lse_vals and denom_vals for this i
                lse_vec = lse_vals[i_vec]  # [BLOCK_Q]
                lse_vec = lse_vec[:, None]  # [BLOCK_Q, 1]
                denom_vec = denom_vals[i_vec]  # [BLOCK_Q]
                denom_vec = denom_vec[:, None]  # [BLOCK_Q, 1]
                # Compute output: out[i, h, d] = sum_j softmax_j * v_sub[j, h, d]
                # Load v_sub chunk
                v_sub_ptrs = v_ptr + j_vec[:, None] * (H * head_dim) + h * head_dim
                v_sub = tl.load(v_sub_ptrs, mask=valid_j[:, None], other=0.0)  # [BLOCK_K, 128]
                # Initialize output chunk for this (i,h)
                out_sub = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)
                # For each d in head_dim, compute scalar sum over j of softmax_j * v_sub[:, d]
                for dd in tl.static_range(0, head_dim):
                    # Compute softmax per i
                    # softmax[i] = exp(logits_sub[i, :]) / denom[i]
                    softmax_vec = tl.zeros((BLOCK_Q,), dtype=tl.float32)
                    for kk in tl.static_range(0, BLOCK_K):
                        # Accumulate only for valid j
                        if valid_j[kk]:
                            softmax_vec += tl.exp(logits_sub[:, kk] - lse_vec[:, 0]) / denom_vec[:, 0]
                    # Now out_sub[:, dd] = sum_k softmax_vec[:, None] * v_sub[k, dd]
                    out_sub[:, dd] = tl.sum(softmax_vec[:, None] * v_sub[:, dd], axis=0)
                # Store out_sub into output tensor
                out_ptrs = out_ptr + i_vec[:, None] * (H * head_dim) + h * head_dim
                tl.store(out_ptrs, out_sub, mask=valid_i[:, None])

        # Finally, store lse for this head h
        # lse_ptr layout is [Q, H], so we store lse_vals to lse_ptr[i, h]
        for i in tl.static_range(0, Q):
            lse_ptr[i * H + h] = lse_vals[i]


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # q: [total_q, 32, 128], k: [total_kv, 8, 128], v: [total_kv, 8, 128]
        # qo_indptr: [len_indptr], kv_indptr: [len_indptr], int32
        total_q, H_q, head_dim = q.shape
        total_kv, H_k, _ = k.shape
        assert H_q == 32, "Expected 32 heads for q"
        assert H_k == 8, "Expected 8 heads for k"
        assert head_dim == 128, "Expected head_dim == 128"

        # Cast to float32 for compute
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Prepare output and lse
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        # Process segments
        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice q, k, v for this segment
            q_batch = q_f32[q_start:q_end]                  # [Q, 32, 128]
            k_batch = k_f32[kv_start:kv_end]               # [K, 8, 128]
            v_batch = v_f32[kv_start:kv_end]               # [K, 8, 128]

            Q = q_batch.shape[0]
            K = k_batch.shape[0]

            # Expand K/V heads by 4 (GQA) to 32
            # Note: repeat_interleave is fine here, Triton will handle the resulting tensors.
            k_expanded = k_batch.repeat_interleave(4, dim=1)  # [K, 32, 128]
            v_expanded = v_batch.repeat_interleave(4, dim=1)  # [K, 32, 128]

            # Precompute mask: mask[i, j] = 1 if j < (i + 1 + delta) else 0
            # Create indices
            i = torch.arange(Q, device=q.device)               # [Q]
            j = torch.arange(K, device=q.device)               # [K]
            delta = K - Q
            cond = j[None, :] < (i[:, None] + 1 + delta)       # [Q, K]
            mask = cond.to(torch.int8)                         # [Q, K] int8

            # Launch Triton kernel for this segment
            BLOCK_Q = 64
            BLOCK_K = 64
            ln2 = math.log(2.0)
            segment_attention_kernel[(1,)](
                q_batch, k_expanded, v_expanded,
                output, lse,
                float(sm_scale),
                Q, K, delta,
                H_q, ln2, head_dim,
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

        # Return output as bfloat16 and lse as float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
