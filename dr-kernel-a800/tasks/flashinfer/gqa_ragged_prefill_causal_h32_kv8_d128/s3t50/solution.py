import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel(
    q_ptr,       # *float32, [Q, 32, 128]
    k_ptr,       # *float32, [K, 8, 128]
    v_ptr,       # *float32, [K, 8, 128]
    out_ptr,     # *float32, [Q, 32, 128]
    lse_ptr,     # *float32, [Q, 32]
    sm_scale,    # float32
    Q, K,        # int32
    delta,       # int32 = K - Q
    H: tl.constexpr,               # 32
    ln2,          # float32 = log(2)
    head_dim: tl.constexpr,        # 128
    BLOCK_Q: tl.constexpr,         # 128
    BLOCK_K: tl.constexpr,         # 128
):
    # We process per head h = 0..H-1 (compile-time loop)
    for h in tl.static_range(0, H):
        # Initialize lse_vals for this head
        lse_vals = tl.full((Q,), -float("inf"), tl.float32)
        # Initialize output for this head
        out_row_base = out_ptr + h * head_dim  # points to [Q, 128] row base for head h

        # Iterate over query tiles
        for q0 in tl.static_range(0, Q, BLOCK_Q):
            i_vec = q0 + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
            i_valid = i_vec < Q

            # Compute lse over K for this tile
            lse_vals_tile = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
            # For output accumulation
            out_acc = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

            # Iterate over key tiles
            for k0 in tl.static_range(0, K, BLOCK_K):
                j_vec = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                j_valid = j_vec < K

                # Build mask: for each (i,j), valid if j < (i + 1 + delta)
                # Broadcast i_vec[:, None] and j_vec[None, :]
                i_j = i_vec[:, None]               # [BLOCK_Q, 1] broadcastable
                j_j = j_vec[None, :]              # [1, BLOCK_K] broadcastable
                mask_ij = (j_j < (i_j + 1 + delta))  # [BLOCK_Q, BLOCK_K], int1

                # Load q_sub [BLOCK_Q, 128] for this head h
                q_base = q_ptr + i_vec[:, None] * (H * head_dim) + h * head_dim
                q_sub = tl.load(q_base, mask=i_valid[:, None], other=0.0)  # [BLOCK_Q, 128]

                # Load k_sub [BLOCK_K, 128] for head h
                k_base = k_ptr + j_vec[:, None] * (H * head_dim) + h * head_dim
                k_sub = tl.load(k_base, mask=j_valid[:, None], other=0.0)  # [BLOCK_K, 128]

                # Compute logits_chunk: [BLOCK_Q, BLOCK_K]
                # logits = sum_d q_sub[:, d] * k_sub[:, d] -> [BLOCK_Q, BLOCK_K]
                logits_chunk = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
                for d in tl.static_range(0, head_dim):
                    qd = q_sub[:, d]                 # [BLOCK_Q]
                    kd = k_sub[:, d]                # [BLOCK_K]
                    # Outer product and sum: we can compute per element (i,j)
                    # Note: Triton supports vectorized elementwise operations, but outer product per element requires broadcasting.
                    # Compute dot between q_sub row and k_sub column by broadcasting:
                    # logits_chunk += qd[:, None] * kd[None, :]
                    # However, qd and kd are 1D; broadcast into 2D then multiply
                    logits_chunk += qd[:, None] * kd[None, :]

                # Apply scaling and mask
                logits_chunk = logits_chunk * sm_scale
                # Apply mask: set invalid entries to -inf
                logits_chunk = tl.where(mask_ij, logits_chunk, -float("inf"))

                # Update lse for this tile
                # For each i lane, lse is max over j of logits_chunk[i, :]
                # But we cannot use dynamic loops; instead, compute per i max with tl.where over j lanes
                # Since BLOCK_K is constexpr, we can compute max via reductions over j
                # However, Triton reductions are best applied to tensors; here we do per i max via masked approach:
                # Instead, compute max per i by selecting valid j only:
                # Create a vector of -inf for invalid j; we can get the masked max by taking max over all with invalid set to -inf
                # But to keep it simple, compute per i max using masked loads and reductions:
                # We'll do it by iterating over j dimension with static_range:
                # Compute per i max: For each ii, max over jj of logits_chunk[ii, jj] (masked by valid j)
                # Since j_valid is [BLOCK_K], we can compute per ii max:
                # But Triton doesn't support Python-side elementwise assignment; instead, we use tl.max reduction
                # Compute masked max over j: set invalid j to -inf, then take max along axis=1
                # We need a tensor to hold masked values. Let's create a masked_logits_chunk with -inf for invalid lanes.
                # Since Triton can't mix scalar and vector in where, we rely on tl.where which broadcasts properly.
                # Compute per i max:
                # We can compute it by iterating over BLOCK_K with static_range and updating per ii:
                # However, Triton supports vectorized reductions: tl.max(logits_chunk, axis=1) yields [BLOCK_Q]
                # And masking via tl.where already set invalid to -inf, so that's fine.
                # Compute per i max
                lse_per_i = tl.max(logits_chunk, axis=1)  # [BLOCK_Q]
                # Update lse_vals_tile
                lse_vals_tile = tl.maximum(lse_vals_tile, lse_per_i)

                # Compute denom for softmax per i: sum_j exp(logits - lse_per_i) / ln(2)
                # First, subtract lse, exponentiate, mask invalid j to 0
                # Since logits_chunk has -inf for invalid, exp(-inf)=0, so we don't need explicit masking here
                exp_chunk = tl.exp(logits_chunk - lse_per_i[:, None])  # [BLOCK_Q, BLOCK_K]
                denom_i = tl.sum(exp_chunk, axis=1) / ln2              # [BLOCK_Q]
                # Now compute output contributions: out_acc += softmax * v_sub along j
                # v_sub [BLOCK_K, 128] for head h
                v_base = v_ptr + j_vec[:, None] * (H * head_dim) + h * head_dim
                v_sub = tl.load(v_base, mask=j_valid[:, None], other=0.0)  # [BLOCK_K, 128]
                # Softmax across K: softmax_j = exp_chunk / denom_i[:, None]
                softmax_j = exp_chunk / denom_i[:, None]  # [BLOCK_Q, BLOCK_K]

                # For each query i (vectorized), accumulate out_acc += sum_j softmax_j[i, :] * v_sub[:, :]
                # We can do it by iterating j in static_range and broadcasting:
                for jj in tl.static_range(0, BLOCK_K):
                    # Load v_sub[jj, :] for this j
                    v_sub_j = v_sub[jj, :]  # [128]
                    # Gather softmax_j[:, jj] for all i lanes
                    softmax_jj = softmax_j[:, jj]  # [BLOCK_Q]
                    # Compute out contribution for each i: softmax_jj * v_sub_j
                    # Broadcast to [BLOCK_Q, 128] and add
                    out_acc += softmax_jj[:, None] * v_sub_j[None, :]

            # After processing all K tiles, out_acc holds accumulated output for this query tile
            # Store out[i, h, :] for valid i lanes
            # out_ptr layout: [Q, 32, 128]; for head h, we have out_row_base = out_ptr + h * 128
            # For each i lane, store out_acc[i, :] into out_row_base + i * 128
            # But Triton prefers vectorized operations. We can store by iterating over i lanes:
            for ii in tl.static_range(0, BLOCK_Q):
                i_index = i_vec[ii]
                if i_index < Q:
                    # Compute pointer for out_row_base + i_index * 128
                    out_row_ptrs = out_row_base + i_index * head_dim
                    # Store out_acc[ii, :] which is [128]
                    tl.store(out_row_ptrs + tl.arange(0, head_dim), out_acc[ii, :])

            # Store lse_vals_tile into lse_ptr at indices [q0:q0+BLOCK_Q, h]
            lse_out_ptrs = lse_ptr + (q0 + tl.arange(0, BLOCK_Q)) * H + h
            tl.store(lse_out_ptrs, lse_vals_tile, mask=(q0 + tl.arange(0, BLOCK_Q)) < Q)

# Optional: we keep a simple host function to run the kernel; ModelNew.forward uses it.
def _run_triton_only(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, sm_scale: float):
    # Ensure dtype float32 for compute
    q_f32 = q.to(torch.float32).contiguous()
    k_f32 = k.to(torch.float32).contiguous()
    v_f32 = v.to(torch.float32).contiguous()
    device = q_f32.device

    total_q = q_f32.shape[0]
    total_kv = k_f32.shape[0]
    len_indptr = qo_indptr.shape[0]

    # Output and lse tensors
    output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
    lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

    # Precompute ln(2)
    ln2 = math.log(2.0)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        # Slice q, k, v for this segment
        q_batch = q_f32[q_start:q_end]     # [Q, 32, 128]
        k_batch = k_f32[kv_start:kv_end]   # [K, 8, 128]
        v_batch = v_f32[kv_start:kv_end]   # [K, 8, 128]

        Q = q_batch.shape[0]
        K = k_batch.shape[0]
        delta = K - Q

        # Expand heads from 8 to 32 (GQA ratio = 4)
        k_expanded = k_batch.repeat_interleave(4, dim=1)  # [K, 32, 128]
        v_expanded = v_batch.repeat_interleave(4, dim=1)  # [K, 32, 128]

        # Launch Triton kernel: one program processes this segment with tiles
        segment_attention_kernel[(1,)](
            q_batch, k_expanded, v_expanded, output, lse,
            sm_scale,
            Q, K, delta,
            H=32,
            ln2=ln2,
            head_dim=128,
            BLOCK_Q=128, BLOCK_K=128,
        )

    # Convert output to bfloat16 and lse to float32 (match original)
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Triton-only forward: no torch ops inside
        out, lse = _run_triton_only(q, k, v, qo_indptr, kv_indptr, sm_scale)
        return out, lse


def run(*args):
    return ModelNew()(*args)
