import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel(
    q_ptr,       # *float32, shape [Q, 32, 128], but we pass q_batch already
    k_ptr,       # *float32, shape [K, 32, 128], but we pass k_batch already
    v_ptr,       # *float32, shape [K, 32, 128], but we pass v_batch already
    out_ptr,     # *float32, shape [Q, 32, 128]
    lse_ptr,     # *float32, shape [Q, 32]
    Q,           # int32, number of queries in this segment
    K,           # int32, number of keys/values in this segment
    delta,       # int32, K - Q
    sm_scale,    # float32
    ln2,         # float32, log(2)
    H: tl.constexpr,               # number of query heads, set to 32
    BLOCK_Q: tl.constexpr,         # tile size for Q dimension, e.g., 128
    BLOCK_K: tl.constexpr,         # tile size for K dimension, e.g., 128
):
    # We process one (i, h) pair per program. Grid is (Q, H).
    i = tl.program_id(0)
    h = tl.program_id(1)
    i_valid = i < Q

    # Initialize lse for this (i, h)
    lse_val = tl.full((), -float("inf"), tl.float32)

    # Pass 1: compute lse = logsumexp over K (masked) for this (i, h)
    for k0 in tl.static_range(0, 128):  # 128 is meta-bound; mask handles runtime K
        j0 = k0 * BLOCK_K
        j_vec = j0 + tl.arange(0, BLOCK_K)
        j_valid = j_vec < K

        # Build mask: j < (i + 1 + delta)
        # Note: i is runtime; we cast to int32 to avoid type issues
        cond = j_vec[None, :] < (i + 1 + delta)
        mask_sub = cond.to(tl.int8)

        # Load q[i, h, :] and compute logits for each j in this tile
        q_row_base = q_ptr + i * (H * 128) + h * 128
        q_row = tl.load(q_row_base, mask=i_valid, other=0.0)  # [128]
        logits_vec = tl.zeros((BLOCK_K,), tl.float32)

        # Compute logits[i, h, j_vec] = sum_d q[i, h, d] * k[j, h, d] * sm_scale
        for d in tl.static_range(0, 128):
            qd = q_row[d]  # scalar
            k_sub_ptrs = k_ptr + j_vec * (H * 128) + h * 128
            kd = tl.load(k_sub_ptrs, mask=j_valid, other=0.0)  # [BLOCK_K]
            logits_vec += qd * kd * sm_scale

        # Apply mask: invalid j -> -inf
        logits_vec = tl.where(j_valid & (mask_sub == 1), logits_vec, tl.full((BLOCK_K,), -float("inf"), tl.float32))

        # Update lse as logsumexp over j in this tile, with base-2 scaling in denom later
        # max over j
        cur_max = tl.max(logits_vec, axis=0)
        # sum exp(logits - cur_max)
        exp_sum = 0.0
        for jj in tl.static_range(0, BLOCK_K):
            vj = j_valid[jj]
            ljj = logits_vec[jj]
            # If j is invalid, exp(-inf) = 0
            exp_sum += tl.where(vj, tl.exp(ljj - cur_max), 0.0)
        # Compute denom and update lse
        # denom = exp_sum * ln(2); lse = log(denom) if denom > 0 else -inf
        denom = exp_sum * ln2
        lse_candidate = tl.log(denom) if denom > 0 else -float("inf")
        lse_val = tl.maximum(lse_val, lse_candidate)

    # Pass 2: compute output[i, h, :] = sum_j softmax(logits[i,h,j]) * v[j,h,:] for all j
    out_row_base = out_ptr + i * (H * 128) + h * 128
    out_vec = tl.zeros((128,), tl.float32)
    for k0 in tl.static_range(0, 128):
        j0 = k0 * BLOCK_K
        j_vec = j0 + tl.arange(0, BLOCK_K)
        j_valid = j_vec < K

        # Build mask: j < (i + 1 + delta)
        cond = j_vec[None, :] < (i + 1 + delta)
        mask_sub = cond.to(tl.int8)

        # Load q[i, h, :]
        q_row_base = q_ptr + i * (H * 128) + h * 128
        q_row = tl.load(q_row_base, mask=i_valid, other=0.0)  # [128]
        logits_vec = tl.zeros((BLOCK_K,), tl.float32)

        # Compute logits[i, h, j_vec]
        for d in tl.static_range(0, 128):
            qd = q_row[d]
            k_sub_ptrs = k_ptr + j_vec * (H * 128) + h * 128
            kd = tl.load(k_sub_ptrs, mask=j_valid, other=0.0)  # [BLOCK_K]
            logits_vec += qd * kd * sm_scale

        # Apply mask
        logits_vec = tl.where(j_valid & (mask_sub == 1), logits_vec, tl.full((BLOCK_K,), -float("inf"), tl.float32))

        # Compute softmax along j (BLOCK_K) for this i,h
        # softmax_j = exp(logits - lse_val) / sum exp(logits - lse_val)
        # But lse_val was computed as log(denom). We need proper softmax for numerical stability.
        # Instead, recompute using max trick within this tile:
        cur_max = tl.max(logits_vec, axis=0)
        numerators = tl.where(j_valid, tl.exp(logits_vec - cur_max), 0.0)
        sum_numerators = 0.0
        for jj in tl.static_range(0, BLOCK_K):
            sum_numerators += numerators[jj]
        sum_numerators = tl.where(sum_numerators > 0, sum_numerators, 1.0)  # avoid div by 0
        softmax_vec = tl.where(j_valid, numerators / sum_numerators, 0.0)

        # Accumulate output: out[i, h, :] += softmax_vec * v[j,h,:]
        # Load v for this tile and heads h
        v_sub_ptrs = v_ptr + j_vec * (H * 128) + h * 128
        v_sub = tl.load(v_sub_ptrs, mask=j_valid, other=0.0)  # [BLOCK_K, 128]
        # Multiply and reduce over j
        for jj in tl.static_range(0, BLOCK_K):
            # For each jj, v_sub[jj, :] and softmax_vec[jj]
            v_col = v_sub[jj, :]  # [128]
            out_vec += softmax_vec[jj] * v_col

    # Store output row
    # If i is invalid, output row stays zeros due to initialization
    # Store with mask if we want to be explicit, but out_vec has zeros already
    tl.store(out_row_base, out_vec, mask=i_valid)

    # Store lse[i, h]
    lse_addr = lse_ptr + i * H + h
    tl.store(lse_addr, lse_val, mask=i_valid)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Device and dtype setup
        device = q.device
        # Ensure inputs are contiguous and in float32 for Triton compute
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]

        # Output buffers
        out = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=device)

        # Precompute ln(2)
        ln2 = math.log(2.0)

        # Process segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            Q = q_end - q_start
            K = kv_end - kv_start
            delta = K - Q

            # Slice q, k, v for this segment
            q_batch = q_f32[q_start:q_end]  # [Q, 32, 128]
            k_batch = k_f32[kv_start:kv_end]  # [K, 8, 128]
            v_batch = v_f32[kv_start:kv_end]  # [K, 8, 128]

            # Expand heads (GQA: 8->32) using repeat_interleave on host
            # This is fine because it's per-segment and relatively small
            # Triton cannot index different head dims directly; we pass per-head segments via slicing in kernel using H constant.
            # For Triton kernel, we pass the already expanded forms. Since we don't compute expanded in-kernel (to keep it simple),
            # we will expand heads with repeat_interleave. But since Triton kernel doesn't have easy per-head loops over 8->32 mapping,
            # we recompute expanded on-the-fly in Python, or better: compute per h using slices. However, Triton expects fixed head count
            # so we set H=32 in the kernel and rely on k_batch/v_batch expanded on host before calling kernel.

            # Expand k/v to 32 heads: repeat_interleave along head dimension
            # k_batch: [K, 8, 128] -> [K, 32, 128]
            k_expanded = k_batch.repeat_interleave(4, dim=1).contiguous()
            v_expanded = v_batch.repeat_interleave(4, dim=1).contiguous()

            # Launch Triton kernel over grid (Q, H)
            grid = (Q, 32)
            segment_attention_kernel[grid](
                q_batch, k_expanded, v_expanded, out, lse,
                Q, K, delta,
                sm_scale, ln2,
                H=32,
                BLOCK_Q=128, BLOCK_K=128,
            )

        # Cast output and lse to match original behavior: output bfloat16, lse float32
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
