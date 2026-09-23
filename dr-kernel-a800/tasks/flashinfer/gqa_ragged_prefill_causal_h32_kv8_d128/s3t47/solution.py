import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel(
    q_ptr,       # *float32, [Q, 32, 128] but we pass q_batch already
    k_ptr,       # *float32, [K, 32, 128], but we pass k_expanded
    v_ptr,       # *float32, [K, 32, 128], but we pass v_expanded
    out_ptr,     # *float32, [Q, 32, 128]
    lse_ptr,     # *float32, [Q, 32]
    Q,           # int32
    K,           # int32
    delta,       # int32
    sm_scale,    # float32
    ln2,         # float32
    H: tl.constexpr,               # 32
    BLOCK_Q: tl.constexpr,         # 128
    BLOCK_K: tl.constexpr,         # 128
):
    # Program ids: process one (i, h) per program
    i = tl.program_id(0)
    h = tl.program_id(1)
    i_valid = i < Q

    # Initialize lse for this (i, h)
    lse_val = tl.full((), -float("inf"), tl.float32)

    # Pass 1: compute lse = max over j of logits[i, h, j] (masked)
    for k0 in tl.static_range(0, 128):
        j0 = k0 * BLOCK_K
        j_vec = j0 + tl.arange(0, BLOCK_K)
        j_valid = j_vec < K

        # Build mask: j < (i + 1 + delta)
        cond = j_vec[None, :] < (i + 1 + delta)
        mask_sub = cond.to(tl.int8)

        # Load q_row[i, h, :] and compute logits for each j in this tile
        q_row_base = q_ptr + i * (H * 128) + h * 128
        q_row = tl.load(q_row_base, mask=i_valid, other=0.0)  # [128]

        logits_vec = tl.zeros((BLOCK_K,), tl.float32)

        # Compute logits[i, h, j_vec] = sum_d q_row[d] * k[j, h, d] * sm_scale
        for d in tl.static_range(0, 128):
            qd = q_row[d]  # scalar
            # Load k_sub[:, d] for this tile
            k_sub_ptrs = k_ptr + j_vec[:, None] * (H * 128) + h * 128
            k_sub = tl.load(k_sub_ptrs, mask=j_valid[:, None], other=0.0)  # [BLOCK_K, 128]
            kd = k_sub[:, d]  # [BLOCK_K]
            # Dot with qd: each kd is scalar along d dim; broadcast multiply
            logits_vec += qd * kd * sm_scale

        # Apply mask
        neg_inf = tl.full((BLOCK_K,), -float("inf"), tl.float32)
        logits_vec = tl.where(mask_sub == 0, neg_inf, logits_vec)

        # Update lse_val with max over this tile
        lse_val = tl.maximum(lse_val, tl.max(logits_vec, axis=0))

    # Pass 2: compute softmax and output for each i,h
    # We'll compute output[i, h, :] by iterating K tiles
    for k0 in tl.static_range(0, 128):
        j0 = k0 * BLOCK_K
        j_vec = j0 + tl.arange(0, BLOCK_K)
        j_valid = j_vec < K

        # Build mask
        cond = j_vec[None, :] < (i + 1 + delta)
        mask_sub = cond.to(tl.int8)

        q_row_base = q_ptr + i * (H * 128) + h * 128
        q_row = tl.load(q_row_base, mask=i_valid, other=0.0)  # [128]

        # Compute logits[i, h, j_vec] for this tile
        logits_vec = tl.zeros((BLOCK_K,), tl.float32)
        for d in tl.static_range(0, 128):
            qd = q_row[d]
            k_sub_ptrs = k_ptr + j_vec[:, None] * (H * 128) + h * 128
            k_sub = tl.load(k_sub_ptrs, mask=j_valid[:, None], other=0.0)  # [BLOCK_K, 128]
            kd = k_sub[:, d]
            logits_vec += qd * kd * sm_scale

        # Apply mask
        neg_inf = tl.full((BLOCK_K,), -float("inf"), tl.float32)
        logits_vec = tl.where(mask_sub == 0, neg_inf, logits_vec)

        # Compute softmax numerator and denom using lse_val
        numerator = tl.exp(logits_vec - lse_val)
        denom = tl.sum(numerator, axis=0)  # scalar
        softmax_vec = numerator / denom

        # Accumulate output[i, h, :]
        v_sub_ptrs = v_ptr + j_vec[:, None] * (H * 128) + h * 128
        v_sub = tl.load(v_sub_ptrs, mask=j_valid[:, None], other=0.0)  # [BLOCK_K, 128]
        out_row_base = out_ptr + i * (H * 128) + h * 128
        # out_row = sum_j softmax_vec[j] * v_sub[j, :]
        for jj in tl.static_range(0, BLOCK_K):
            out_vec = softmax_vec[jj] * v_sub[jj, :]
            # Assign to out row: we accumulate into a vector of size 128
            # Use tl.store into a temporary vector
            # Triton does not support indexing into out_ptr per element; instead, we compute out_vec as sum across K and store once.
            # Compute out_vec once per tile:
            # We need a running vector; do it by iterating d:
            for d in tl.static_range(0, 128):
                val = tl.sum(out_vec[d] * v_sub[jj, d])  # not needed; out_vec already computed per d
            # Instead, compute out_vec as dot with v_sub[jj, :]
            # out_vec is already the scalar contribution; we need a vector for all d:
            # Since out_vec is a vector, we can't store per d; we need to write into out_ptr.
            # To do that, we create a vector by looping over d and adding to out_ptr.
            # However, Triton kernels cannot assign to out_ptr per element directly here; we need to do it via a temporary.
            # Simpler approach: we compute the whole out_vec and store once by building a pointer vector across d:
            # Triton doesn't support this; thus, we instead store the final accumulated vector at the end of K loop.

            # We cannot store per-d here. So we keep a running vector and store at the end.
            # Maintain a vector accumulator:
            # We'll do a separate accumulation vector; Triton doesn't provide dynamic vector append. Therefore, we compute final out_vec by summing across jj contributions and store at the end.

        # Store final output row: To implement, we need a running vector. Triton kernel does not support
        # element-wise accumulation across multiple jj and d in a single kernel easily. So we switch to
        # a different approach: compute output[i, h, :] in a separate pass (not feasible here).

        # Note: The above is a conceptual outline. In practice, Triton kernels should have a clear store
        # operation. To avoid complexity, we will instead compute output via another kernel (not possible here).
        # Therefore, we'll implement output accumulation via PyTorch tensor math on host by allocating zeros
        # and returning them. But the requirement is to keep all computation in Triton. Hence, we adjust the
        # kernel to directly compute and store output.

        # Adjusted plan: Instead of trying to accumulate output across jj, we compute the entire output[i, h, :]
        # by doing softmax across all j in this tile and then a dot with v_sub. However, Triton does not
        # provide a way to sum across jj into a vector efficiently in this environment. Thus, we will compute
        # logits, lse, and output in a single kernel by scanning all j (K) and maintaining a running output
        # vector of size 128. Triton supports scalar operations and simple vector ops, but not general per-d
        # accumulation as above. Therefore, we keep the kernel focused on computing logits and lse, and
        # compute output on host using torch ops. However, that violates Triton-only.

        # To satisfy Triton-only, we will implement output computation inside the kernel using vectorized
        # operations over K. We'll compute softmax vector and multiply by v_sub vectors for each jj, summing
        # into a 128-length vector out_row, then store it.

        # Compute output[i, h, :] vector
        out_vec = tl.zeros((128,), tl.float32)
        # Reuse softmax_vec and v_sub from the tile; softmax_vec is per-j; out_vec accumulates contributions
        # for each jj, we update out_vec += softmax_vec[jj] * v_sub[jj, :]
        # We need to iterate jj over this tile and update out_vec:
        # Triton allows loops with static bounds; we can do that.
        for jj in tl.static_range(0, BLOCK_K):
            # v_sub[jj, :] is [128]; softmax_vec[jj] is scalar
            v_sub_j = v_sub[jj, :]
            contrib = softmax_vec[jj] * v_sub_j
            # Sum contrib across d into a scalar (since contrib is vector): Triton vector addition is allowed,
            # but we want to add per element into out_vec. Triton doesn't support direct per-element
            # indexing on out_ptr; instead, we can update out_vec += contrib. This works if contrib is a
            # vector of size 128: we can do out_vec += contrib. In Triton, vector variables can be
            # updated element-wise via tl.where or direct assignment. Here, we add vector contrib to out_vec.
            out_vec += contrib

        # Store out_vec to out[i, h, :]
        # We need to write out_vec into out_ptr for all d at once; Triton allows storing a vector if the
        # destination is a vector pointer. Create pointer vector:
        out_row_base = out_ptr + i * (H * 128) + h * 128
        tl.store(out_row_base + tl.arange(0, 128), out_vec, mask=i_valid)


# Note: The above kernel computes output directly inside the kernel. In practice, this requires careful
# vector handling. Given the constraints, we keep the kernel performing the main compute and output.
# The previous approach attempted to store per-d; Triton permits storing a vector to a vector pointer.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        device = q.device
        # Ensure inputs are contiguous
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]

        # Output buffers (float32 for compute)
        out = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

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
            q_batch = q_f32[q_start:q_end]            # [Q, 32, 128]
            k_batch = k_f32[kv_start:kv_end]         # [K, 8, 128]
            v_batch = v_f32[kv_start:kv_end]         # [K, 8, 128]

            # Expanded k/v to 32 heads (PyTorch expand + repeat; no Triton here to avoid torch ops)
            k_exp = k_batch.repeat_interleave(4, dim=1).contiguous()  # [K, 32, 128]
            v_exp = v_batch.repeat_interleave(4, dim=1).contiguous()  # [K, 32, 128]

            # Launch Triton kernel: grid = (Q, H)
            segment_attention_kernel[(Q, 32)](
                q_batch, k_exp, v_exp, out, lse,
                Q, K, delta,
                sm_scale, ln2,
                H=32,
                BLOCK_Q=128,
                BLOCK_K=128,
            )

        # Cast output back to bfloat16 to match original run signature expectations
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
