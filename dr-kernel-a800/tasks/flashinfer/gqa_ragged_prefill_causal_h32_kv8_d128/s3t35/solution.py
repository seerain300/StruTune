import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel(
    q_ptr,       # *float32, [Q, 32, 128]
    k_ptr,       # *float32, [K, 32, 128] (GQA expanded)
    v_ptr,       # *float32, [K, 32, 128] (GQA expanded)
    out_ptr,     # *float32, [Q, 32, 128]
    lse_ptr,     # *float32, [Q, 32]
    sm_scale,    # float32
    Q: tl.constexpr,        # number of queries in this segment
    K: tl.constexpr,        # number of keys in this segment
    delta: tl.constexpr,    # K - Q
    H: tl.constexpr,        # 32 (num_qo_heads)
    head_dim: tl.constexpr, # 128
    ln2: tl.constexpr,      # log(2)
    BLOCK_Q: tl.constexpr,  # e.g., 128
    BLOCK_K: tl.constexpr,  # e.g., 128
):
    # We iterate over all i in this segment using static range. Masks handle out-of-range.
    for i0 in tl.static_range(0, BLOCK_Q):
        i = i0
        i_valid = i < Q

        # For each head h
        for h in tl.static_range(0, H):
            # Compute logits[i, h, :] across all j and apply mask j < (i + 1 + delta)
            lse_val = -float("inf")
            # We'll first compute lse_val (max over j)
            for j0 in tl.static_range(0, BLOCK_K):
                j = j0
                j_valid = j < K
                mask_valid = j < (i + 1 + delta)
                q_row_ptrs = q_ptr + i * (H * head_dim) + h * head_dim
                q_row = tl.load(q_row_ptrs, mask=i_valid, other=0.0)  # [128]
                k_col_ptrs = k_ptr + j * (H * head_dim) + h * head_dim
                k_col = tl.load(k_col_ptrs, mask=j_valid, other=0.0)  # [128]
                dot_val = 0.0
                for d in tl.static_range(0, head_dim):
                    dot_val += q_row[d] * k_col[d]
                logits_ij = dot_val * sm_scale
                logits_ij = tl.where(mask_valid, logits_ij, -float("inf"))
                lse_val = tl.maximum(lse_val, logits_ij)

            # Now compute denom = sum(exp(logits - lse_val)) across j (masked)
            sum_exp = 0.0
            for j0 in tl.static_range(0, BLOCK_K):
                j = j0
                j_valid = j < K
                mask_valid = j < (i + 1 + delta)
                q_row_ptrs = q_ptr + i * (H * head_dim) + h * head_dim
                q_row = tl.load(q_row_ptrs, mask=i_valid, other=0.0)  # [128]
                k_col_ptrs = k_ptr + j * (H * head_dim) + h * head_dim
                k_col = tl.load(k_col_ptrs, mask=j_valid, other=0.0)  # [128]
                dot_val = 0.0
                for d in tl.static_range(0, head_dim):
                    dot_val += q_row[d] * k_col[d]
                logits_ij = dot_val * sm_scale
                logits_ij = tl.where(mask_valid, logits_ij, -float("inf"))
                sum_exp += tl.exp(logits_ij - lse_val)

            # Compute output[i, h, :] = sum_j softmax[i,h,j] * v[j,h,:]
            out_vec = tl.zeros((head_dim,), tl.float32)
            for j0 in tl.static_range(0, BLOCK_K):
                j = j0
                j_valid = j < K
                mask_valid = j < (i + 1 + delta)
                q_row_ptrs = q_ptr + i * (H * head_dim) + h * head_dim
                q_row = tl.load(q_row_ptrs, mask=i_valid, other=0.0)  # [128]
                k_col_ptrs = k_ptr + j * (H * head_dim) + h * head_dim
                k_col = tl.load(k_col_ptrs, mask=j_valid, other=0.0)  # [128]
                dot_val = 0.0
                for d in tl.static_range(0, head_dim):
                    dot_val += q_row[d] * k_col[d]
                logits_ij = dot_val * sm_scale
                logits_ij = tl.where(mask_valid, logits_ij, -float("inf"))
                exp_val = tl.exp(logits_ij - lse_val)
                softmax_j = exp_val / (sum_exp * ln2)  # original divides by ln(2)
                v_col_ptrs = v_ptr + j * (H * head_dim) + h * head_dim
                v_col = tl.load(v_col_ptrs, mask=j_valid, other=0.0)  # [128]
                out_vec += softmax_j * v_col

            # Store output
            out_ptrs = out_ptr + i * (H * head_dim) + h * head_dim
            tl.store(out_ptrs, out_vec, mask=i_valid)
            # Store lse
            lse_ptrs = lse_ptr + i * H + h
            tl.store(lse_ptrs, lse_val, mask=i_valid)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Inputs: q [total_q, 32, 128], k [total_kv, 8, 128], v [total_kv, 8, 128]
        device = q.device
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"

        # Compute segments
        Lq = qo_indptr.shape[0] - 1
        Lk = kv_indptr.shape[0] - 1
        assert Lq == Lk, "len_indptr of qo and kv must match"
        len_indptr = Lq

        # Output buffers
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Process each segment
        for b in range(len_indptr):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            Q = q_end - q_start
            K = kv_end - kv_start
            delta = K - Q

            # Slice tensors and convert to float32
            q_batch = q[q_start:q_end].contiguous().to(torch.float32)   # [Q, 32, 128]
            k_batch = k[kv_start:kv_end].contiguous().to(torch.float32) # [K, 8, 128]
            v_batch = v[kv_start:kv_end].contiguous().to(torch.float32) # [K, 8, 128]

            # GQA expansion: 8 -> 32 heads
            k_expanded = k_batch.repeat_interleave(4, dim=1)  # [K, 32, 128]
            v_expanded = v_batch.repeat_interleave(4, dim=1)  # [K, 32, 128]

            # Launch Triton kernel: one program processes this segment. Loops inside are static.
            BLOCK_Q = 128
            BLOCK_K = 128
            grid = (1,)
            segment_attention_kernel[grid](
                q_batch, k_expanded, v_expanded, output, lse,
                float(sm_scale),
                Q=Q, K=K, delta=delta, H=32, head_dim=128, ln2=math.log(2.0),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2
            )

        # Match original output dtypes: output in bfloat16, lse in float32
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
