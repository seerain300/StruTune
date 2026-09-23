import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel(
    q_ptr,       # *float32, shape [Q, 32, 128]
    k_ptr,       # *float32, shape [K, 32, 128] (expanded heads)
    v_ptr,       # *float32, shape [K, 32, 128] (expanded heads)
    out_ptr,     # *float32, shape [Q, 32, 128]
    lse_ptr,     # *float32, shape [Q, 32]
    sm_scale,    # float32
    Q, K,        # int32 (runtime Q and K for this segment)
    H: tl.constexpr,               # 32
    ln2_inv: tl.constexpr,         # 1.0 / log(2) as float
    head_dim: tl.constexpr,        # 128
    BLOCK_Q: tl.constexpr,         # e.g., 128
    BLOCK_K: tl.constexpr,         # e.g., 128
):
    # Pass 1: compute per-(i,h) lse as max_j logits[i,h,j] (masked)
    for i0 in tl.static_range(0, BLOCK_Q):
        i = i0
        i_valid = i < Q
        lse_vals = tl.full((H,), -float("inf"), tl.float32)
        for h in tl.static_range(0, H):
            for j0 in tl.static_range(0, BLOCK_K):
                j = j0
                j_valid = j < K
                # Mask: j < (i + 1 + delta), delta = K - Q (per segment)
                delta = K - Q
                mask_valid = (j < (i + 1 + delta)) & i_valid & j_valid
                # Compute dot product q[i,h,:] dot k[j,h,:]
                q_ptrs = q_ptr + i * (H * head_dim) + h * head_dim
                k_ptrs = k_ptr + j * (H * head_dim) + h * head_dim
                q_vec = tl.load(q_ptrs, mask=i_valid, other=0.0)  # [128]
                k_vec = tl.load(k_ptrs, mask=j_valid, other=0.0)  # [128]
                dot = tl.zeros((), tl.float32)
                for d in tl.static_range(0, head_dim):
                    dot += q_vec[d] * k_vec[d]
                logit = dot * sm_scale
                if not mask_valid:
                    logit = -float("inf")
                lse_vals[h] = tl.maximum(lse_vals[h], logit)

    # Pass 2: compute denom per (i,h) and second pass recomputes logit for softmax
    for i0 in tl.static_range(0, BLOCK_Q):
        i = i0
        i_valid = i < Q
        for h in tl.static_range(0, H):
            denom = tl.zeros((), tl.float32)
            for j0 in tl.static_range(0, BLOCK_K):
                j = j0
                j_valid = j < K
                delta = K - Q
                mask_valid = (j < (i + 1 + delta)) & i_valid & j_valid
                q_ptrs = q_ptr + i * (H * head_dim) + h * head_dim
                k_ptrs = k_ptr + j * (H * head_dim) + h * head_dim
                q_vec = tl.load(q_ptrs, mask=i_valid, other=0.0)
                k_vec = tl.load(k_ptrs, mask=j_valid, other=0.0)
                dot = tl.zeros((), tl.float32)
                for d in tl.static_range(0, head_dim):
                    dot += q_vec[d] * k_vec[d]
                logit = dot * sm_scale
                if not mask_valid:
                    logit = -float("inf")
                # Numerator scaled for log2
                num = tl.exp(logit - lse_vals[h]) * ln2_inv
                denom += num

            # Pass 3: compute softmax per j and accumulate output
            for j0 in tl.static_range(0, BLOCK_K):
                j = j0
                j_valid = j < K
                delta = K - Q
                mask_valid = (j < (i + 1 + delta)) & i_valid & j_valid
                q_ptrs = q_ptr + i * (H * head_dim) + h * head_dim
                k_ptrs = k_ptr + j * (H * head_dim) + h * head_dim
                q_vec = tl.load(q_ptrs, mask=i_valid, other=0.0)
                k_ptrs2 = k_ptr + j * (H * head_dim) + h * head_dim
                k_vec = tl.load(k_ptrs2, mask=j_valid, other=0.0)  # same as above
                dot = tl.zeros((), tl.float32)
                for d in tl.static_range(0, head_dim):
                    dot += q_vec[d] * k_vec[d]
                logit = dot * sm_scale
                if not mask_valid:
                    logit = -float("inf")
                soft = tl.exp(logit - lse_vals[h]) / denom
                v_ptrs = v_ptr + j * (H * head_dim) + h * head_dim
                v_vec = tl.load(v_ptrs, mask=j_valid, other=0.0)
                out_ptrs = out_ptr + i * (H * head_dim) + h * head_dim
                # Accumulate output[i,h,:] += soft * v[j,h,:]
                # We can add scalar soft times vector v_vec to output row
                # Use simple scalar multiply:
                out_val = soft * v_vec
                # Store into output row; out_ptr points to [i, h, :]
                # Write over 128 dims:
                # We don't have a vectorized store here; do elementwise:
                # Triton supports scalar store via pointer + offset, but we need to update 128 elements.
                # To keep code simple, we compute per d via a loop:
                for dd in tl.static_range(0, head_dim):
                    tl.store(out_ptrs + dd, out_val[dd], mask=i_valid)

        # Store lse per (i,h)
        for h in tl.static_range(0, H):
            tl.store(lse_ptr + i * H + h, lse_vals[h], mask=(i < Q))


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Shapes: q [Lq, 32, 128], k [Lk, 8, 128], v [Lk, 8, 128], indptr ints
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors for Triton."
        device = q.device

        # Compute segment start/end from indptr
        qo_start = 0
        qo_end = int(qo_indptr[1].item()) if qo_indptr.numel() > 1 else int(qo_indptr[0].item())
        kv_start = 0
        kv_end = int(kv_indptr[1].item()) if kv_indptr.numel() > 1 else int(kv_indptr[0].item())

        # Slice per segment
        q_batch = q[qo_start:qo_end]  # [Q, 32, 128]
        k_batch = k[kv_start:kv_end]  # [K, 8, 128]
        v_batch = v[kv_start:kv_end]  # [K, 8, 128]

        # Convert to float32 for computation
        q_f32 = q_batch.to(torch.float32).contiguous()
        k_f32 = k_batch.to(torch.float32).contiguous()
        v_f32 = v_batch.to(torch.float32).contiguous()

        # GQA: expand 8 -> 32 heads via repeat_interleave(4) along head dim
        # Note: repeat_interleave along dim=1
        k_expanded = k_f32.repeat_interleave(4, dim=1).contiguous()  # [K, 32, 128]
        v_expanded = v_f32.repeat_interleave(4, dim=1).contiguous()  # [K, 32, 128]

        Q = q_f32.shape[0]
        K = k_expanded.shape[0]
        H = 32
        head_dim = 128

        # Allocate outputs
        out = torch.empty((Q, H, head_dim), dtype=torch.float32, device=device)  # we'll store float32 in kernel
        lse = torch.empty((Q, H), dtype=torch.float32, device=device)

        # Launch Triton kernel for this segment
        BLOCK_Q = 128
        BLOCK_K = 128
        ln2_inv = 1.0 / math.log(2.0)

        segment_attention_kernel[(1,)](
            q_f32,
            k_expanded,
            v_expanded,
            out,
            lse,
            sm_scale,
            Q,
            K,
            H,
            ln2_inv,
            head_dim,
            BLOCK_Q,
            BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # Return output as bfloat16 and lse as float32 to match original behavior
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
