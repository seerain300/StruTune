import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    q_ptr,                 # *fp32, pointer to q[b, h] vector, length = HEAD_DIM
    K_ptr,                 # *fp32, pointer to K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, pointer to V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, single scalar lse for this (b, h)
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM) (here 1/sqrt(128))
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # One program instance per (b, h) where h = program_id(0)
    h = tl.program_id(0)

    # First pass: compute lse = logsumexp(logits_scaled) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # shape [HEAD_DIM]
        # Load k vector for token t
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # shape [HEAD_DIM]
        # Dot product
        logits_t = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits_t * SM_SCALE
        # numerically stable logsumexp update
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(scaled - running_max)

    lse_ln = tl.log(running_sum) + running_max  # natural logsumexp of scaled logits
    lse = lse_ln * LOG2_INVERSE                 # divide by ln(2) to match original
    # Store lse (single scalar)
    tl.store(LSE_ptr, lse)

    # Second pass: compute output = sum_j exp(scaled_j - lse) * V_j
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        logits_t = tl.sum(q_vec * k_vec, axis=0)                        # scalar
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse)  # softmax over tokens
        out_vec += attn * v_vec

    # Store output
    tl.store(OUT_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # q: [B, 32, 128], k_cache: [N, 1, 8, 128], v_cache: [N, 1, 8, 128]
        # kv_indptr: int32 [B+1], kv_indices: int32 [num_kv_indices]
        # sm_scale: float32
        assert q.shape[1] == 32 and q.shape[2] == 128
        assert k_cache.shape[3] == 128 and v_cache.shape[3] == 128
        assert kv_indptr.shape[0] == q.shape[0] + 1
        device = q.device
        B = q.shape[0]
        H = q.shape[1]
        D = q.shape[2]

        # Compute num_tokens per batch
        num_tokens = kv_indptr[1:] - kv_indptr[:-1]  # [B], int64
        # Prepare output and lse buffers (float32 for stability), per batch
        output_b = [torch.empty((H, D), dtype=torch.float32, device=device) for _ in range(B)]
        lse_b = [torch.empty((H,), dtype=torch.float32, device=device) for _ in range(B)]

        # For each batch, gather K_t and V_t for the corresponding kv_indices slice
        for b in range(B):
            n = int(num_tokens[b].item())
            # Extract kv_indices range for this batch
            # Since len_indptr[b+1] = total tokens up to and including b, the total tokens for batch b are kv_indptr[b+1] - kv_indptr[b].
            # kv_indices are global indices in [0, N). We gather them per batch by slicing: kv_indices[kv_indptr[b]:kv_indptr[b+1]].
            # But kv_indices is 1D, and the sum equals num_kv_indices; we can use torch.index_select to get the used indices for this batch.
            # However, to strictly follow the original logic, we need to get the indices used for this batch. Since kv_indptr[b+1] - kv_indptr[b] equals n,
            # we can compute the slice start as accumulated sum of previous batches. But the provided tests have len_indptr[B+1] and num_tokens per batch already.
            # Simpler: compute start = sum_{i=0}^{b-1} num_tokens[i], end = start + n. But we don't have per-batch slice here because len_indptr only gives global ranges for the entire sequence. For the benchmark, len_indptr has one entry per batch (size B+1), so the total tokens equals the sum, and kv_indices length equals that sum.
            # In any case, we can form K_t and V_t using q, k_cache, v_cache as follows:
            # Build K_t: k_cache[:, 0, :, :] but we need rows indexed by kv_indices. We need to slice k_cache per batch. Triton kernel expects contiguous [num_tokens, head_dim] per batch.
            # To keep Triton-only heavy math and avoid torch ops in forward beyond allocations, we pre-gather K_t and V_t on host using torch.gather based on kv_indptr and kv_indices.
            # However, since we cannot rely on implicit slicing in Triton, we instead use torch.gather here to create per-batch contiguous tensors.

            # We need to find which kv_indices correspond to this batch. The original data provides kv_indptr and kv_indices. The global indices of used tokens for batch b are:
            # start = sum_{i=0}^{b-1} num_tokens[i], end = start + n. But we don't have per-batch start here. The benchmark provides len_indptr length B+1 and num_tokens per batch, implying that kv_indices already segments tokens per batch correctly. For simplicity and correctness in this environment, we can gather per-batch by using the full kv_indices range [kv_indptr[b], kv_indptr[b+1>). This works because num_tokens[b] = kv_indptr[b+1] - kv_indptr[b], and kv_indices length equals that sum. Therefore, the slice [kv_indptr[b]:kv_indptr[b+1]] selects exactly n indices for batch b.

            # Now gather K_t and V_t:
            # K_t: [n, 8, 128] -> select kv_indices[kv_indptr[b]:kv_indptr[b+1]] per batch
            # v_cache shape is [N, 1, 8, 128] -> select same indices
            # We will use torch.gather to form K_t and V_t per batch. This is minimal and necessary to feed Triton with correct data.
            # Note: k_cache and v_cache are [N, 1, 8, 128]. We want [n, 8, 128], which we can form by indexing.
            # Create index tensors for batch b:
            # For k_cache: k_cache[kv_indices[kv_indptr[b]:kv_indptr[b+1]], 0, :, :]
            # For v_cache: v_cache[kv_indices[kv_indptr[b]:kv_indptr[b+1]], 0, :, :]
            # We need to construct a 2D index tensor of shape [n, 8] for k_cache and [n, 8] for v_cache. Since kv_indices is [N], we can build by iterating t and selecting the corresponding row. However, Triton cannot index per-t; so we create contiguous tensors using torch operations.

            # Build K_t: select rows from k_cache using kv_indices[b:b+1] slice
            # Construct indices for k_cache: we need to form idx_k of shape [n, 8, 128] but Triton expects 2D K_ptr. So we create K_t as [n, 8*128] by flattening heads. Instead, we can create per-head [n, 128] and pass H=8 heads separately. To keep it simple, we gather per head.

            # We'll do per-head K_t_h and V_t_h for each head h. But Triton kernel expects K_ptr to be a contiguous [n, D] for a given head. We can gather K_t_h and V_t_h separately.

            # Gather K per head h: k_cache[:, 0, h, :] -> shape [N, 128]. Select rows for batch b.
            # But we need rows indexed by kv_indices range. Since k_cache is [N, 1, 8, 128], we can gather by selecting idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]], which gives global N indices. Then k_cache[idx, 0, h, :] gives [n, 128].
            # We'll gather K_t_h as [n, 128] and V_t_h as [n, 128] for each head h. Triton kernel expects K_ptr and V_ptr to be [n, D] contiguous, which is fine.

            # Prepare K_t_h and V_t_h for each head h:
            # Compute start and end for this batch:
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())

            # Global indices used by batch b: indices = kv_indices[start:end]
            # Gather K and V per head h
            for h_idx in range(H):
                # kv_head mapping for GQA: kv_head = h_idx // (H//8) = h_idx // 4
                kv_h = h_idx // (H // 8)  # since H=32, 32//8=4
                # Select rows from k_cache and v_cache at indices[start:end] for head kv_h
                # Note: k_cache shape [N, 1, 8, 128], v_cache [N, 1, 8, 128]
                # We want [n, 128] per batch per head. So we gather k_cache[idx, 0, kv_h, :] and v_cache[idx, 0, kv_h, :].
                # idx = kv_indices[start:end] -> shape [n], int32
                # Create K_t_h contiguous [n, 128]
                # We need to form a tensor of indices for k_cache: k_cache[idx, 0, kv_h, :]. Using torch.gather to produce a contiguous tensor of shape [n, 128] per head.
                # torch.gather along dim=0 of k_cache (which is N) is done by selecting rows. Since k_cache is [N,1,8,128], we can directly index with idx tensor.
                # Note: Triton kernel will read these contiguous tensors. We can create them as fp32.
                # However, creating tensors per head inside forward loop is allowed; the heavy math stays in Triton.
                # Gather K_t_h and V_t_h
                # We need to select rows k_cache[idx, 0, kv_h, :], where idx is a slice of kv_indices. Triton kernel will read these as contiguous [n, 128].
                # For clarity, we can perform gather using torch indexing:
                K_t_h = k_cache[idx, 0, kv_h, :].to(torch.float32).contiguous()  # shape [n, 128]
                V_t_h = v_cache[idx, 0, kv_h, :].to(torch.float32).contiguous()  # shape [n, 128]

                # Launch Triton kernel for this (b, h)
                # Allocate output and lse for this (b, h)
                out_vec = output_b[b][h_idx]  # tensor of shape [128]
                lse_scalar = lse_b[b][h_idx]  # scalar tensor

                # Launch kernel with grid=(1,) and pass pointers
                softmax_and_attention_single_bh[
                    (1,)
                ](
                    q[b, h_idx].to(torch.float32).contiguous(),            # q_ptr: [128]
                    K_t_h,                                                    # K_ptr: [n, 128]
                    V_t_h,                                                    # V_ptr: [n, 128]
                    out_vec,                                                  # OUT_ptr: [128]
                    lse_scalar,                                               # LSE_ptr: scalar
                    NUM_TOKENS=n,
                    HEAD_DIM=D,
                    SM_SCALE=sm_scale,
                    LOG2_INVERSE=1.4426950408889634,                        # 1/ln(2)
                )

        # Cast output to bfloat16 to match original
        output = [out.to(torch.bfloat16) for out in output_b]
        # Return output and lse (lse as float32)
        lse = [lse for lse in lse_b]
        # Return as list to match original run() signature
        return [output], [lse]


def run(*args):
    return ModelNew()(*args)
