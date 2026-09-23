import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel(
    q_ptr,           # *float32, [Q, 32, 128]
    k_ptr,           # *float32, [K, 32, 128]
    v_ptr,           # *float32, [K, 32, 128]
    mask_ptr,        # *int8,    [Q, K], 0/1 mask (j < (i + 1 + delta))
    output_ptr,      # *float32, [Q, 32, 128]
    lse_ptr,         # *float32, [Q, 32]
    sm_scale,        # float32
    Q, K,            # int32
    delta,           # int32 = K - Q
    H: tl.constexpr,               # 32
    gqa_ratio: tl.constexpr,       # 4
    ln2,              # float32 = log(2)
    BLOCK_Q: tl.constexpr,         # tile size over Q, e.g., 64
    BLOCK_K: tl.constexpr,         # tile size over K, e.g., 64
    head_dim: tl.constexpr,        # 128
):
    # Tile over queries and keys; one program handles up to BLOCK_Q queries in this segment
    for i0 in tl.static_range(0, Q, BLOCK_Q):
        for j0 in tl.static_range(0, K, BLOCK_K):
            # Local tile counts
            Q_tile = Q - i0 if (Q - i0) <= BLOCK_Q else BLOCK_Q
            K_tile = K - j0 if (K - j0) <= BLOCK_K else BLOCK_K

            # Compute lse[i,h] and output[i,h,:] for each i in tile
            for i in tl.static_range(0, BLOCK_Q):
                i_global = i0 + i
                if i_global >= Q:
                    break
                # Initialize lse and denom for this (i_global, h)
                lse_val = -float("inf")
                sum_exp = 0.0
                out_vec = [0.0] * head_dim

                # First pass: compute lse and sum_exp across K tile
                for j in tl.static_range(0, BLOCK_K):
                    j_global = j0 + j
                    if j_global >= K:
                        break
                    # Load mask[i_global, j_global]
                    mask_ij = tl.load(mask_ptr + i_global * K + j_global)
                    # Compute dot product over head_dim
                    q_base = q_ptr + i_global * H * head_dim + 0 * head_dim  # h loop will recompute per h
                    k_base = k_ptr + j_global * H * head_dim + 0 * head_dim
                    dot = 0.0
                    for d in tl.static_range(0, head_dim):
                        qd = tl.load(q_base + d)
                        kd = tl.load(k_base + d)
                        dot += qd * kd
                    logits_ij = dot * sm_scale
                    # Apply mask and bounded attention rule
                    if (mask_ij != 0) and (j_global < (i_global + 1 + delta)):
                        # Update lse
                        lse_val = tl.maximum(lse_val, logits_ij)
                    # Accumulate sum_exp for denominator
                    if (mask_ij != 0) and (j_global < (i_global + 1 + delta)):
                        sum_exp += tl.exp(logits_ij - lse_val)
                    else:
                        sum_exp += tl.exp(-float("inf"))  # 0 contribution

                # If no valid j (sum_exp == 0), set denom to 0 to avoid div by 0; but sum_exp won't be 0 since lse_val was -inf
                denom_ih = sum_exp / ln2

                # Second pass: accumulate output[i,h,:] = sum_j exp(logits[i,h,j] - lse[i,h]) * v_expanded[j,h,:] / denom_ih
                for j in tl.static_range(0, BLOCK_K):
                    j_global = j0 + j
                    if j_global >= K:
                        break
                    mask_ij = tl.load(mask_ptr + i_global * K + j_global)
                    q_base = q_ptr + i_global * H * head_dim + 0 * head_dim
                    k_base = k_ptr + j_global * H * head_dim + 0 * head_dim
                    dot = 0.0
                    for d in tl.static_range(0, head_dim):
                        qd = tl.load(q_base + d)
                        kd = tl.load(k_base + d)
                        dot += qd * kd
                    logits_ij = dot * sm_scale
                    if (mask_ij != 0) and (j_global < (i_global + 1 + delta)):
                        numerator_j = tl.exp(logits_ij - lse_val)  # ln2 in denom_ih
                        # v_expanded[j,h,:] base = j * (H * head_dim) + h * head_dim
                        for h2 in tl.static_range(0, H):
                            v_base = v_ptr + j_global * (H * head_dim) + h2 * head_dim
                            for d in tl.static_range(0, head_dim):
                                vd = tl.load(v_base + d)
                                out_vec[d] += (numerator_j * vd) / denom_ih

                # Store output[i,h,:]
                for h2 in tl.static_range(0, H):
                    out_base = output_ptr + i_global * H * head_dim + h2 * head_dim
                    for d in tl.static_range(0, head_dim):
                        tl.store(out_base + d, out_vec[d])

                # Store lse[i,h] for all heads h
                for h2 in tl.static_range(0, H):
                    lse_entry = lse_ptr + i_global * H + h2
                    # We computed lse_val for all h by final update via denom; here we use lse_val computed above for h=0. To store per h, we must update lse_val for each h. Since we loop over h2, recompute lse_val for that h2 by considering only j in tile that are valid for h2. For simplicity, we rely on the fact that lse depends only on j and not on h, so we can store lse_val for h2==0. But we need lse per head; thus, we recompute lse and denom per h by re-deriving them (above we stored only h=0). To fix this, we'll recompute per h.

                # Note: The above code stores lse for h=0; to store per h, we must recompute. Let's fix: recompute lse and denom per h inside the h2 loop.
            # After i loop, move to next j tile
        # After j tiles, move to next i tile


# Redefine kernel to correctly compute and store lse per head h by looping h as well.

@triton.jit
def segment_attention_kernel_correct(
    q_ptr,           # *float32, [Q, 32, 128]
    k_ptr,           # *float32, [K, 32, 128]
    v_ptr,           # *float32, [K, 32, 128]
    mask_ptr,        # *int8,    [Q, K], 0/1 mask (j < (i + 1 + delta))
    output_ptr,      # *float32, [Q, 32, 128]
    lse_ptr,         # *float32, [Q, 32]
    sm_scale,        # float32
    Q, K,            # int32
    delta,           # int32 = K - Q
    H: tl.constexpr,               # 32
    gqa_ratio: tl.constexpr,       # 4
    ln2,              # float32 = log(2)
    BLOCK_Q: tl.constexpr,         # tile size over Q, e.g., 64
    BLOCK_K: tl.constexpr,         # tile size over K, e.g., 64
    head_dim: tl.constexpr,        # 128
):
    for i0 in tl.static_range(0, Q, BLOCK_Q):
        for j0 in tl.static_range(0, K, BLOCK_K):
            Q_tile = Q - i0 if (Q - i0) <= BLOCK_Q else BLOCK_Q
            K_tile = K - j0 if (K - j0) <= BLOCK_K else BLOCK_K

            # For each i in tile
            for i in tl.static_range(0, BLOCK_Q):
                i_global = i0 + i
                if i_global >= Q:
                    break

                # For each head h
                for h in tl.static_range(0, H):
                    # Compute lse_val and denom_ih for this (i_global, h)
                    lse_val = -float("inf")
                    sum_exp = 0.0

                    # First pass: lse and sum_exp
                    for j in tl.static_range(0, BLOCK_K):
                        j_global = j0 + j
                        if j_global >= K:
                            break
                        mask_ij = tl.load(mask_ptr + i_global * K + j_global)
                        q_base = q_ptr + i_global * H * head_dim + h * head_dim
                        k_base = k_ptr + j_global * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                        if (mask_ij != 0) and (j_global < (i_global + 1 + delta)):
                            lse_val = tl.maximum(lse_val, logits_ij)
                            sum_exp += tl.exp(logits_ij - lse_val)
                        else:
                            sum_exp += tl.exp(-float("inf"))

                    denom_ih = sum_exp / ln2
                    out_vec = [0.0] * head_dim

                    # Second pass: accumulate output
                    for j in tl.static_range(0, BLOCK_K):
                        j_global = j0 + j
                        if j_global >= K:
                            break
                        mask_ij = tl.load(mask_ptr + i_global * K + j_global)
                        q_base = q_ptr + i_global * H * head_dim + h * head_dim
                        k_base = k_ptr + j_global * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                        if (mask_ij != 0) and (j_global < (i_global + 1 + delta)):
                            numerator_j = tl.exp(logits_ij - lse_val)
                            # v_expanded[j,h,:] base = j * (H * head_dim) + h * head_dim
                            v_base = v_ptr + j_global * (H * head_dim) + h * head_dim
                            for d in tl.static_range(0, head_dim):
                                vd = tl.load(v_base + d)
                                out_vec[d] += (numerator_j * vd) / denom_ih

                    # Store output[i,h,:]
                    out_base = output_ptr + i_global * H * head_dim + h * head_dim
                    for d in tl.static_range(0, head_dim):
                        tl.store(out_base + d, out_vec[d])

                    # Store lse[i,h]
                    lse_entry = lse_ptr + i_global * H + h
                    tl.store(lse_entry, lse_val)

            # Move to next j tile
        # Move to next i tile


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels require CUDA tensors"
        # Cast to float32 for compute
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Shapes
        assert q_f32.shape[1:] == (32, 128), "q must have shape [*, 32, 128]"
        assert k_f32.shape[1:] == (8, 128), "k must have shape [*, 8, 128]"
        assert v_f32.shape[1:] == (8, 128), "v must have shape [*, 8, 128]"

        total_q = q_f32.shape[0]
        device = q_f32.device

        # Output buffers (float32 compute; return bfloat16)
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        # lse initialized to -inf (no torch.full on host)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=device)

        # Number of segments
        len_indptr = qo_indptr.numel()
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice for this segment
            q_batch = q_f32[q_start:q_end]      # [Q, 32, 128]
            k_batch = k_f32[kv_start:kv_end]    # [K, 8, 128]
            v_batch = v_f32[kv_start:kv_end]    # [K, 8, 128]

            Q = q_batch.shape[0]
            K = k_batch.shape[0]

            # Expand k and v along heads (GQA mapping: 8 -> 32)
            gqa_ratio = 4
            k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)   # [K, 32, 128]
            v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)   # [K, 32, 128]

            # Precompute bounded attention mask: mask[i, j] = 1 if j < (i + 1 + delta), else 0
            delta = K - Q
            q_positions = torch.arange(Q, device=device)  # [Q]
            kv_positions = torch.arange(K, device=device)  # [K]
            mask_mat = (kv_positions[None, :] < (q_positions[:, None] + 1 + delta))  # [Q, K], bool
            # Convert to int8 for Triton
            mask_int8 = mask_mat.to(torch.int8)

            # Launch Triton kernel for this segment: one program per segment tile
            segment_attention_kernel_correct[(1,)](
                q_batch, k_expanded, v_expanded, mask_int8,
                output[q_start:q_end], lse[q_start:q_end],
                float(sm_scale),
                Q, K, delta,
                H=32, gqa_ratio=4,
                ln2=1.4426950408889634,  # log(2)
                BLOCK_Q=64, BLOCK_K=64,
                head_dim=128,
                num_warps=4, num_stages=2
            )

        #


def run(*args):
    return ModelNew()(*args)
