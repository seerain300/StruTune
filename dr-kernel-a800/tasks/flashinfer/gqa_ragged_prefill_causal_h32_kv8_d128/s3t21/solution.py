import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_kernel(
    q_ptr,       # *float32, [Q, 32, 128]
    k_ptr,       # *float32, [K, 32, 128]
    v_ptr,       # *float32, [K, 32, 128]
    output_ptr,  # *float32, [Q, 32, 128]
    lse_ptr,     # *float32, [Q, 32]
    sm_scale,    # float32
    Q, K,        # int32
    delta,       # int32 = K - Q
    H: tl.constexpr,               # 32
    gqa_ratio: tl.constexpr,       # 4
    ln2,          # float32 = log(2)
    head_dim: tl.constexpr,        # 128
    BLOCK_Q: tl.constexpr,         # e.g., 64
    BLOCK_K: tl.constexpr,         # e.g., 64
):
    # Process segment in tiles over Q and K. We avoid dynamic loops by using tl.static_range.
    for q0 in tl.static_range(0, Q, BLOCK_Q):
        for h in range(0, H):
            # 1) Compute lse[i,h] = max over j of logits[i,h,j] (masked). We need per-(i,h) lse for all i in this tile.
            # First compute lse per i for this h
            lse_vals = [0.0] * BLOCK_Q
            # Nested loops over K tiles and then per i within tile
            for k0 in tl.static_range(0, K, BLOCK_K):
                for i in tl.static_range(0, BLOCK_Q):
                    qi = q0 + i
                    if qi >= Q:
                        break
                    lse_val = -float("inf")
                    for j in tl.static_range(0, BLOCK_K):
                        kj = k0 + j
                        if kj >= K:
                            continue
                        # Apply bounded attention mask: j < (qi + 1 + delta)
                        if kj >= (qi + 1 + delta):
                            logits_ij = -float("inf")
                        else:
                            # Dot product over head_dim
                            q_base = q_ptr + qi * H * head_dim + h * head_dim
                            k_base = k_ptr + kj * H * head_dim + h * head_dim
                            dot = 0.0
                            for d in tl.static_range(0, head_dim):
                                qd = tl.load(q_base + d)
                                kd = tl.load(k_base + d)
                                dot += qd * kd
                            logits_ij = dot * sm_scale
                        lse_val = tl.maximum(lse_val, logits_ij)
                    lse_vals[i] = lse_val

            # 2) For each i in tile, compute denom = sum_j exp(logits[i,h,j] - lse_vals[i]) / ln2
            # and output vector = sum_j soft_j * v_expanded[j,h,:] with soft_j = exp(...)/denom
            for k0 in tl.static_range(0, K, BLOCK_K):
                for i in tl.static_range(0, BLOCK_Q):
                    qi = q0 + i
                    if qi >= Q:
                        break
                    sum_exp = 0.0
                    for j in tl.static_range(0, BLOCK_K):
                        kj = k0 + j
                        if kj >= K:
                            continue
                        if kj >= (qi + 1 + delta):
                            exp_j = 0.0
                        else:
                            q_base = q_ptr + qi * H * head_dim + h * head_dim
                            k_base = k_ptr + kj * H * head_dim + h * head_dim
                            dot = 0.0
                            for d in tl.static_range(0, head_dim):
                                qd = tl.load(q_base + d)
                                kd = tl.load(k_base + d)
                                dot += qd * kd
                            logits_ij = dot * sm_scale
                            exp_j = tl.exp(logits_ij - lse_vals[i])
                        sum_exp += exp_j
                    denom = sum_exp / ln2

                    # 3) Compute output[i, h, :] = sum_j soft_j * v_expanded[j, h, :]
                    out_vec = [0.0] * head_dim
                    for j in tl.static_range(0, BLOCK_K):
                        kj = k0 + j
                        if kj >= K:
                            continue
                        if kj >= (qi + 1 + delta):
                            soft_j = 0.0
                        else:
                            q_base = q_ptr + qi * H * head_dim + h * head_dim
                            k_base = k_ptr + kj * H * head_dim + h * head_dim
                            dot = 0.0
                            for d in tl.static_range(0, head_dim):
                                qd = tl.load(q_base + d)
                                kd = tl.load(k_base + d)
                                dot += qd * kd
                            logits_ij = dot * sm_scale
                            soft_j = tl.exp(logits_ij - lse_vals[i]) / denom
                        v_base = v_ptr + kj * (H * head_dim) + h * head_dim
                        for d in tl.static_range(0, head_dim):
                            vd = tl.load(v_base + d)
                            out_vec[d] += soft_j * vd

                    # Store output[qi, h, :]
                    out_base = output_ptr + qi * H * head_dim + h * head_dim
                    for d in tl.static_range(0, head_dim):
                        tl.store(out_base + d, out_vec[d])

            # 4) Optionally, store lse per (qi,h) if lse_ptr is provided. Here we store only for real i in tile.
            for i in tl.static_range(0, BLOCK_Q):
                qi = q0 + i
                if qi >= Q:
                    break
                lse_entry = lse_ptr + qi * H + h
                tl.store(lse_entry, lse_vals[i])


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
        # lse (we won't initialize with torch.full to avoid host torch compute; kernel writes into it)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

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

            # Bounded attention delta
            delta = K - Q

            # Launch Triton kernel for this segment (grid size = 1, handle entire segment in the program)
            segment_attention_kernel[(1,)](
                q_batch, k_expanded, v_expanded,
                output[q_start:q_end], lse[q_start:q_end],
                float(sm_scale),
                Q, K, delta,
                H=32, gqa_ratio=4, ln2=1.4426950408889634,  # log(2)
                head_dim=128,
                BLOCK_Q=64, BLOCK_K=64,
                num_warps=4, num_stages=2
            )

        # Return output as bfloat16 (original) and lse as float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
