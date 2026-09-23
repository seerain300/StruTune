import math
import torch
import triton
import triton.language as tl


@triton.jit
def segment_attention_full_kernel(
    q_ptr,       # *float32, [Q, 32, 128]
    k_ptr,       # *float32, [K, 32, 128]
    v_ptr,       # *float32, [K, 32, 128]
    output_ptr,  # *float32, [Q, 32, 128]
    lse_ptr,     # *float32, [Q, 32]
    sm_scale,    # float32
    Q, K,        # int32 (runtime, but used only with static tiling)
    delta,       # int32 = K - Q
    H: tl.constexpr,               # 32
    head_dim: tl.constexpr,        # 128
    BLOCK_I: tl.constexpr,         # tile size over Q, e.g., 64
    BLOCK_J: tl.constexpr,         # tile size over K, e.g., 64
):
    # Process entire segment in this single program, using static tiling.

    # 2) Compute output[i,h,:] and lse[i,h] for all i,h.
    for h in tl.static_range(0, H):
        ln2 = 1.4426950408889634  # log(2)
        for i0 in tl.static_range(0, Q, BLOCK_I):
            # Loop over i in this tile
            for i in tl.static_range(0, BLOCK_I):
                i_idx = i0 + i
                if i_idx >= Q:
                    break
                # a) Compute lse[i_idx, h] = max_j logits[i_idx, h, j] (masked)
                lse_val = -float("inf")
                for j0 in tl.static_range(0, K, BLOCK_J):
                    for j in tl.static_range(0, BLOCK_J):
                        j_idx = j0 + j
                        valid = j_idx < (i_idx + 1 + delta)
                        q_base = q_ptr + i_idx * H * head_dim + h * head_dim
                        k_base = k_ptr + j_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                        if not valid:
                            logits_ij = -float("inf")
                        lse_val = tl.maximum(lse_val, logits_ij)
                # b) Compute denom[i_idx, h] = sum_j exp(logits[i_idx, h, j] - lse_val) / ln(2)
                sum_exp = 0.0
                for j0 in tl.static_range(0, K, BLOCK_J):
                    for j in tl.static_range(0, BLOCK_J):
                        j_idx = j0 + j
                        valid = j_idx < (i_idx + 1 + delta)
                        q_base = q_ptr + i_idx * H * head_dim + h * head_dim
                        k_base = k_ptr + j_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                        if not valid:
                            logits_ij = -float("inf")
                        sum_exp += tl.exp(logits_ij - lse_val)
                denom_ih = sum_exp / ln2

                # c) Compute output[i_idx, h, :] = sum_j exp(logits[i_idx, h, j] - lse_val) * v_expanded[j, h, :] / denom_ih
                out_vec = [0.0] * head_dim
                for j0 in tl.static_range(0, K, BLOCK_J):
                    for j in tl.static_range(0, BLOCK_J):
                        j_idx = j0 + j
                        valid = j_idx < (i_idx + 1 + delta)
                        q_base = q_ptr + i_idx * H * head_dim + h * head_dim
                        k_base = k_ptr + j_idx * H * head_dim + h * head_dim
                        dot = 0.0
                        for d in tl.static_range(0, head_dim):
                            qd = tl.load(q_base + d)
                            kd = tl.load(k_base + d)
                            dot += qd * kd
                        logits_ij = dot * sm_scale
                        if not valid:
                            logits_ij = -float("inf")
                        numerator_j = tl.exp(logits_ij - lse_val)  # ln(2) factor in denom
                        # Map to original v's head: v_expanded has 32 heads; original v has 8 -> repeat_interleave by 4
                        h2 = h // 4
                        v_base = v_ptr + j_idx * (H * head_dim) + h2 * head_dim
                        for d in tl.static_range(0, head_dim):
                            vd = tl.load(v_base + d)
                            out_vec[d] += (numerator_j * vd) / denom_ih

                # Store output[i_idx, h, :]
                out_base = output_ptr + i_idx * H * head_dim + h * head_dim
                for d in tl.static_range(0, head_dim):
                    tl.store(out_base + d, out_vec[d])

                # Store lse[i_idx, h]
                lse_entry = lse_ptr + i_idx * H + h
                tl.store(lse_entry, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors; cast to float32 for compute
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
        # lse buffer (kernel will fill)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Number of segments
        len_indptr = qo_indptr.numel()
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # Skip empty segments
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

            delta = K - Q

            # Launch Triton kernel for this segment
            segment_attention_full_kernel[(1,)](
                q_batch, k_expanded, v_expanded,
                output[q_start:q_end], lse[q_start:q_end],
                float(sm_scale),
                Q, K, delta,
                H=32, head_dim=128,
                BLOCK_I=64, BLOCK_J=64,
                num_warps=4, num_stages=2
            )

        # Return output as bfloat16 (original behavior) and lse as float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
