import math
import torch
import triton
import triton.language as tl


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure tensors are on the same device
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be CUDA tensors."
        assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16, \
            "q, k, v must be bfloat16 as in original."

        # Cast to float32 for computation (matching original)
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Get shapes and constraints
        total_q, num_qo_heads, head_dim = q_f32.shape
        total_kv, num_kv_heads, _ = k_f32.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Compute len_indptr and validity
        len_indptr = qo_indptr.shape[0]
        assert qo_indptr[-1].item() == total_q
        assert kv_indptr[-1].item() == total_kv

        # Precompute gqa ratio
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Process each segment b
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue  # skip empty segments

            # Slice batch tensors
            q_batch = q_f32[q_start:q_end]          # [Q, 32, 128]
            k_batch = k_f32[kv_start:kv_end]        # [K, 8, 128]
            v_batch = v_f32[kv_start:kv_end]        # [K, 8, 128]

            Q = q_batch.shape[0]   # number of query tokens in this segment
            K = k_batch.shape[0]   # number of key/value tokens in this segment
            H = num_qo_heads       # output heads = 32

            # Expand k and v along heads (GQA mapping: 8 -> 32 by repeating 4x)
            k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)  # [K, 32, 128]
            v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)  # [K, 32, 128]

            # Precompute mask per segment: j < (i + 1 + delta), where delta = K - Q (since qo/kv tokens may differ)
            delta = K - Q
            # Build indices for mask
            q_positions = torch.arange(Q, device=device)  # [Q]
            kv_positions = torch.arange(K, device=device)  # [K]
            # mask[i, j] = (j < (i + 1 + delta))
            mask = (kv_positions[None, :] < (q_positions[:, None] + 1 + delta))  # [Q, K]
            # Triton expects 1-byte mask (int8/bool); convert to int8 for bandwidth
            mask_int8 = mask.to(torch.int8)

            # Launch Triton kernel: one program per segment
            grid = (1,)
            compute_atten_mask_and_output_kernel[grid](
                q_batch,            # pointer to [Q, 32, 128]
                k_expanded,         # pointer to [K, 32, 128]
                v_expanded,         # pointer to [K, 32, 128]
                mask_int8,          # pointer to [Q, K] int8 mask
                output[q_start:q_end],  # pointer to [Q, 32, 128] output
                lse[q_start:q_end],      # pointer to [Q, 32] lse
                sm_scale,           # float32 scale
                Q=Q, K=K, H=H,
                BLOCK_Q=32, BLOCK_K=64,
                num_warps=4,
                num_stages=2,
            )

        # Return output in bfloat16 to match original, and lse in float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
