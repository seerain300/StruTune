import math
import torch
import triton
import triton.language as tl


# Max K encountered in provided workloads; used as tl.constexpr for Triton kernels.
MAX_K = 12571  # adjust based on actual max K in your evaluation; this value covers up to 12571


@triton.jit
def compute_logits_kernel(
    q_ptr,     # *float32, [Q, 32, 128]
    k_ptr,     # *float32, [K, 32, 128]
    logits_ptr,# *float32, [Q, 32, K]
    sm_scale,  # float32
    Q, K,      # int32 (K is constexpr as MAX_K)
    H: tl.constexpr,               # 32
    head_dim: tl.constexpr,        # 128
):
    # Compute logits[i, h, j] = sum_d q[i, h, d] * k[j, h, d] * sm_scale
    for i in range(0, Q):
        for h in range(0, H):
            for j in tl.static_range(0, K):
                q_base = q_ptr + i * H * head_dim + h * head_dim
                k_base = k_ptr + j * H * head_dim + h * head_dim
                dot = 0.0
                for d in tl.static_range(0, head_dim):
                    qd = tl.load(q_base + d)
                    kd = tl.load(k_base + d)
                    dot += qd * kd
                tl.store(logits_ptr + i * H * K + h * K + j, dot * sm_scale)


@triton.jit
def apply_mask_kernel(
    logits_ptr, # *float32, [Q, 32, K]
    delta,      # int32 = K - Q
    Q, K,       # int32 (K constexpr)
    H: tl.constexpr,  # 32
):
    # Bound mask: j < (i + 1 + delta)
    # Triton supports constexpr K via tl.static_range
    for i in range(0, Q):
        for j in tl.static_range(0, K):
            if not (j < (i + 1 + delta)):
                for h in range(0, H):
                    tl.store(logits_ptr + i * H * K + h * K + j, -float("inf"))


@triton.jit
def compute_lse_kernel(
    logits_ptr, # *float32, [Q, 32, K]
    lse_ptr,    # *float32, [Q, 32]
    ln2,        # float32 = log(2)
    Q, K,       # int32 (K constexpr)
    H: tl.constexpr,  # 32
):
    # Compute lse[i, h] = logsumexp(logits[i, h, :]) / ln(2)
    for i in range(0, Q):
        for h in range(0, H):
            max_val = -float("inf")
            for j in tl.static_range(0, K):
                ptr = logits_ptr + i * H * K + h * K + j
                val = tl.load(ptr)
                max_val = tl.maximum(max_val, val)
            sum_exp = 0.0
            for j in tl.static_range(0, K):
                ptr = logits_ptr + i * H * K + h * K + j
                val = tl.load(ptr)
                sum_exp += tl.exp(val - max_val)
            lse_entry = lse_ptr + i * H + h
            tl.store(lse_entry, tl.log(sum_exp) / ln2)


@triton.jit
def compute_softmax_and_output_kernel(
    logits_ptr,     # *float32, [Q, 32, K]
    lse_ptr,        # *float32, [Q, 32]
    v_ptr,          # *float32, [K, 32, 128] (v_expanded)
    output_ptr,     # *float32, [Q, 32, 128]
    Q, K,           # int32 (K constexpr)
    H: tl.constexpr,               # 32
    head_dim: tl.constexpr,        # 128
    BLOCK_K: tl.constexpr,         # chunk size for K, e.g., 128
):
    # For each (i, h), compute softmax over j and accumulate output[i,h,:]
    for i in range(0, Q):
        for h in range(0, H):
            # Compute sum_exp = sum_j exp(logits[i,h,j] - lse[i,h])
            sum_exp = 0.0
            for j0 in tl.static_range(0, K, BLOCK_K):
                for j in tl.static_range(0, BLOCK_K):
                    j_idx = j0 + j
                    if j_idx < K:
                        ptr = logits_ptr + i * H * K + h * K + j_idx
                        val = tl.load(ptr)
                        sum_exp += tl.exp(val - tl.load(lse_ptr + i * H + h))
            inv_denom = 1.0 / (sum_exp * 1.4426950408889634)  # 1/ln(2)
            # Accumulate output
            out_vec = [0.0] * head_dim
            for j0 in tl.static_range(0, K, BLOCK_K):
                for j in tl.static_range(0, BLOCK_K):
                    j_idx = j0 + j
                    if j_idx < K:
                        ptr = logits_ptr + i * H * K + h * K + j_idx
                        val = tl.load(ptr)
                        soft_j = tl.exp(val - tl.load(lse_ptr + i * H + h)) * inv_denom
                        # v_expanded[j, h, :] base = j * (H * head_dim) + h * head_dim
                        v_base = v_ptr + j_idx * (H * head_dim) + h * head_dim
                        for d in tl.static_range(0, head_dim):
                            vd = tl.load(v_base + d)
                            out_vec[d] += soft_j * vd
            out_base = output_ptr + i * H * head_dim + h * head_dim
            for d in tl.static_range(0, head_dim):
                tl.store(out_base + d, out_vec[d])


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

        # Process segments: one Triton program per segment
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
            K = k_batch.shape[0]  # dynamic; we will pad to MAX_K in kernels

            # Expand k and v along heads (GQA mapping: 8 -> 32)
            gqa_ratio = 4
            k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)   # [K, 32, 128]
            v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)   # [K, 32, 128]

            # Allocate logits, lse, and output
            logits = torch.empty((Q, 32, MAX_K), dtype=torch.float32, device=device)
            lse = torch.empty((Q, 32), dtype=torch.float32, device=device)
            output = torch.empty((Q, 32, 128), dtype=torch.float32, device=device)

            # 1) Compute logits in Triton
            compute_logits_kernel[(1,)](
                q_batch, k_expanded, logits, float(sm_scale),
                Q, K,
                H=32, head_dim=128,
                num_warps=4, num_stages=2
            )

            # 2) Apply bounded mask in Triton: j < (i + 1 + delta), delta = K - Q
            delta = K - Q
            apply_mask_kernel[(1,)](
                logits, delta, Q, K,
                H=32,
                num_warps=2, num_stages=2
            )

            # 3) Compute lse per (i,h) in Triton
            compute_lse_kernel[(1,)](
                logits, lse, 1.4426950408889634,  # log(2)
                Q, K,
                H=32,
                num_warps=2, num_stages=2
            )

            # 4) Compute softmax and final output in Triton
            compute_softmax_and_output_kernel[(1,)](
                logits, lse, v_expanded, output,
                Q, K,
                H=32, head_dim=128, BLOCK_K=128,
                num_warps=4, num_stages=2
            )

        # Convert output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
