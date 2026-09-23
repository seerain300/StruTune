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
    Q, K,        # int32 (runtime)
    delta,       # int32 = K - Q
    H: tl.constexpr,               # 32
    gqa_ratio: tl.constexpr,       # 4
    ln2,          # float32 = log(2)
    head_dim: tl.constexpr,        # 128
):
    # Compute attention for this segment: output[i, h, :] and lse[i, h]
    for i in range(0, Q):
        for h in range(0, H):
            # 1) Compute lse[i,h] = max over j of logits[i,h,j] (masked)
            lse_val = -float("inf")
            for j in range(0, K):
                if j < (i + 1 + delta):
                    # Dot product over head_dim
                    q_base = q_ptr + i * H * head_dim + h * head_dim
                    k_base = k_ptr + j * H * head_dim + h * head_dim
                    dot = 0.0
                    for d in range(0, head_dim):
                        qd = tl.load(q_base + d)
                        kd = tl.load(k_base + d)
                        dot += qd * kd
                    logits_ij = dot * sm_scale
                else:
                    logits_ij = -float("inf")
                lse_val = tl.maximum(lse_val, logits_ij)

            # 2) Compute denom[i,h] = sum_j exp(logits[i,h,j] - lse[i,h]) / ln(2)
            sum_exp = 0.0
            for j in range(0, K):
                if j < (i + 1 + delta):
                    q_base = q_ptr + i * H * head_dim + h * head_dim
                    k_base = k_ptr + j * H * head_dim + h * head_dim
                    dot = 0.0
                    for d in range(0, head_dim):
                        qd = tl.load(q_base + d)
                        kd = tl.load(k_base + d)
                        dot += qd * kd
                    logits_ij = dot * sm_scale
                else:
                    logits_ij = -float("inf")
                sum_exp += tl.exp(logits_ij - lse_val)
            denom_ih = sum_exp / ln2

            # 3) Compute output[i,h,:] = sum_j exp(logits[i,h,j] - lse[i,h]) * v_expanded[j,h,:] / denom_ih
            out_vec = [0.0] * head_dim
            for j in range(0, K):
                if j < (i + 1 + delta):
                    q_base = q_ptr + i * H * head_dim + h * head_dim
                    k_base = k_ptr + j * H * head_dim + h * head_dim
                    dot = 0.0
                    for d in range(0, head_dim):
                        qd = tl.load(q_base + d)
                        kd = tl.load(k_base + d)
                        dot += qd * kd
                    logits_ij = dot * sm_scale
                    numerator_j = tl.exp(logits_ij - lse_val)  # ln(2) already in denom_ih
                    # Map v_expanded back to original v's head: v_expanded has 32 heads; original v has 8 heads -> repeat_interleave by 4
                    h2 = h // gqa_ratio
                    v_base = v_ptr + j * (H * head_dim) + h2 * head_dim
                    for d in range(0, head_dim):
                        vd = tl.load(v_base + d)
                        out_vec[d] += (numerator_j * vd) / denom_ih
                else:
                    pass

            # Store output[i,h,:]
            out_base = output_ptr + i * H * head_dim + h * head_dim
            for d in range(0, head_dim):
                tl.store(out_base + d, out_vec[d])

            # Store lse[i,h]
            lse_entry = lse_ptr + i * H + h
            tl.store(lse_entry, lse_val)


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

        # Output buffers
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)  # we will fill via kernel

        len_indptr = qo_indptr.numel()
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # skip empty segments
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

            # Launch Triton kernel for this segment: one program per segment
            segment_attention_kernel[(1,)](
                q_batch, k_expanded, v_expanded,
                output[q_start:q_end], lse[q_start:q_end],
                float(sm_scale),
                Q, K, delta,
                H=32, gqa_ratio=4,
                ln2=1.4426950408889634,  # log(2)
                head_dim=128,
                num_warps=4, num_stages=2
            )

        # Return output as bfloat16 (original behavior) and lse as float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
