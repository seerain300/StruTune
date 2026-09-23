import torch
import math
import triton
import triton.language as tl


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants per the original model
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.head_dim = 128
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Shapes must match the original assumptions
        assert q.shape == (q.shape[0], self.num_qo_heads, self.head_dim)
        assert k.shape == (k.shape[0], self.num_kv_heads, self.head_dim)
        assert v.shape == (v.shape[0], self.num_kv_heads, self.head_dim)
        assert self.num_qo_heads == 32 and self.num_kv_heads == 8 and self.head_dim == 128

        total_q = q.shape[0]
        total_kv = k.shape[0]
        len_indptr = qo_indptr.shape[0]
        assert kv_indptr.shape[0] == len_indptr

        device = q.device
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch segment
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice and expand for GQA
            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end]  # [num_kv_tokens, 8, 128]

            # Expand K and V by GQA ratio along head dimension
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Compute in float32
            q_f32 = q_batch.to(torch.float32)  # [Q, H_q, D]
            k_f32 = k_expanded.to(torch.float32)  # [K, H_k, D]
            v_f32 = v_expanded.to(torch.float32)  # [K, H_v, D]

            # Allocate per-segment outputs (float32)
            output_seg = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
            lse_seg = torch.empty((num_q_tokens, self.num_qo_heads), dtype=torch.float32, device=device)

            # 1) Compute logits[q,h,k] = sum_d q[q,h,d] * k[k,h,d] for all q,k,h using Triton
            # Triton grid over (heads, d-blocks). No runtime loops inside kernel.
            _compute_logits_kernel[(self.num_qo_heads, 8)](  # 8 blocks of D=128
                q_f32, k_f32,
                output_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_f32.stride(0), q_f32.stride(1), q_f32.stride(2),
                k_f32.stride(0), k_f32.stride(1), k_f32.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=1, BLOCK_D=16, BLOCK_K=1
            )

            # 2) Compute lse[q,h] = logsumexp(logits[q,h,:]) / ln(2) with causal mask in Triton
            _lse_masked_kernel[(num_q_tokens, self.num_qo_heads)](
                output_seg, lse_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                self.ln2, (num_kv_tokens - num_q_tokens),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                BLOCK_Q=1, BLOCK_K=128
            )

            # 3) Compute output = softmax(logits) @ V_expanded with causal mask in Triton
            _softmax_output_kernel[(num_q_tokens, self.num_qo_heads)](
                output_seg, v_f32, lse_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                v_f32.stride(0), v_f32.stride(1), v_f32.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=1, BLOCK_D=16, BLOCK_K=128
            )

            # Copy segment results to global output/lse
            output[q_start:q_end] = output_seg.to(torch.bfloat16)
            lse[q_start:q_end] = lse_seg

        return output, lse


# Triton kernels (no runtime-dependent loops; use constexpr bounds)
@triton.jit
def _compute_logits_kernel(
    Q, K, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_stride_k, K_stride_h, K_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (heads, d_blocks)
    h = tl.program_id(0)
    d_block = tl.program_id(1)  # 0..7 for BLOCK_D=16
    d_idx = d_block * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]
    d_mask = d_idx < head_dim

    # For each k in K


def run(*args):
    return ModelNew()(*args)
