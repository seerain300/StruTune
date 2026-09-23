import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_lse_kernel(
    q_ptr, k_ptr, qo_indptr_ptr, kv_indptr_ptr,
    lse_ptr,         # [len_indptr, total_q, 32] float32
    total_q,         # int
    sm_scale,        # float32
    NUM_QO_HEADS: tl.constexpr,   # 32
    NUM_KV_HEADS: tl.constexpr,   # 8
    HEAD_DIM: tl.constexpr,       # 128
    GQA_RATIO: tl.constexpr,      # 4
):
    # Grid: (b, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load batch ranges
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    delta = num_kv_tokens - num_q_tokens  # causal window end

    # Base pointer for q vector: q[b*q_start + q_token, qo_head, :]
    q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM

    # Accumulator for LSE
    sum_exp = 0.0

    # Loop over original KV heads and GQA expansions (compile-time)
    for j in range(NUM_KV_HEADS):
        for r in range(GQA_RATIO):
            kv_pos = j * GQA_RATIO + r  # expanded position
            if kv_pos < (q_token + 1 + delta):
                # Load q row and k row
                q_row = tl.load(q_vec_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
                k_row_base = k_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM) + j * HEAD_DIM
                k_row = tl.load(k_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
                # Dot product
                val = 0.0
                for i in range(HEAD_DIM):
                    val += q_row[i] * k_row[i]
                val = val * sm_scale
                # Accumulate exp
                sum_exp += tl.exp(val)

    lse_val = tl.log(sum_exp) / 1.4426950408889634  # log2(sum_exp) = log(sum_exp) / ln(2)
    # Store lse[b, q_token, qo_head]
    out_index = b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head
    tl.store(lse_ptr + out_index, lse_val)


@triton.jit
def _compute_output_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    lse_ptr, output_ptr,  # [len_indptr, total_q, 32, 128] float32
    total_q,              # int
    sm_scale,             # float32
    NUM_QO_HEADS: tl.constexpr,   # 32
    NUM_KV_HEADS: tl.constexpr,   # 8
    HEAD_DIM: tl.constexpr,       # 128
    GQA_RATIO: tl.constexpr,      # 4
):
    # Grid: (b, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load batch ranges
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    delta = num_kv_tokens - num_q_tokens  # causal window end

    # Base pointer for output vector
    out_base = output_ptr + (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * HEAD_DIM

    # Load lse
    out_index = b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head
    lse_val = tl.load(lse_ptr + out_index)

    # Accumulate output across all allowed (j, r)
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for j in range(NUM_KV_HEADS):
        for r in range(GQA_RATIO):
            kv_pos = j * GQA_RATIO + r
            if kv_pos < (q_token + 1 + delta):
                # Recompute logits for this position
                q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM
                q_row = tl.load(q_vec_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
                k_row_base = k_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM) + j * HEAD_DIM
                k_row = tl.load(k_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
                val = 0.0
                for i in range(HEAD_DIM):
                    val += q_row[i] * k_row[i]
                val = val * sm_scale
                attn = tl.exp(val - lse_val)
                # Load V expanded: v_expanded[j, r, :] = v[kv_start + kv_pos, j, :]
                v_base = v_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM) + j * HEAD_DIM
                v_row = tl.load(v_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
                out_vec += attn * v_row

    # Store out_vec
    tl.store(out_base + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure inputs are on CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be CUDA tensors."
        device = q.device

        # Shapes
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = 4

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute lse per (b, q_token, qo_head)
        grid = (qo_indptr.numel(), total_q, num_qo_heads)
        _compute_lse_kernel[grid](
            q_f32, k_f32, qo_indptr, kv_indptr,
            lse,
            total_q,
            sm_scale,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            GQA_RATIO=gqa_ratio,
            num_warps=4,
            num_stages=2,
        )

        # Launch Triton kernel to compute output vectors
        grid2 = (qo_indptr.numel(), total_q, num_qo_heads)
        _compute_output_kernel[grid2](
            q_f32, k_f32, v_f32, qo_indptr, kv_indptr,
            lse, output,
            total_q,
            sm_scale,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            GQA_RATIO=gqa_ratio,
            num_warps=4,
            num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
