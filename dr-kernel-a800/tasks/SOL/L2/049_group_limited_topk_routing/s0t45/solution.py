import torch
import triton
import triton.language as tl


# GEMM: logits = hidden @ weight.T
@triton.jit
def _linear_mm_kernel(
    hidden_ptr,      # *f32, shape [M, K]
    weight_ptr,      # *f32, shape [N, K]
    logits_ptr,      # *f32, shape [M, N]
    M, N, K,
    stride_hm, stride_hk,
    stride_wk, stride_wn,
    stride_lm, stride_ln,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    if m >= M or n >= N:
        return
    # Accumulator
    acc = 0.0
    # Iterate over K
    for k in range(0, K):
        a = tl.load(hidden_ptr + m * stride_hm + k * stride_hk)
        b = tl.load(weight_ptr + n * stride_wn + k * stride_wk)
        acc += a * b
    tl.store(logits_ptr + m * stride_lm + n * stride_ln, acc)


# Elementwise: scores = sigmoid(logits) + bias
@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr,      # *f32, [M, N]
    bias_ptr,        # *f32, [N]
    scores_ptr,      # *f32, [M, N]
    M, N,
    stride_lm, stride_ln,
    stride_b,
    stride_sm, stride_sn,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    if m >= M or n >= N:
        return
    val = tl.load(logits_ptr + m * stride_lm + n * stride_ln)
    # Sigmoid
    val = 1.0 / (1.0 + tl.exp(-val))
    b = tl.load(bias_ptr + n * stride_b)
    val += b
    tl.store(scores_ptr + m * stride_sm + n * stride_sn, val)


# Group top-2 sum per token: input scores [M, N], output group_scores [M, 8]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,      # *f32, [M, N]
    group_scores_ptr,# *f32, [M, 8]
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    # Sum of top-2 in each of the 8 groups of 32
    for g in range(8):
        start = g * 32
        # Initialize top1, top2 with first two values; if N < 32, we safely set to -inf
        idx = start
        v1 = tl.load(scores_ptr + m * stride_sm + idx * stride_sn)
        idx += 1
        v2 = tl.load(scores_ptr + m * stride_sm + idx * stride_sn)
        top1 = v1
        top2 = v2
        # Scan remaining 28 in the group
        for i in range(2, 32):
            idx = start + i
            vi = tl.load(scores_ptr + m * stride_sm + idx * stride_sn)
            if vi > top1:
                # top2 gets old top1, top1 gets vi
                old1 = top1
                top1 = vi
                top2 = old1
            elif vi > top2:
                top2 = vi
        total = top1 + top2
        tl.store(group_scores_ptr + m * stride_gm + g * stride_gn, total)


# Select top-4 group indices per token: input group_scores [M, 8], output group_idx [M, 4]
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,  # *f32, [M, 8]
    top4_ptr,          # *i32, [M, 4]
    M,
    stride_gm, stride_gn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    # Iteratively pick maxima (4 times) and store
    for r in range(4):
        maxv = -float('inf')
        max_idx = -1
        for g in range(8):
            v = tl.load(group_scores_ptr + m * stride_gm + g * stride_gn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, g, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(top4_ptr + m * stride_tm + r * stride_tn, max_idx)


# Mask non-selected groups: input scores [M, N], group_idx [M, 4], output masked_scores [M, N]
@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr,          # *f32, [M, N]
    group_idx_ptr,       # *i32, [M, 4]
    masked_ptr,          # *f32, [M, N]
    M, N,
    stride_sm, stride_sn,
    stride_im, stride_in,
    stride_mm, stride_mn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    for g in range(4):
        idx = tl.load(group_idx_ptr + m * stride_im + g * stride_in)  # i32
        # Compute start and end of this group
        start = idx * 32
        end = start + 32
        for n in range(N):
            # If n not in [start, end), set to -inf
            cond = (n >= start) & (n < end)
            val = tl.load(scores_ptr + m * stride_sm + n * stride_sn)
            val = tl.where(cond, val, -float('inf'))
            tl.store(masked_ptr + m * stride_mm + n * stride_mn, val)


# Final selection of top-8 within masked scores per token
@triton.jit
def _final_top8_kernel(
    scores_ptr,      # *f32, [M, N] (masked)
    top8_idx_ptr,    # *i32, [M, 8]
    M, N,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    # Iteratively select maxima 8 times
    for r in range(8):
        maxv = -float('inf')
        max_idx = -1
        for n in range(0, N):
            v = tl.load(scores_ptr + m * stride_sm + n * stride_sn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, n, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(top8_idx_ptr + m * stride_tm + r * stride_tn, max_idx)


# Normalize selected scores and apply scaling factor
@triton.jit
def _normalize_and_scale_kernel(
    scores_ptr,      # *f32, [M, N] (masked scores, we'll gather selected)
    top8_idx_ptr,    # *i32, [M, 8]
    top8_weight_ptr, # *f32, [M, 8]
    M, N,
    scale,           # f32
    stride_sm, stride_sn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    total = 0.0
    for r in range(8):
        idx = tl.load(top8_idx_ptr + m * stride_tm + r * stride_tn)  # i32
        v = tl.load(scores_ptr + m * stride_sm + idx * stride_sn)
        total += v
    total = total + 1e-20  # epsilon
    # Write scaled normalized weights
    for r in range(8):
        idx = tl.load(top8_idx_ptr + m * stride_tm + r * stride_tn)
        v = tl.load(scores_ptr + m * stride_sm + idx * stride_sn)
        w = (v / total) * scale
        tl.store(top8_weight_ptr + m * stride_tm + r * stride_tn, w)


# Helper: select first 8 columns and scale (simple Triton kernel for completeness; forward uses it for weights)
@triton.jit
def _select_first8_and_scale_kernel(
    scores_ptr,      # *f32, [M, N]
    top8_idx_ptr,    # *i32, [M, 8]
    top8_weight_ptr, # *f32, [M, 8]
    M, N,
    routed_scale,    # f32
    stride_sm, stride_sn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    total = 0.0
    for r in range(8):
        n = r  # first 8 columns
        v = tl.load(scores_ptr + m * stride_sm + n * stride_sn)
        total += v
    total = total + 1e-20
    for r in range(8):
        n = r
        v = tl.load(scores_ptr + m * stride_sm + n * stride_sn)
        w = (v / total) * routed_scale
        tl.store(top8_weight_ptr + m * stride_tm + r * stride_tn, w)
        tl.store(top8_idx_ptr + m * stride_tm + r * stride_tn, n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype and device
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]
        assert weight.shape[1] == K, "weight second dim must match hidden_dim"
        assert expert_bias.shape[0] == N, "expert_bias length must match num_experts"

        # 1) Triton GEMM: logits = hidden @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        grid_mm = (M, N)
        _linear_mm_kernel[grid_mm](
            hidden_states, weight, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(1), weight.stride(0),  # weight[n, k] strides
            logits.stride(0), logits.stride(1),
            num_warps=4, num_stages=2,
        )

        # 2) Triton elementwise: scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        grid_bias = (M, N)
        _sigmoid_add_bias_kernel[grid_bias](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            expert_bias.stride(0),
            scores.stride(0), scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 3) Triton group top-2 sum: group_scores [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        grid_group = (M,)
        _group_top2_sum_kernel[grid_group](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 4) Triton select top-4 group indices: top4_groups [M, 4]
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        grid_top4 = (M,)
        _select_top4_groups_kernel[grid_top4](
            group_scores, top4_groups,
            M,
            group_scores.stride(0), group_scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            num_warps=1, num_stages=1,
        )

        # 5) Triton mask non-selected groups in scores -> masked_scores [M, N]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=device)
        grid_mask = (M,)
        _mask_nonselected_groups_kernel[grid_mask](
            scores, top4_groups, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 6) Triton final top-8 selection indices from masked_scores: top8_idx [M, 8]
        top8_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        grid_final = (M,)
        _final_top8_kernel[grid_final](
            masked_scores, top8_idx,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            num_warps=1, num_stages=1,
        )

        # 7) Triton normalize and scale to produce topk_weight [M, 8]
        top8_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        grid_norm = (M,)
        _normalize_and_scale_kernel[grid_norm](
            masked_scores, top8_idx, top8_weight,
            M, N,
            routed_scaling_factor,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            num_warps=1, num_stages=1,
        )

        # Convert indices to int64 as required by original (gathered positions from masked_scores)
        topk_idx = top8_idx.to(torch.int64)

        return topk_idx, top8_weight


# Optional: quick local test
if __name__ == "__main__":
    # Example inputs
    M = 2048
    K = 256
    N = 256
    hidden_states = torch.randn(M, K, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, K, device='cuda', dtype=torch.float32)  # [N, K]
    expert_bias = torch.randn(N, device='cuda', dtype=torch.float32)
    routed_scaling_factor = 1.5

    model = ModelNew().cuda()
    idx, weights = model(hidden_states, weight, expert_bias, routed_scaling_factor)
    print("topk_idx shape:", idx.shape, idx.dtype)
    print("topk_weight shape:", weights.shape, weights.dtype)


def run(*args):
    return ModelNew()(*args)
