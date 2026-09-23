import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) Triton kernel: logits = hidden_states @ weight^T + expert_bias
# A: [M, K] (row-major), W: [N, K] (row-major), bias: [N]
@triton.jit
def linear_bias_kernel(
    A_ptr,        # *fp32
    W_ptr,        # *fp32
    BIAS_ptr,     # *fp32
    OUT_ptr,      # *fp32 logits: [M, N]
    M: tl.constexpr,
    N: tl.constexpr,   # num_experts = 256
    K: tl.constexpr,   # hidden_dim
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    # Accumulator for logits for this m across N
    acc = tl.zeros((N,), dtype=tl.float32)
    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # For each output expert n in tiles
        for n0 in range(0, N, BLOCK_N):
            n_offsets = n0 + tl.arange(0, BLOCK_N)
            # Load W[n, k] -> [BLOCK_N, BLOCK_K]
            W_tile = tl.load(W_ptr + n_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk)
            # Load A[m, k] -> [BLOCK_K]
            A_row = tl.load(A_ptr + m * stride_am + (k_offsets * stride_ak))
            # Accumulate acc[n] += sum(W_tile * A_row)
            acc += tl.sum(W_tile * A_row[None, :], axis=1)
        # Add bias[n]
        bias_vals = tl.load(BIAS_ptr + n_offsets * stride_wn)
        acc += bias_vals
    # Store logits[m, :]
    for n in range(0, N):
        tl.store(OUT_ptr + m * stride_om + n * stride_on, acc[n])


# 2) Triton kernel: scores = sigmoid(logits) + expert_bias
# LOGITS: [M, N], BIAS: [N], OUT: [M, N]
@triton.jit
def sigmoid_bias_kernel(
    LOGITS_ptr,   # *fp32
    BIAS_ptr,     # *fp32
    OUT_ptr,      # *fp32
    M: tl.constexpr,
    N: tl.constexpr,
    stride_lm, stride_ln,
    stride_bn,
    stride_om, stride_on,
):
    m = tl.program_id(0)
    for n in range(0, N):
        x = tl.load(LOGITS_ptr + m * stride_lm + n * stride_ln)
        s = 1.0 / (1.0 + tl.exp(-x))
        b = tl.load(BIAS_ptr + n * stride_bn)
        y = s + b
        tl.store(OUT_ptr + m * stride_om + n * stride_on, y)


# 3) Triton kernel: compute group scores per token
# SCORES: [M, N], GROUP_SCORES: [M, 8]
# Grouping: 8 groups of 32 experts each
@triton.jit
def compute_group_scores_kernel(
    SCORES_ptr,   # *fp32, [M,N]
    GROUPS_ptr,   # *fp32, [M,8]
    M: tl.constexpr,
    N: tl.constexpr,       # num_experts = 256
    EP_GROUP: tl.constexpr,# 32
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    m = tl.program_id(0)
    for g in range(0, 8):
        start = g * EP_GROUP
        max1 = -float('inf')
        max2 = -float('inf')
        for e in range(0, EP_GROUP):
            idx = start + e
            score = tl.load(SCORES_ptr + m * stride_sm + idx * stride_sn)
            if score > max1:
                max2 = max1
                max1 = score
            elif score > max2:
                max2 = score
        group_score = max1 + max2
        tl.store(GROUPS_ptr + m * stride_gm + g * stride_gn, group_score)


# 4) Triton kernel: select top-4 groups per token (iterative elimination, sorted=False)
# GROUP_SCORES: [M,8], SELECTED_GROUPS: [M,4] int32
@triton.jit
def select_top4_groups_kernel(
    GROUP_SCORES_ptr,  # *fp32, [M,8]
    SELECTED_GROUPS_ptr,  # *int32, [M,4]
    M: tl.constexpr,
    GROUPS: tl.constexpr,  # 8
    stride_gm, stride_gn,
    stride_sm, stride_sn,
):
    m = tl.program_id(0)
    used = tl.zeros((GROUPS,), dtype=tl.int1)
    for k in range(0, 4):
        curr_max = -float('inf')
        sel_group = 0
        for g in range(0, GROUPS):
            score = tl.load(GROUP_SCORES_ptr + m * stride_gm + g * stride_gn)
            if (not used[g]) and (score > curr_max):
                curr_max = score
                sel_group = g
        used[sel_group] = True
        tl.store(SELECTED_GROUPS_ptr + m * stride_sm + k * stride_sn, sel_group)


# 5) Triton kernel: mask scores with selected groups (set non-selected groups to -inf)
# SCORES: [M,N], SELECTED_GROUPS: [M,4], MASKED_OUT: [M,N]
@triton.jit
def mask_scores_with_groups_kernel(
    SCORES_ptr,            # *fp32, [M,N]
    SELECTED_GROUPS_ptr,   # *int32, [M,4]
    MASKED_OUT_ptr,        # *fp32, [M,N]
    M: tl.constexpr,
    N: tl.constexpr,       # 256
    GROUPS: tl.constexpr,  # 8
    EP_GROUP: tl.constexpr, # 32
    stride_sm, stride_sn,
    stride_sg, stride_sgi,  # strides for selected_groups
    stride_mm, stride_mn,
):
    m = tl.program_id(0)
    for g in range(0, GROUPS):
        flag = 0
        for k in range(0, 4):
            sel = tl.load(SELECTED_GROUPS_ptr + m * stride_sg + k * stride_sgi)
            if sel == g:
                flag = 1
                break
        if flag == 0:
            for e in range(0, N):
                score = tl.load(SCORES_ptr + m * stride_sm + e * stride_sn)
                tl.store(MASKED_OUT_ptr + m * stride_mm + e * stride_mn, -float('inf'))
        # else: keep scores (we can leave as-is)


# 6) Triton kernel: final selection of top-8 from masked scores, compute weight using original logits
# LOGITS: [M,N], MASKED_SCORES: [M,N], OUT_IDX: [M,8] int32, OUT_WEIGHT: [M,8] float32
@triton.jit
def final_top8_with_weight_and_normalize_kernel(
    LOGITS_ptr,           # *fp32, [M,N]
    MASKED_SCORES_ptr,    # *fp32, [M,N]
    OUT_IDX_ptr,          # *int32, [M,8]
    OUT_WEIGHT_ptr,       # *fp32, [M,8]
    M: tl.constexpr,
    N: tl.constexpr,      # 256
    routed_scaling_factor,  # runtime float
    stride_lm, stride_ln,
    stride_ms, stride_msn,
    stride_i0m, stride_i0n,
    stride_w0m, stride_w0n,
    eps: tl.constexpr = 1e-20,
):
    m = tl.program_id(0)
    numerator = 0.0
    # Iteratively select 8 highest from masked scores
    for t in range(0, 8):
        curr_max = -float('inf')
        sel_idx = 0
        for e in range(0, N):
            score = tl.load(MASKED_SCORES_ptr + m * stride_ms + e * stride_msn)
            if score > curr_max:
                curr_max = score
                sel_idx = e
        tl.store(OUT_IDX_ptr + m * stride_i0m + t * stride_i0n, sel_idx)
        orig = tl.load(LOGITS_ptr + m * stride_lm + sel_idx * stride_ln)
        numerator += orig
    # Compute weight: routed_scaling_factor * (numerator / (numerator + eps))
    denom = numerator + eps
    # Triton doesn't allow arbitrary Python scalar arguments for computation without passing;
    # we pass routed_scaling_factor as runtime scalar here.
    weight_val = routed_scaling_factor * (numerator / denom)
    for t in range(0, 8):
        tl.store(OUT_WEIGHT_ptr + m * stride_w0m + t * stride_w0n, weight_val)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Device and shape checks
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]  # num experts
        assert N == 256, "num_experts must be 256"
        device = hidden_states.device
        assert device.type == 'cuda', "Triton kernels require CUDA device"

        # Cast and ensure contiguous
        A = hidden_states.to(torch.float32).contiguous()
        W = weight.to(torch.float32).contiguous()
        bias = expert_bias.to(torch.float32).contiguous()

        # Buffers
        logits = torch.empty((M, N), dtype=torch.float32, device=device)          # [M, N]
        scores = torch.empty((M, N), dtype=torch.float32, device=device)          # [M, N]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)    # [M, 8]
        selected_groups = torch.empty((M, 4), dtype=torch.int32, device=device)   # [M, 4]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=device)   # [M, N]
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=device)          # [M, 8]
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)     # [M, 8]

        # 1) Compute logits
        grid1 = (M,)
        linear_bias_kernel[grid1](
            A, W, bias, logits,
            M, N, K,
            A.stride(0), A.stride(1),
            W.stride(0), W.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
        )

        # 2) Compute scores = sigmoid(logits) + bias
        grid2 = (M,)
        sigmoid_bias_kernel[grid2](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            bias.stride(0),
            scores.stride(0), scores.stride(1),
        )

        # 3) Compute group scores per token
        grid3 = (M,)
        compute_group_scores_kernel[grid3](
            scores, group_scores,
            M, N, 32,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
        )

        # 4) Select top-4 groups per token
        grid4 = (M,)
        select_top4_groups_kernel[grid4](
            group_scores, selected_groups,
            M, 8,
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
        )

        # 5) Mask scores with selected groups (set non-selected groups to -inf)
        grid5 = (M,)
        mask_scores_with_groups_kernel[grid5](
            scores, selected_groups, masked_scores,
            M, N, 8, 32,
            scores.stride(0), scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
        )

        # 6) Final selection of top-8 from masked scores, compute weight using original logits
        grid6 = (M,)
        final_top8_with_weight_and_normalize_kernel[grid6](
            logits, masked_scores, topk_idx, topk_weight,
            M, N, routed_scaling_factor,
            logits.stride(0), logits.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
        )

        # Cast indices to int64 to match original return type for topk_idx
        topk_idx = topk_idx.to(torch.int64)
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
