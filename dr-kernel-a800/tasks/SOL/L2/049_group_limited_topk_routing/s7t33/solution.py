import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) Triton kernel: logits = hidden_states @ weight^T + expert_bias
# A: [M, K] (row-major), W: [N, K] (row-major), bias: [N]
@triton.jit
def linear_bias_kernel(
    A_ptr,      # *fp32
    W_ptr,      # *fp32
    BIAS_ptr,   # *fp32
    OUT_ptr,    # *fp32 logits: [M, N]
    M: tl.constexpr,   # num_tokens
    N: tl.constexpr,   # num_experts (256)
    K: tl.constexpr,   # hidden_dim
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        # Load A tile: [BLOCK_M, BLOCK_K] -> A[m, k]
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0)
        # Load W^T tile: W[n, k] -> [BLOCK_N, BLOCK_K]
        W_tile_ptr = W_ptr + (offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk)
        W_mask = (offs_n[:, None] < N) & (k_ids[None, :] < K)
        W_tile = tl.load(W_tile_ptr, mask=W_mask, other=0.0)
        acc += tl.dot(A_tile, W_tile)

    # Add bias per expert
    bias_vals = tl.load(BIAS_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc += bias_vals[None, :]

    # Store to OUT[m, n]
    OUT_tile_ptr = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    OUT_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(OUT_tile_ptr, acc, mask=OUT_mask)


# 2) Triton kernel: elementwise sigmoid on logits
# IN: [M, N], OUT: [M, N]
@triton.jit
def sigmoid_kernel(
    IN_ptr,     # *fp32
    OUT_ptr,    # *fp32
    M: tl.constexpr,
    N: tl.constexpr,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Load tile
    in_ptr = IN_ptr + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(in_ptr, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    out_ptr = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptr, y, mask=mask)


# 3) Triton kernel: compute group scores = sum of top-2 per group (8 groups, 32 experts/group)
# INPUT: scores [M, N], OUTPUT: group_scores [M, 8]
@triton.jit
def compute_group_scores_kernel(
    SCORES_ptr,             # *fp32, [M, N]
    GROUP_SCORES_ptr,       # *fp32, [M, 8]
    M: tl.constexpr,
    N: tl.constexpr,        # 256
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    EXPERTS_PER_GROUP: tl.constexpr,   # 32
    GROUPS: tl.constexpr,               # 8
):
    m = tl.program_id(0)
    g = tl.program_id(1)  # 0..7
    base_e = g * EXPERTS_PER_GROUP
    # Compute top-2 within [base_e, base_e+31]
    max1 = -float('inf')
    max2 = -float('inf')
    for e in range(0, EXPERTS_PER_GROUP):
        score = tl.load(SCORES_ptr + (m * stride_sm + (base_e + e) * stride_sn))
        if score > max1:
            max2 = max1
            max1 = score
        elif score > max2:
            max2 = score
    group_score = max1 + max2
    tl.store(GROUP_SCORES_ptr + (m * stride_gm + g * stride_gn), group_score)


# 4) Triton kernel: select top-4 groups per token (iterative elimination), write to SELECTED_GROUPS[m,4] int32
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
    used = tl.zeros((GROUPS,), dtype=tl.int1)  # vector of used flags for groups
    # Perform 4 selections
    for k in range(0, 4):
        curr_max = -float('inf')
        sel_group = tl.zeros((), dtype=tl.int32)
        for g in range(0, GROUPS):
            score = tl.load(GROUP_SCORES_ptr + (m * stride_gm + g * stride_gn))
            if (not used[g]) and (score > curr_max):
                curr_max = score
                sel_group = g
        used[sel_group] = 1  # mark used
        tl.store(SELECTED_GROUPS_ptr + (m * stride_sm + k * stride_sn), sel_group)


# 5) Triton kernel: mask scores with selected groups (set non-selected groups to -inf) per token
@triton.jit
def mask_scores_with_groups_kernel(
    SCORES_ptr,            # *fp32, [M,N]
    SELECTED_GROUPS_ptr,   # *int32, [M,4]
    MASKED_OUT_ptr,        # *fp32, [M,N]
    M: tl.constexpr,
    N: tl.constexpr,       # 256
    GROUPS: tl.constexpr,  # 8
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
    stride_sm, stride_sn,
    stride_mom, stride_mon,
):
    m = tl.program_id(0)
    for g in range(0, GROUPS):
        found = 0
        for k in range(0, 4):
            sel = tl.load(SELECTED_GROUPS_ptr + (m * stride_sm + k * stride_sn))
            if sel == g:
                found = 1
                break
        # If not found, set all scores in this group to -inf
        for e in range(0, EXPERTS_PER_GROUP):
            idx = g * EXPERTS_PER_GROUP + e
            ptr = SCORES_ptr + (m * stride_sm + idx * stride_sn)
            # Load current score
            score = tl.load(ptr)
            # If not selected, set to -inf
            if found == 0:
                score = -float('inf')
            tl.store(MASKED_OUT_ptr + (m * stride_mom + idx * stride_mon), score)


# 6) Triton kernel: final top-8 selection and compute weight = routed * (sum(selected_logits)/sum_selected+eps)
# Iteratively select 8 indices from masked scores and accumulate selected logits in float32.
@triton.jit
def final_top8_with_weight_and_normalize_kernel(
    SCORES_ptr,                 # *fp32, [M,N]
    SCORES_COPY_ptr,            # *fp32, [M,N] (original scores before masking)
    OUT_IDX_ptr,                # *int32, [M,8]
    OUT_WEIGHT_ptr,             # *fp32,  [M,8]
    M: tl.constexpr,
    N: tl.constexpr,            # 256
    stride_sm, stride_sn,
    out_stride_im, out_stride_in,
    routed_scale: tl.float32,
    eps: tl.float32,
):
    m = tl.program_id(0)
    # We don't have direct gather with idx, so perform iterative elimination.
    # Selection state: indices are 0..N-1, not groups. We keep track of who is selected
    used = tl.zeros((N,), dtype=tl.int1)
    # Accumulator for sum of selected logits
    numerator = tl.zeros((), dtype=tl.float32)
    # Loop 8 times to pick top8
    for k in range(0, 8):
        curr_max = -float('inf')
        sel_idx = tl.zeros((), dtype=tl.int32)
        for e in range(0, N):
            # Skip if already used
            if used[e] == 0:
                score = tl.load(SCORES_ptr + (m * stride_sm + e * stride_sn))
                if score > curr_max:
                    curr_max = score
                    sel_idx = e
        used[sel_idx] = 1  # mark selected
        # Store index
        tl.store(OUT_IDX_ptr + (m * out_stride_im + k * out_stride_in), sel_idx)
        # Accumulate original logits (for normalization), from SCORES_COPY
        orig_score = tl.load(SCORES_COPY_ptr + (m * stride_sm + sel_idx * stride_sn))
        numerator += orig_score  # float32 accumulation
    # Compute weight after normalization
    denom = numerator + eps
    weight = routed_scale * (numerator / denom)
    # Store weight for this token at position k (weight is same for all k? If so, write once; otherwise write in loop)
    # We write weight for each k position
    for k in range(0, 8):
        tl.store(OUT_WEIGHT_ptr + (m * out_stride_im + k * out_stride_in), weight)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype and device
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Inputs must be on CUDA"
        hidden = hidden_states.contiguous().to(torch.float32)   # [M, K]
        weight_w = weight.contiguous().to(torch.float32)        # [N, K]
        bias_e = expert_bias.contiguous().to(torch.float32)     # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight_w.shape[0]  # 256
        EXPERTS_PER_GROUP = 32
        GROUPS = 8

        # 1) Compute logits: [M,N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid_linear = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        linear_bias_kernel[grid_linear](
            hidden, weight_w, bias_e, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight_w.stride(0), weight_w.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 2) scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid_sigmoid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        sigmoid_kernel[grid_sigmoid](
            logits, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=64, BLOCK_N=64
        )
        # Add expert bias
        scores = scores + bias_e[None, :]

        # 3) Group scores [M,8]
        group_scores = torch.empty((M, GROUPS), dtype=torch.float32, device=hidden.device)
        grid_group = (M, GROUPS)
        compute_group_scores_kernel[grid_group](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXPERTS_PER_GROUP, GROUPS
        )

        # 4) Select top-4 groups [M,4] int32
        selected_groups = torch.empty((M, 4), dtype=torch.int32, device=hidden.device)
        grid_select_groups = (M, 1)  # one program per m
        select_top4_groups_kernel[grid_select_groups](
            group_scores, selected_groups,
            M, GROUPS,
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1)
        )

        # 5) Mask scores with selected groups -> [M,N]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid_mask = (M, 1)
        mask_scores_with_groups_kernel[grid_mask](
            scores, selected_groups, masked_scores,
            M, N, GROUPS, EXPERTS_PER_GROUP,
            scores.stride(0), scores.stride(1),
            masked_scores.stride(0), masked_scores.stride(1)
        )

        # 6) Final top-8 selection and compute weights [M,8] float32
        scores_copy = scores.clone()  # preserve original scores for normalization
        out_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden.device)
        out_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid_final = (M, 1)
        final_top8_with_weight_and_normalize_kernel[grid_final](
            masked_scores, scores_copy, out_idx, out_weight,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            out_idx.stride(0), out_idx.stride(1),
            routed_scaling_factor, 1e-20
        )

        # Return (topk_idx, topk_weight) matching original signature:
        # - topk_idx is the indices chosen (int64 in original), we have int32 here; cast to int64
        # - topk_weight is float32
        topk_idx = out_idx.to(torch.int64)
        topk_weight = out_weight

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
