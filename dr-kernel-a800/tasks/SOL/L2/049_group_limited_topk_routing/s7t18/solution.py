import torch
import torch.nn as nn
import triton
import triton.language as tl


# Kernel 1: Linear + bias: OUT[M, N] = A[M, K] @ W[N, K]^T + bias[N]
@triton.jit
def linear_bias_kernel(
    A_ptr,      # *fp32, shape [M, K]
    W_ptr,      # *fp32, shape [N, K]
    BIAS_ptr,   # *fp32, shape [N]
    OUT_ptr,    # *fp32, shape [M, N]
    M: tl.constexpr,        # num_tokens
    N: tl.constexpr,        # num_experts (256)
    K: tl.constexpr,        # hidden_dim
    stride_am, stride_ak,   # strides for A
    stride_wn, stride_wk,   # strides for W
    stride_om, stride_on,   # strides for OUT
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile over tokens
    pid_n = tl.program_id(1)  # tile over experts
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        # Load A tile [BLOCK_M, BLOCK_K]
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0)
        # Load W^T tile [BLOCK_N, BLOCK_K]
        W_tile_ptr = W_ptr + (offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk)
        W_mask = (offs_n[:, None] < N) & (k_ids[None, :] < K)
        W_tile = tl.load(W_tile_ptr, mask=W_mask, other=0.0)
        acc += tl.dot(A_tile, W_tile)

    # Add bias per expert
    bias_vals = tl.load(BIAS_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc += bias_vals[None, :]

    # Store
    OUT_ptr_tile = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    OUT_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(OUT_ptr_tile, acc, mask=OUT_mask)


# Kernel 2: Elementwise sigmoid: OUT[M, N] = sigmoid(IN[M, N]) + bias[N]
@triton.jit
def sigmoid_kernel(
    IN_ptr,     # *fp32, [M, N]
    BIAS_ptr,   # *fp32, [N]
    OUT_ptr,    # *fp32, [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_im, stride_in,
    stride_bm, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    in_ptr = IN_ptr + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
    bias_ptr = BIAS_ptr + offs_n
    out_ptr = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(in_ptr, mask=mask, other=0.0)
    b = tl.load(bias_ptr, mask=(offs_n < N), other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    y = y + b[None, :]
    tl.store(out_ptr, y, mask=mask)


# Kernel 3: Compute group scores: sum of top-2 per group. Input scores [M, N], Output group_scores[M, 8].
@triton.jit
def compute_group_scores_kernel(
    SCORES_ptr,     # *fp32, [M, N]
    GROUPS_ptr,     # *fp32, [M, 8]
    M: tl.constexpr,
    N: tl.constexpr,            # num_experts
    n_group: tl.constexpr,      # 8
    experts_per_group: tl.constexpr,  # 32
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    pid_m = tl.program_id(0)  # token row
    pid_g = tl.program_id(1)  # group id in [0, 7]
    group_start = pid_g * experts_per_group
    top1 = -float('inf')
    idx1 = -1
    top2 = -float('inf')
    idx2 = -1

    for j in range(0, experts_per_group):
        idx = group_start + j
        val = tl.load(SCORES_ptr + pid_m * stride_sm + idx * stride_sn)
        if val > top1:
            top2 = top1
            idx2 = idx1
            top1 = val
            idx1 = idx
        elif val > top2:
            top2 = val
            idx2 = idx

    total = top1 + top2
    tl.store(GROUPS_ptr + pid_m * stride_gm + pid_g * stride_gn, total)


# Kernel 4: Select top-4 groups per token. Output int32 [M, 4]
@triton.jit
def select_top4_groups_kernel(
    GROUPS_ptr,         # *fp32, [M, 8]
    SELECTED_ptr,       # *int32, [M, 4]
    M: tl.constexpr,
    n_group: tl.constexpr,  # 8
    stride_gm, stride_gn,
    stride_sm, stride_sn,
):
    pid_m = tl.program_id(0)
    for t in range(0, 4):
        max_val = -float('inf')
        max_idx = -1
        for g in range(0, n_group):
            val = tl.load(GROUPS_ptr + pid_m * stride_gm + g * stride_gn)
            if val > max_val:
                max_val = val
                max_idx = g
        tl.store(SELECTED_ptr + pid_m * stride_sm + t * stride_sn, max_idx)
        # Invalidate selected group for next iteration
        tl.store(GROUPS_ptr + pid_m * stride_gm + max_idx * stride_gn, -float('inf'))


# Kernel 5: Mask scores: set non-selected groups' scores to -inf based on selected_groups[M,4].
@triton.jit
def mask_scores_with_groups_kernel(
    SCORES_ptr,         # *fp32, [M, N]
    SELECTED_groups_ptr,# *int32, [M, 4]
    M: tl.constexpr,
    N: tl.constexpr,            # num_experts
    n_group: tl.constexpr,      # 8
    experts_per_group: tl.constexpr,  # 32
    stride_sm, stride_sn,
    stride_sgm, stride_sgn,   # strides for selected groups
):
    pid_m = tl.program_id(0)
    for g in range(0, n_group):
        # Check if group g was selected by any of the 4 slots
        found = 0
        for t in range(0, 4):
            sel = tl.load(SELECTED_groups_ptr + pid_m * stride_sgm + t * stride_sgn)
            if sel == g:
                found = 1
                break
        if found == 0:
            # Set all scores in this group to -inf
            for j in range(0, experts_per_group):
                idx = g * experts_per_group + j
                tl.store(SCORES_ptr + pid_m * stride_sm + idx * stride_sn, -float('inf'))


# Kernel 6: Final top-8 selection with weight normalization. Input scores [M, N] (already masked), Output indices [M, 8] and weights [M, 8].
@triton.jit
def final_top8_with_weight_and_normalize_kernel(
    SCORES_ptr,            # *fp32, [M, N] masked
    SELECTED_IDX_ptr,      # *int32, [M, 8]
    WEIGHTS_ptr,           # *fp32,  [M, 8]
    M: tl.constexpr,
    N: tl.constexpr,                 # num_experts
    routed_scaling_factor: tl.constexpr,   # float32 scalar
    stride_sm, stride_sn,
    stride_ism, stride_issn,
    stride_wm, stride_wn,
):
    pid_m = tl.program_id(0)
    denom = 1e-20
    # iterative elimination to select top-8
    for t in range(0, 8):
        max_val = -float('inf')
        max_idx = -1
        for j in range(0, N):
            val = tl.load(SCORES_ptr + pid_m * stride_sm + j * stride_sn)
            if val > max_val:
                max_val = val
                max_idx = j
        # record index
        tl.store(SELECTED_IDX_ptr + pid_m * stride_ism + t * stride_issn, max_idx)
        # record weight as score * scaling / denom (denom is small epsilon)
        weight = (max_val * routed_scaling_factor) / denom
        tl.store(WEIGHTS_ptr + pid_m * stride_wm + t * stride_wn, weight)
        # invalidate this expert
        tl.store(SCORES_ptr + pid_m * stride_sm + max_idx * stride_sn, -float('inf'))


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Prepare shapes
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors"
        M = hidden_states.shape[0]
        N = 256
        K = hidden_states.shape[1]
        device = hidden_states.device

        # Ensure dtype float32 and contiguous
        A = hidden_states.contiguous().to(torch.float32)          # [M, K]
        W = weight.contiguous().to(torch.float32)                # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)        # [N]

        # 1) Linear + bias
        logits = torch.empty((M, N), device=device, dtype=torch.float32)
        grid1 = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        linear_bias_kernel[grid1](
            A, W, bias, logits,
            M, N, K,
            A.stride(0), A.stride(1),
            W.stride(0), W.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # 2) Sigmoid + bias
        scores = torch.empty((M, N), device=device, dtype=torch.float32)
        grid2 = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        sigmoid_kernel[grid2](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            bias.stride(0), bias.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=64, BLOCK_N=64,
        )

        # 3) Compute group_scores [M, 8]
        group_scores = torch.empty((M, 8), device=device, dtype=torch.float32)
        grid3 = (M, 8)
        compute_group_scores_kernel[grid3](
            scores, group_scores,
            M, N, 8, 32,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
        )

        # 4) Select top-4 groups per token [M, 4] int32
        selected_groups = torch.empty((M, 4), device=device, dtype=torch.int32)
        grid4 = (M, )
        select_top4_groups_kernel[grid4](
            group_scores, selected_groups,
            M, 8,
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
        )

        # 5) Mask scores: set non-selected groups' scores to -inf
        masked_scores = scores.clone()  # keep original scores for normalization (final stage)
        grid5 = (M, )
        mask_scores_with_groups_kernel[grid5](
            masked_scores, selected_groups,
            M, N, 8, 32,
            masked_scores.stride(0), masked_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
        )

        # 6) Final top-8 selection and normalization -> output indices and weights
        out_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        out_weight = torch.empty((M, 8), device=device, dtype=torch.float32)
        grid6 = (M, )
        final_top8_with_weight_and_normalize_kernel[grid6](
            masked_scores, out_idx, out_weight,
            M, N, routed_scaling_factor,
            masked_scores.stride(0), masked_scores.stride(1),
            out_idx.stride(0), out_idx.stride(1),
            out_weight.stride(0), out_weight.stride(1),
        )

        # Return topk_idx (int64) and topk_weight (float32)
        return out_idx.to(torch.int64), out_weight


def run(*args):
    return ModelNew()(*args)
