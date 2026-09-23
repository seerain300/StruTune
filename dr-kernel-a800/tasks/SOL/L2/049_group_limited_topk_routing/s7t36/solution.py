import torch
import torch.nn as nn
import triton
import triton.language as tl


# Kernel 1: logits = hidden_states @ weight^T + expert_bias
# A: [M, K] row-major, W: [N, K] row-major, bias: [N]
@triton.jit
def linear_bias_kernel(
    A_ptr,        # *fp32
    W_ptr,        # *fp32
    BIAS_ptr,     # *fp32
    OUT_ptr,      # *fp32 logits: [M, N]
    M: tl.constexpr,   # num_tokens
    N: tl.constexpr,   # num_experts = 256
    K: tl.constexpr,   # hidden_dim
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m * BLOCK_M
    offs_m = m + tl.arange(0, BLOCK_M)
    # pointers for output row
    out_row_ptr = OUT_ptr + offs_m * stride_om  # vector of M pointers

    # accumulator
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # loop over N in chunks
    for n0 in range(0, N, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)  # vector of N indices
        # pointer to weight rows
        w_ptrs = W_ptr + n_idx * stride_wn  # [BLOCK_N] pointers
        # compute A[m, :] dot W[n, :]
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            # load A rows [BLOCK_M, BLOCK_K]
            a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak)
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
            # load W rows [BLOCK_N, BLOCK_K]
            w_ptrs_k = W_ptr + (n_idx[:, None] * stride_wn + k_idx[None, :] * stride_wk)
            w = tl.load(w_ptrs_k, mask=(n_idx[:, None] < N) & (k_idx[None, :] < K), other=0.0)
            # acc += sum over K of a * w^T
            acc += tl.sum(a * w, axis=1)
        # add bias
        b = tl.load(BIAS_ptr + n_idx, mask=n_idx < N, other=0.0)  # [BLOCK_N]
        acc += b[None, :]

    # store
    out_ptrs = OUT_ptr + offs_m * stride_om
    tl.store(out_ptrs, acc, mask=offs_m < M)


# Kernel 2: scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_bias_kernel(
    LOGITS_ptr,   # *fp32
    BIAS_ptr,     # *fp32
    OUT_ptr,      # *fp32 scores: [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_lm, stride_ln,
    stride_bm,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m * BLOCK_M
    offs_m = m + tl.arange(0, BLOCK_M)
    out_row_ptr = OUT_ptr + offs_m * stride_om

    for n0 in range(0, N, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)
        logits_ptrs = LOGITS_ptr + (offs_m[:, None] * stride_lm + n_idx[None, :] * stride_ln)
        logits = tl.load(logits_ptrs, mask=(offs_m[:, None] < M) & (n_idx[None, :] < N), other=0.0)
        # sigmoid
        scores = 1.0 / (1.0 + tl.exp(-logits))
        bias = tl.load(BIAS_ptr + n_idx, mask=(n_idx < N), other=0.0)  # [BLOCK_N]
        scores = scores + bias[None, :]
        tl.store(OUT_ptr + (offs_m[:, None] * stride_om + n_idx[None, :] * stride_on), scores,
                 mask=(offs_m[:, None] < M) & (n_idx[None, :] < N))


# Kernel 3: compute group scores (sum of top-2 per group)
@triton.jit
def compute_group_scores_kernel(
    SCORES_ptr,      # *fp32 [M, N]
    GROUP_OUT_ptr,   # *fp32 [M, 8]
    M: tl.constexpr,
    N: tl.constexpr,            # 256
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    # Each program handles one token m; loops over groups
    pid_m = tl.program_id(0)
    m = pid_m
    # Find top2 per group g in [0..7]
    for g in range(0, 8):
        # Base offset for this group
        base = g * EXPERTS_PER_GROUP
        # vector of expert indices for this group
        e = base + tl.arange(0, EXPERTS_PER_GROUP)  # [32]
        # Load scores [32]
        scores_vec = tl.load(SCORES_ptr + (m * stride_sm + e * stride_sn), mask=e < N, other=-float('inf'))
        # Compute top-2
        # Sort ascending trick: take max and second max via pairwise max/min without built-in topk
        # We'll compute iteratively:
        # First max
        m1 = tl.max(scores_vec, axis=0)
        mask_m1 = scores_vec == m1
        # Set that position to -inf and take max again
        scores_vec = tl.where(mask_m1, -float('inf'), scores_vec)
        m2 = tl.max(scores_vec, axis=0)
        group_score = m1 + m2
        # Store
        tl.store(GROUP_OUT_ptr + (m * stride_gm + g * stride_gn), group_score)


# Kernel 4: select 4 groups per token via iterative elimination (sorted=False)
@triton.jit
def select_top4_groups_kernel(
    GROUP_SCORES_ptr,   # *fp32 [M,8]
    SELECTED_ptr,       # *int32 [M,4]
    M: tl.constexpr,
    GROUPS: tl.constexpr,  # 8
    stride_gm, stride_gn,
    stride_sm, stride_sn,
):
    pid_m = tl.program_id(0)
    m = pid_m
    used = tl.zeros((GROUPS,), dtype=tl.int1)
    for k in range(0, 4):
        best_score = -float('inf')
        sel = 0
        for g in range(0, GROUPS):
            score = tl.load(GROUP_SCORES_ptr + (m * stride_gm + g * stride_gn))
            if (not used[g]) and (score > best_score):
                best_score = score
                sel = g
        used[sel] = True
        tl.store(SELECTED_ptr + (m * stride_sm + k * stride_sn), sel)


# Kernel 5: mask scores: for selected groups only keep scores, else set to -inf
@triton.jit
def mask_scores_with_groups_kernel(
    SCORES_ptr,            # *fp32 [M,N]
    SELECTED_GROUPS_ptr,   # *int32 [M,4]
    MASKED_OUT_ptr,        # *fp32 [M,N]
    M: tl.constexpr,
    N: tl.constexpr,           # 256
    stride_sm, stride_sn,
    stride_sgm, stride_sgn,
    stride_mom, stride_mon,
):
    pid_m = tl.program_id(0)
    m = pid_m
    for g in range(0, 4):
        sel = tl.load(SELECTED_GROUPS_ptr + (m * stride_sgm + g * stride_sgn))  # int32
        # For all e in this group: if e not equal sel, set to -inf
        base = sel * 32
        for e in range(0, 32):
            idx = base + e
            # Load current score
            score = tl.load(SCORES_ptr + (m * stride_sm + idx * stride_sn))
            is_selected = (sel == sel)  # always true; keep selected
            new_score = tl.where((idx == sel), score, -float('inf'))
            tl.store(MASKED_OUT_ptr + (m * stride_mom + idx * stride_mon), new_score)


# Kernel 6: final selection of top-8 from masked scores and compute weight
@triton.jit
def final_top8_with_weight_and_normalize_kernel(
    LOGITS_ptr,        # *fp32 [M,N]
    MASKED_SCORES_ptr, # *fp32 [M,N]
    OUT_IDX_ptr,       # *int32 [M,8]
    OUT_WEIGHT_ptr,    # *fp32  [M,8]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_lm, stride_ln,
    stride_msm, stride_msn,
    stride_i0m, stride_i0n,
    stride_w0m, stride_w0n,
    routed_scaling_factor: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m
    numerator = 0.0
    # Iteratively select top-8 via elimination
    for t in range(0, 8):
        curr_max = -float('inf')
        sel_idx = 0
        for e in range(0, N):
            score = tl.load(MASKED_SCORES_ptr + (m * stride_msm + e * stride_msn))
            if score > curr_max:
                curr_max = score
                sel_idx = e
        # store index
        tl.store(OUT_IDX_ptr + (m * stride_i0m + t * stride_i0n), sel_idx)
        # accumulate numerator from original logits
        logits_val = tl.load(LOGITS_ptr + (m * stride_lm + sel_idx * stride_ln))
        numerator += logits_val
    # compute weight
    denom = numerator + eps
    weight_val = routed_scaling_factor * (numerator / denom)
    # store weight for all 8 positions (same value)
    for t in range(0, 8):
        tl.store(OUT_WEIGHT_ptr + (m * stride_w0m + t * stride_w0n), weight_val)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Enforce dtype and device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]  # must be 256
        assert N == 256, "num_experts must be 256"
        device = hidden_states.device
        assert device.type == 'cuda', "Triton kernels require CUDA device"

        # Ensure float32
        A = hidden_states.to(torch.float32)
        W = weight.to(torch.float32)
        bias = expert_bias.to(torch.float32)

        # Allocate buffers
        logits = torch.empty((M, N), dtype=torch.float32, device=device)          # [M, N]
        scores = torch.empty((M, N), dtype=torch.float32, device=device)          # [M, N]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)    # [M, 8]
        selected_groups = torch.empty((M, 4), dtype=torch.int32, device=device)   # [M, 4]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=device)   # [M, N]
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=device)          # [M, 8]
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)     # [M, 8]

        # Launch kernels
        # 1) logits
        BLOCK_M1 = 128
        BLOCK_N1 = 64
        BLOCK_K1 = 64
        grid1 = (triton.cdiv(M, BLOCK_M1),)
        linear_bias_kernel[grid1](
            A, W, bias, logits,
            M, N, K,
            A.stride(0), A.stride(1),
            W.stride(0), W.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
        )

        # 2) scores = sigmoid(logits) + bias
        BLOCK_M2 = 128
        BLOCK_N2 = 128
        grid2 = (triton.cdiv(M, BLOCK_M2),)
        sigmoid_bias_kernel[grid2](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            bias.stride(0),
            scores.stride(0), scores.stride(1),
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
        )

        # 3) compute group scores
        BLOCK_SM = 1
        BLOCK_SN = 1
        grid3 = (M,)
        compute_group_scores_kernel[grid3](
            scores, group_scores,
            M, N, 32,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
        )

        # 4) select top-4 groups per token
        grid4 = (M,)
        select_top4_groups_kernel[grid4](
            group_scores, selected_groups,
            M, 8,
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
        )

        # 5) mask scores: non-selected groups set to -inf
        grid5 = (M,)
        mask_scores_with_groups_kernel[grid5](
            scores, selected_groups, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
        )

        # 6) final top-8 from masked scores and compute weight
        BLOCK_M6 = 1
        grid6 = (M,)
        final_top8_with_weight_and_normalize_kernel[grid6](
            logits, masked_scores, topk_idx, topk_weight,
            M, N,
            logits.stride(0), logits.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_scaling_factor, 1e-20,
        )

        # Return indices (int64) and weights (float32)
        topk_idx_out = topk_idx.to(torch.int64)
        return topk_idx_out, topk_weight


# Example test harness can call ModelNew as:
# model = ModelNew().cuda()
# hidden = torch.randn(2048, 128, device='cuda', dtype=torch.float32)
# weight = torch.randn(256, 128, device='cuda', dtype=torch.float32)
# bias = torch.randn(256, device='cuda', dtype=torch.float32)
# routed_factor = 1.2
# idx, weights = model(hidden, weight, bias, routed_factor)


def run(*args):
    return ModelNew()(*args)
