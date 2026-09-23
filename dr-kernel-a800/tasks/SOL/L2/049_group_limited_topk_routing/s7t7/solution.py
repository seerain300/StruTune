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


# 2) Triton kernel: elementwise sigmoid on logits (IN: [M, N], OUT: [M, N])
@triton.jit
def sigmoid_kernel(
    IN_ptr,     # *fp32 logits
    OUT_ptr,    # *fp32 sigmoid
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

    in_ptr = IN_ptr + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
    out_ptr = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(in_ptr, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr, y, mask=mask)


# 3) Triton kernel: compute group scores (sum of top-2 per group) from scores [M, N]
# Group: 8 groups of 32 experts, so EXPERTS_PER_GROUP = 32
@triton.jit
def compute_group_scores_kernel(
    IN_ptr,     # *fp32 scores [M, N]
    OUT_ptr,    # *fp32 group_scores [M, 8]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_is_m, stride_is_n,
    stride_os_m, stride_os_e,
    EXPERTS_PER_GROUP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)  # e = 0..7
    offs_m = pid_m * 64 + tl.arange(0, 64)  # tile over tokens
    e = pid_e

    # Initialize top1 and top2
    top1 = tl.full((64,), -1e20, dtype=tl.float32)
    top2 = tl.full((64,), -1e20, dtype=tl.float32)

    # Loop over 32 experts in the group
    for j in tl.static_range(0, EXPERTS_PER_GROUP):
        col = e * EXPERTS_PER_GROUP + j
        # scores per token row: IN[m, col]
        in_ptr_j = IN_ptr + offs_m * stride_is_m + col * stride_is_n
        mask = offs_m < M
        v = tl.load(in_ptr_j, mask=mask, other=-1e20)
        better = v > top1
        # Update top2 where v is better than top1
        top2 = tl.where(better, top1, top2)
        # Update top1
        top1 = tl.where(better, v, top1)

    # Sum of top2
    top2_sum = top1 + top2  # per token
    out_ptr = OUT_ptr + offs_m * stride_os_m + pid_e * stride_os_e
    tl.store(out_ptr, top2_sum, mask=(offs_m < M))


# 4) Triton kernel: select top-4 groups per token (iterative elimination)
# Input: group_scores [M, 8], Output: selected_groups [M, 4] (int32 indices 0..7)
@triton.jit
def select_top4_groups_kernel(
    IN_ptr,     # *fp32 group_scores [M, 8]
    OUT_ptr,    # *int32 selected_groups [M, 4]
    M: tl.constexpr,
    E: tl.constexpr,  # num_groups = 8
    stride_is_m, stride_is_e,
    stride_os_m, stride_os_e,
):
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)  # e = 0..3
    offs_m = pid_m * 64 + tl.arange(0, 64)

    # For each token row, keep a list of selected flags
    selected = tl.zeros((64,), dtype=tl.int32)

    # Iteratively find best among not-selected groups
    # We do 4 iterations, selecting one each time and marking it as selected.
    # This requires maintaining a mask for not-selected groups; Triton supports int masks.
    for t in tl.static_range(0, 4):
        best_val = tl.full((64,), -1e20, dtype=tl.float32)
        best_idx = tl.full((64,), 0, dtype=tl.int32)
        for i in tl.static_range(0, E):
            if selected[i] == 0:  # not selected
                in_ptr_i = IN_ptr + offs_m * stride_is_m + i * stride_is_e
                val = tl.load(in_ptr_i, mask=(offs_m < M), other=-1e20)
                better = val > best_val
                best_val = tl.where(better, val, best_val)
                best_idx = tl.where(better, i, best_idx)
        # Mark selected
        selected = tl.where(best_idx == tl.arange(0, E)[None, :], 1, selected)  # update selected based on best_idx
        # Store best_idx per token
        out_ptr = OUT_ptr + offs_m * stride_os_m + t * stride_os_e
        tl.store(out_ptr, best_idx, mask=(offs_m < M))


# 5) Triton kernel: compute masked_scores by scanning selected_groups [M, 4] and set corresponding 4 groups to -inf
# IN_ptr scores [M, N], selected_groups [M, 4], OUT_ptr masked_scores [M, N]
@triton.jit
def mask_scores_kernel(
    IN_ptr,        # *fp32 scores [M, N]
    SG_ptr,        # *int32 selected_groups [M, 4]
    OUT_ptr,       # *fp32 masked_scores [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,  # num_groups = 8
    S: tl.constexpr,  # experts_per_group = 32
    stride_in_m, stride_in_n,
    stride_sg_m, stride_sg_e,
    stride_out_m, stride_out_n,
):
    # We implement masking by scanning each token row and zero-out 4 groups. We do this per token-row tiles.
    # This kernel loops over 4 selections per token and writes to OUT.
    pid_m = tl.program_id(0)
    offs_m = pid_m * 64 + tl.arange(0, 64)

    for t in tl.static_range(0, 4):
        sg_ptr = SG_ptr + offs_m * stride_sg_m + t * stride_sg_e
        group_idx = tl.load(sg_ptr, mask=(offs_m < M), other=0).to(tl.int32)
        start = group_idx * S
        # Loop over S=32 and write -inf to OUT for those columns
        for j in tl.static_range(0, S):
            col = start + j
            in_ptr = IN_ptr + offs_m * stride_in_m + col * stride_in_n
            mask = (offs_m < M) & (col < N)
            # load current score
            val = tl.load(in_ptr, mask=mask, other=0.0)
            # write -inf
            out_ptr = OUT_ptr + offs_m * stride_out_m + col * stride_out_n
            tl.store(out_ptr, tl.full((64,), -1e20, dtype=tl.float32), mask=mask)


# 6) Triton kernel: final top-8 selection and normalization from masked_scores
# We perform iterative elimination to select top-8 and gather selected logits for normalization. Then compute weight and index.
@triton.jit
def final_top8_with_weight_and_normalize_kernel(
    IN_ptr,        # *fp32 masked_scores [M, N]
    OUT_idx_ptr,   # *int32 [M, 8]
    OUT_weight_ptr,# *fp32 [M, 8]
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,  # hidden_dim (but we only need N=experts here for scores)
    eps: tl.constexpr,
    scale: tl.constexpr,
    stride_in_m, stride_in_n,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * 64 + tl.arange(0, 64)

    selected_idx = tl.full((64, 8), 0, dtype=tl.int32)  # will store indices 0..N-1
    selected_val = tl.full((64, 8), -1e20, dtype=tl.float32)

    # Iterative selection for top-8
    for t in tl.static_range(0, 8):
        best_val = tl.full((64,), -1e20, dtype=tl.float32)
        best_idx = tl.full((64,), 0, dtype=tl.int32)
        # scan all columns (experts)
        for j in tl.static_range(0, N):
            in_ptr_j = IN_ptr + offs_m * stride_in_m + j * stride_in_n
            val = tl.load(in_ptr_j, mask=(offs_m < M), other=-1e20)
            better = val > best_val
            best_val = tl.where(better, val, best_val)
            best_idx = tl.where(better, j, best_idx)

        # store best_idx and best_val for this token row
        selected_idx[:, t] = best_idx
        selected_val[:, t] = best_val

        # mark those selected columns as -inf to exclude in next iterations
        # we update IN_ptr logically by writing to OUT temporarily, but here we just keep track. Simpler approach:
        # Since Triton doesn't allow mutating IN, we recompute each iteration from original. So we won't write back here.

    # Now compute normalized weights using selected_val (sum per row), then write weights and indices
    # For each t, selected_val[:, t] is the selected value. We need sum across t.
    for t in tl.static_range(0, 8):
        vals = selected_val[:, t]
        denom = tl.sum(vals, axis=0) + eps
        # Store indices
        idx_out_ptr = OUT_idx_ptr + offs_m * stride_om + t * stride_on
        tl.store(idx_out_ptr, selected_idx[:, t], mask=(offs_m < M))
        # Store weights: (selected_val[:, t] / denom) * scale
        weight_vals = (selected_val[:, t] / denom) * scale
        out_weight_ptr = OUT_weight_ptr + offs_m * stride_om + t * stride_on
        tl.store(out_weight_ptr, weight_vals, mask=(offs_m < M))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants per original problem
        self.num_experts = 256
        self.num_groups = 8
        self.experts_per_group = self.num_experts // self.num_groups  # 32
        self.topk_group = 4
        self.top_k = 8

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        hidden_states: [num_tokens, hidden_dim], CUDA, float16/float32
        weight: [num_experts, hidden_dim], CUDA, float16/float32 (num_experts=256)
        expert_bias: [num_experts], CUDA, float32
        routed_scaling_factor: float
        Returns:
        - topk_idx: [num_tokens, 8] int32
        - topk_weight: [num_tokens, 8] float32
        """
        # Ensure CUDA and float32 for compute
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = self.num_experts

        # 1) Compute logits = hidden_states @ weight^T + expert_bias
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        stride_am, stride_ak = hidden_states.stride(0), hidden_states.stride(1)
        stride_wn, stride_wk = weight.stride(0), weight.stride(1)
        stride_om, stride_on = logits.stride(0), logits.stride(1)

        # Launch linear_bias_kernel
        grid_linear = (M // 64, N // 64)  # tile over tokens and experts; adjust blocks below
        linear_bias_kernel[grid_linear](
            hidden_states, weight, expert_bias, logits,
            M, N, K,
            stride_am, stride_ak,
            stride_wn, stride_wk,
            stride_om, stride_on,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # 2) Sigmoid of logits -> scores
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        stride_sm, stride_sn = logits.stride(0), logits.stride(1)
        stride_ssig_m, stride_ssig_n = scores.stride(0), scores.stride(1)

        sigmoid_kernel[(M // 64, N // 64)](
            logits, scores,
            M, N,
            stride_sm, stride_sn,
            stride_ssig_m, stride_ssig_n,
            BLOCK_M=64, BLOCK_N=64
        )

        # 3) Compute group_scores: sum of top-2 per group (from scores)
        group_scores = torch.empty((M, self.num_groups), dtype=torch.float32, device=device)
        stride_gs_m, stride_gs_e = scores.stride(0), scores.stride(1)
        stride_gm, stride_ge = group_scores.stride(0), group_scores.stride(1)

        compute_group_scores_kernel[(M, self.num_groups)](
            scores, group_scores,
            M, self.num_experts,
            stride_gs_m, stride_gs_e,
            stride_gm, stride_ge,
            EXPERTS_PER_GROUP=self.experts_per_group
        )

        # 4) Select top-4 groups per token
        selected_groups = torch.empty((M, self.topk_group), dtype=torch.int32, device=device)
        stride_sg_m, stride_sg_e = group_scores.stride(0), group_scores.stride(1)
        stride_sel_m, stride_sel_e = selected_groups.stride(0), selected_groups.stride(1)

        select_top4_groups_kernel[(M, self.topk_group)](
            group_scores, selected_groups,
            M, self.num_groups,
            stride_sg_m, stride_sg_e,
            stride_sel_m, stride_sel_e
        )

        # 5) Mask out selected groups: write -inf to masked_scores for selected groups per token
        masked_scores = torch.empty((M, self.num_experts), dtype=torch.float32, device=device)
        stride_in_m, stride_in_n = scores.stride(0), scores.stride(1)
        stride_sg_m2, stride_sg_e2 = selected_groups.stride(0), selected_groups.stride(1)
        stride_out_m, stride_out_n = masked_scores.stride(0), masked_scores.stride(1)

        mask_scores_kernel[(M,)](
            scores, selected_groups, masked_scores,
            M, self.num_experts,
            self.num_groups, self.experts_per_group,
            stride_in_m, stride_in_n,
            stride_sg_m2, stride_sg_e2,
            stride_out_m, stride_out_n
        )

        # 6) Final top-8 selection and normalization from masked_scores, produce topk_idx and topk_weight
        out_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=device)
        out_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=device)
        stride_om_idx, stride_on_idx = out_idx.stride(0), out_idx.stride(1)
        stride_om_w, stride_on_w = out_weight.stride(0), out_weight.stride(1)

        final_top8_with_weight_and_normalize_kernel[(M,)](
            masked_scores, out_idx, out_weight,
            M, self.num_experts,
            0, 1e-20, routed_scaling_factor,
            stride_in_m, stride_in_n,
            stride_om_idx, stride_on_idx
        )

        return out_idx, out_weight


# For consistency with the original example, we define a run function and Model as in the prompt.
@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    # Use ModelNew which does all computation via Triton kernels
    return ModelNew().forward(hidden_states, weight, expert_bias, routed_scaling_factor)

class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
