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

    IN_tile_ptr = IN_ptr + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
    OUT_tile_ptr = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(IN_tile_ptr, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(OUT_tile_ptr, y, mask=mask)


# 3) Triton kernel: compute group_scores [M, n_group=8] as sum of top-2 per group
@triton.jit
def compute_group_scores_kernel(
    SCORES_ptr,   # *fp32 [M, N]
    GROUP_OUT_ptr,  # *fp32 [M, 8]
    M: tl.constexpr,
    N: tl.constexpr,  # num_experts=256
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    n_group: tl.constexpr = 8,
    experts_per_group: tl.constexpr = 32,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    start_exp = pid_g * experts_per_group

    # Load 32 scores for this group into a vector
    idxs = start_exp + tl.arange(0, experts_per_group)
    mask = idxs < N
    vals = tl.load(SCORES_ptr + pid_m * stride_sm + idxs * stride_sn, mask=mask, other=-1.0e30)

    # Find top-2: simple bubble sort descending
    size = experts_per_group  # 32
    for i in range(size):
        for j in range(size - 1, i, -1):
            a = vals[j - 1]
            b = vals[j]
            swap = a < b
            # swap elements
            vals = tl.where(swap, [b, a], [a, b])

    top2_sum = vals[0] + vals[1]
    tl.store(GROUP_OUT_ptr + pid_m * stride_gm + pid_g * stride_gn, top2_sum)


# 4) Triton kernel: select top-4 groups per token (iterative elimination)
@triton.jit
def select_top4_groups_kernel(
    GROUP_scores_ptr,  # *fp32 [M, n_group]
    SELECTED_ptr,      # *int32 [M, 4]
    M: tl.constexpr,
    n_group: tl.constexpr,  # 8
    stride_gs_m, stride_gs_n,
    stride_sel_m, stride_sel_n,
):
    pid_m = tl.program_id(0)
    for i in range(4):
        max_val = -1.0e30
        max_idx = 0
        for g in range(n_group):
            val = tl.load(GROUP_scores_ptr + pid_m * stride_gs_m + g * stride_gs_n)
            cond = val > max_val
            max_val = tl.where(cond, val, max_val)
            max_idx = tl.where(cond, g, max_idx)
        tl.store(SELECTED_ptr + pid_m * stride_sel_m + i * stride_sel_n, max_idx)
        # set that group's score to -inf for future iterations
        tl.store(GROUP_scores_ptr + pid_m * stride_gs_m + max_idx * stride_gs_n, -1.0e30)


# 5) Triton kernel: mask scores with selected groups per token (set unselected groups to -inf)
@triton.jit
def mask_scores_with_groups_kernel(
    SELECTED_ptr,     # *int32 [M, 4]
    SCORES_ptr,       # *fp32 [M, N]
    MASKED_ptr,       # *fp32 [M, N]
    M: tl.constexpr,
    N: tl.constexpr,  # 256
    stride_sel_m, stride_sel_n,
    stride_sc_m, stride_sc_n,
    stride_ms_m, stride_ms_n,
):
    pid_m = tl.program_id(0)
    # iterate over 8 groups
    for g in range(8):
        sel = 0
        # check if this group is selected by any of the 4 selected groups
        for i in range(4):
            idx_i = tl.load(SELECTED_ptr + pid_m * stride_sel_m + i * stride_sel_n)
            sel = sel | (g == idx_i)
        start_exp = g * 32
        if sel == 0:
            # unselected group: set all 32 experts to -inf
            for e in range(32):
                idx = start_exp + e
                val = tl.load(SCORES_ptr + pid_m * stride_sc_m + idx * stride_sc_n)
                tl.store(MASKED_ptr + pid_m * stride_ms_m + idx * stride_ms_n, -1.0e30)
        else:
            # selected group: copy scores
            for e in range(32):
                idx = start_exp + e
                val = tl.load(SCORES_ptr + pid_m * stride_sc_m + idx * stride_sc_n)
                tl.store(MASKED_ptr + pid_m * stride_ms_m + idx * stride_ms_n, val)


# 6) Triton kernel: final top-8 from masked scores, normalize by sum of original scores, apply scaling
@triton.jit
def final_top8_with_weight_and_normalize_kernel(
    MASKED_ptr,          # *fp32 [M, N]
    SCORES_COPY_ptr,     # *fp32 [M, N] (original scores)
    OUT_idx_ptr,         # *int32 [M, 8]
    OUT_w_ptr,           # *fp32 [M, 8]
    M: tl.constexpr,
    N: tl.constexpr,     # 256
    routed_scaling_factor: tl.constexpr,  # float
    stride_ms_m, stride_ms_n,
    stride_sc_m, stride_sc_n,
    stride_i_m, stride_i_n,
    stride_w_m, stride_w_n,
):
    pid_m = tl.program_id(0)
    # iterative elimination for top-8 indices
    for i in range(8):
        max_val = -1.0e30
        max_idx = 0
        for n in range(N):
            val = tl.load(MASKED_ptr + pid_m * stride_ms_m + n * stride_ms_n)
            cond = val > max_val
            max_val = tl.where(cond, val, max_val)
            max_idx = tl.where(cond, n, max_idx)
        tl.store(OUT_idx_ptr + pid_m * stride_i_m + i * stride_i_n, max_idx)
        # zero-out the selected score for next iteration
        tl.store(MASKED_ptr + pid_m * stride_ms_m + max_idx * stride_ms_n, -1.0e30)

    # Now compute weights (sum of selected scores from original scores_copy), then normalize and apply scaling
    total_sum = 0.0
    for j in range(8):
        idx = tl.load(OUT_idx_ptr + pid_m * stride_i_m + j * stride_i_n)
        selected_val = tl.load(SCORES_COPY_ptr + pid_m * stride_sc_m + idx * stride_sc_n)
        total_sum = total_sum + selected_val

    eps = 1e-20
    norm = total_sum + eps
    for j in range(8):
        idx = tl.load(OUT_idx_ptr + pid_m * stride_i_m + j * stride_i_n)
        selected_val = tl.load(SCORES_COPY_ptr + pid_m * stride_sc_m + idx * stride_sc_n)
        w = selected_val / norm * routed_scaling_factor
        tl.store(OUT_w_ptr + pid_m * stride_w_m + j * stride_w_n, w)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure CUDA and float32
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew requires CUDA tensors"

        hidden = hidden_states.contiguous().to(torch.float32)  # [M, K]
        weight = weight.contiguous().to(torch.float32)         # [N, K], N=256
        expert_bias = expert_bias.contiguous().to(torch.float32)  # [N]

        M = hidden.shape[0]
        N = weight.shape[0]  # 256
        K = hidden.shape[1]  # hidden_dim

        # 1) Linear + bias: logits [M, N]
        logits = torch.empty((M, N), device=device, dtype=torch.float32)
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 64
        grid_linear = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_bias_kernel[grid_linear](
            hidden, weight, expert_bias, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3
        )

        # 2) scores = sigmoid(logits) + expert_bias
        # Note: we skip adding expert_bias here since sigmoid_kernel produces sigmoid only; we will add bias later if needed.
        # Compute sigmoid(logits) -> scores
        scores = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_sigmoid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        sigmoid_kernel[grid_sigmoid](
            logits, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )
        # Now add expert_bias to scores in-place to avoid extra tensor
        scores = scores + expert_bias  # broadcast expert_bias [N] to [M, N]

        # 3) Compute group_scores [M, 8]
        group_scores = torch.empty((M, 8), device=device, dtype=torch.float32)
        grid_groups = (M, 8)
        compute_group_scores_kernel[grid_groups](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            num_warps=2, num_stages=2
        )

        # 4) Select top-4 groups per token -> selected_groups [M, 4] (int32)
        selected_groups = torch.empty((M, 4), device=device, dtype=torch.int32)
        grid_select = (M, 4)
        select_top4_groups_kernel[grid_select](
            group_scores, selected_groups,
            M, 8,
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
            num_warps=1, num_stages=1
        )

        # 5) Mask scores: set unselected groups to -inf
        masked_scores = torch.empty((M, N), device=device, dtype=torch.float32)
        # We will write -inf into masked_scores for unselected groups; for selected groups copy scores
        mask_scores_with_groups_kernel[(M,)](
            selected_groups, scores, masked_scores,
            M, N,
            selected_groups.stride(0), selected_groups.stride(1),
            scores.stride(0), scores.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=2, num_stages=1
        )

        # 6) Final top-8 from masked_scores, normalize by sum of original scores, apply scaling -> topk_idx [M, 8] int64, topk_weight [M, 8] float32
        out_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        out_w = torch.empty((M, 8), device=device, dtype=torch.float32)

        # We need original scores_copy for normalization; scores already holds original scores + bias. To ensure correctness, make a copy for normalization.
        scores_copy = scores.clone()

        final_top8_with_weight_and_normalize_kernel[(M,)](
            masked_scores, scores_copy, out_idx, out_w,
            M, N, routed_scaling_factor,
            masked_scores.stride(0), masked_scores.stride(1),
            scores_copy.stride(0), scores_copy.stride(1),
            out_idx.stride(0), out_idx.stride(1),
            out_w.stride(0), out_w.stride(1),
            num_warps=4, num_stages=3
        )

        # Return indices as int64 (original code returns int64), and weights as float32
        return out_idx.to(torch.int64), out_w


def run(*args):
    return ModelNew()(*args)
