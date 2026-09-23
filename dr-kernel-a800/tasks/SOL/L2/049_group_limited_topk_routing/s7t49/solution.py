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
    M, N, K,
    stride_Am, stride_Ak,
    stride_Wn, stride_Wk,
    stride_Om, stride_On,
):
    pid = tl.program_id(axis=0)  # one program per token row
    # Initialize accumulator for this token
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K dimension
    for k in range(0, K):
        # Load hidden state scalar for this token
        a = tl.load(A_ptr + pid * stride_Am + k * stride_Ak)
        # Load weight vector for all N experts at current k
        w = tl.load(W_ptr + tl.arange(0, N) * stride_Wn + k * stride_Wk)
        # Accumulate dot product
        acc += tl.sum(a * w, axis=0)
    # Add bias (broadcast add)
    bias = tl.load(BIAS_ptr + tl.arange(0, N))
    out_vec = acc + bias
    # Store output logits row
    for n in range(0, N):
        tl.store(OUT_ptr + pid * stride_Om + n * stride_On, out_vec[n])


# Kernel 2: scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_bias_kernel(
    LOGITS_ptr,   # *fp32 [M, N]
    BIAS_ptr,     # *fp32 [N]
    OUT_ptr,      # *fp32 [M, N]
    M, N,
    stride_Lm, stride_Ln,
    stride_B,
    stride_Om, stride_On,
):
    pid = tl.program_id(axis=0)
    # Compute sigmoid per column and add bias
    for n in range(0, N):
        logit = tl.load(LOGITS_ptr + pid * stride_Lm + n * stride_Ln)
        sig = 1.0 / (1.0 + tl.exp(-logit))
        bias = tl.load(BIAS_ptr + n * stride_B)
        tl.store(OUT_ptr + pid * stride_Om + n * stride_On, sig + bias)


# Kernel 3: compute group scores per token
# scores: [M, N], group_scores: [M, 8]
@triton.jit
def compute_group_scores_kernel(
    SCORES_ptr,   # *fp32 [M, N]
    GROUPS_ptr,   # *fp32 [M, 8]
    M, N, G, EP,  # N=256, G=8, EP=32
    stride_Sm, stride_Sn,
    stride_Gm, stride_Gn,
):
    pid = tl.program_id(axis=0)
    # We will compute top-2 per group by scanning columns
    for g in range(0, G):
        start = g * EP
        # Initialize top1 and top2 to very small values
        top1 = -1.0e30
        top2 = -1.0e30
        # Scan this group of EP columns
        for j in range(0, EP):
            col = start + j
            score = tl.load(SCORES_ptr + pid * stride_Sm + col * stride_Sn)
            # Update top-2
            if score > top1:
                top2 = top1
                top1 = score
            elif score > top2:
                top2 = score
        # Sum of top-2 for this group
        sum_top2 = top1 + top2
        # Store
        tl.store(GROUPS_ptr + pid * stride_Gm + g * stride_Gn, sum_top2)


# Kernel 4: select top-4 groups per token (iterative elimination, sorted=False)
# scores_group: [M, 8], selected_groups: [M, 4] int32
@triton.jit
def select_top4_groups_kernel(
    GROUPS_ptr,   # *fp32 [M, 8]
    OUT_ptr,      # *int32 [M, 4]
    M, G,
    stride_Gm, stride_Gn,
    stride_Om, stride_On,
):
    pid = tl.program_id(axis=0)
    top = [-1.0e30] * 8  # we can't init arrays, use scalars instead
    # Maintain selected set and update top list
    # We will pick indices and eliminate by setting scores to -inf in a host-pass kernel.
    # But here we just return indices. We need to read groups array repeatedly to find maxima.
    # Iteratively find maxima not already selected.
    # We keep a small array of selected flags. Triton supports scalars.
    sel_flags = [0] * 8
    # Prepare candidates list
    # Simpler: just find 4 maxima positions and store indices
    # We'll use 4 temporary scalars for indices
    idx0, idx1, idx2, idx3 = -1, -1, -1, -1
    # Find first max
    max_val = -1.0e30
    max_idx = -1
    for g in range(0, G):
        val = tl.load(GROUPS_ptr + pid * stride_Gm + g * stride_Gn)
        if val > max_val:
            max_val = val
            max_idx = g
    # Mark and store
    idx0 = max_idx
    sel_flags[idx0] = 1
    # Second max, exclude idx0
    max_val = -1.0e30
    max_idx = -1
    for g in range(0, G):
        if sel_flags[g] == 0:
            val = tl.load(GROUPS_ptr + pid * stride_Gm + g * stride_Gn)
            if val > max_val:
                max_val = val
                max_idx = g
    idx1 = max_idx
    sel_flags[idx1] = 1
    # Third max
    max_val = -1.0e30
    max_idx = -1
    for g in range(0, G):
        if sel_flags[g] == 0:
            val = tl.load(GROUPS_ptr + pid * stride_Gm + g * stride_Gn)
            if val > max_val:
                max_val = val
                max_idx = g
    idx2 = max_idx
    sel_flags[idx2] = 1
    # Fourth max
    max_val = -1.0e30
    max_idx = -1
    for g in range(0, G):
        if sel_flags[g] == 0:
            val = tl.load(GROUPS_ptr + pid * stride_Gm + g * stride_Gn)
            if val > max_val:
                max_val = val
                max_idx = g
    idx3 = max_idx
    # Store as int32
    tl.store(OUT_ptr + pid * stride_Om + 0 * stride_On, idx0)
    tl.store(OUT_ptr + pid * stride_Om + 1 * stride_On, idx1)
    tl.store(OUT_ptr + pid * stride_Om + 2 * stride_On, idx2)
    tl.store(OUT_ptr + pid * stride_Om + 3 * stride_On, idx3)


# Kernel 5: mask scores based on selected groups (set non-selected groups to -inf)
# scores: [M, N], selected_groups: [M, 4], masked_scores: [M, N]
@triton.jit
def mask_scores_with_groups_kernel(
    SCORES_ptr,   # *fp32 [M, N]
    GROUPS_SEL_ptr,  # *int32 [M, 4]
    OUT_mask_ptr,     # *fp32 [M, N]
    M, N, G, EP,
    stride_Sm, stride_Sn,
    stride_Gm, stride_Gn,  # groups_sel
    stride_Om, stride_On,  # out mask
):
    pid = tl.program_id(axis=0)
    # For each column j, determine if it belongs to any selected group
    for j in range(0, N):
        found = 0
        # Check against 4 selected groups
        for t in range(0, 4):
            g = tl.load(GROUPS_SEL_ptr + pid * stride_Gm + t * stride_Gn)
            if g >= 0 and g < G:
                if j >= g * EP and j < (g + 1) * EP:
                    found = 1
                    break
        # Write masked value
        val = tl.load(SCORES_ptr + pid * stride_Sm + j * stride_Sn)
        if found == 0:
            val = -1.0e30  # -inf surrogate
        tl.store(OUT_mask_ptr + pid * stride_Om + j * stride_On, val)


# Kernel 6: select top-8 from masked_scores and compute weight using original logits
# logits: [M, N], masked_scores: [M, N], out_idx: [M, 8] int32, out_weight: [M, 8] fp32
@triton.jit
def select_top8_and_compute_weight_kernel(
    LOGITS_ptr,       # *fp32 [M, N]
    MASKED_ptr,       # *fp32 [M, N]
    OUT_IDX_ptr,      # *int32 [M, 8]
    OUT_WEIGHT_ptr,   # *fp32 [M, 8]
    M, N,
    scale,            # fp32 routed_scaling_factor
    stride_Lm, stride_Ln,
    stride_Mm, stride_Mn,
    stride_I, stride_J,   # strides for OUT_IDX
    stride_Wm, stride_Wn, # strides for OUT_WEIGHT
):
    pid = tl.program_id(axis=0)
    # Iterative elimination to find top-8 (sorted=False)
    candidates = [1.0e30] * N  # maintain "unused" scores list
    used = [0] * N
    for r in range(0, 8):
        max_val = -1.0e30
        max_idx = -1
        # Scan all columns to find current max among unused
        for j in range(0, N):
            if used[j] == 0:
                val = tl.load(MASKED_ptr + pid * stride_Mm + j * stride_Mn)
                if val > max_val:
                    max_val = val
                    max_idx = j
        # Mark used
        used[max_idx] = 1
        # Store index
        tl.store(OUT_IDX_ptr + pid * stride_I + r * stride_J, max_idx)
        # Accumulate numerator using original logits
        logit = tl.load(LOGITS_ptr + pid * stride_Lm + max_idx * stride_Ln)
        numerator = max_val * scale
        # Compute weight: numerator / (numerator + eps)
        eps = 1e-20
        den = numerator + eps
        weight = numerator / den
        tl.store(OUT_WEIGHT_ptr + pid * stride_Wm + r * stride_Wn, weight)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Constants from the original model
        num_experts = 256
        n_group = 8
        experts_per_group = 32
        final_topk = 8

        M = hidden_states.shape[0]
        device = hidden_states.device
        assert device.type == 'cuda', "Triton kernels require CUDA device"

        # Cast to float32 for compute
        A = hidden_states.contiguous().to(torch.float32)  # [M, K]
        W = weight.contiguous().to(torch.float32)         # [N, K]
        bias = expert_bias.contiguous().to(torch.float32) # [N]
        K = A.shape[1]

        # Allocate buffers
        logits = torch.empty((M, num_experts), dtype=torch.float32, device=device)          # [M, N]
        scores = torch.empty((M, num_experts), dtype=torch.float32, device=device)          # [M, N]
        group_scores = torch.empty((M, n_group), dtype=torch.float32, device=device)        # [M, 8]
        selected_groups = torch.empty((M, 4), dtype=torch.int32, device=device)             # [M, 4]
        masked_scores = torch.empty((M, num_experts), dtype=torch.float32, device=device)   # [M, N]
        topk_idx = torch.empty((M, final_topk), dtype=torch.int32, device=device)           # [M, 8]
        topk_weight = torch.empty((M, final_topk), dtype=torch.float32, device=device)      # [M, 8]

        # 1) logits = hidden_states @ weight^T + expert_bias
        grid1 = (M,)
        linear_bias_kernel[grid1](
            A, W, bias, logits,
            M, num_experts, K,
            A.stride(0), A.stride(1),
            W.stride(0), W.stride(1),
            logits.stride(0), logits.stride(1),
            num_warps=4, num_stages=2,
        )

        # 2) scores = sigmoid(logits) + expert_bias
        grid2 = (M,)
        sigmoid_bias_kernel[grid2](
            logits, bias, scores,
            M, num_experts,
            logits.stride(0), logits.stride(1),
            bias.stride(0),
            scores.stride(0), scores.stride(1),
            num_warps=4, num_stages=2,
        )

        # 3) compute group scores per token
        grid3 = (M,)
        compute_group_scores_kernel[grid3](
            scores, group_scores,
            M, num_experts, n_group, experts_per_group,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            num_warps=2, num_stages=2,
        )

        # 4) select top-4 groups per token
        grid4 = (M,)
        select_top4_groups_kernel[grid4](
            group_scores, selected_groups,
            M, n_group,
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
            num_warps=2, num_stages=2,
        )

        # 5) mask scores based on selected groups (set non-selected groups to -inf)
        grid5 = (M,)
        mask_scores_with_groups_kernel[grid5](
            scores, selected_groups, masked_scores,
            M, num_experts, n_group, experts_per_group,
            scores.stride(0), scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=2, num_stages=2,
        )

        # 6) select top-8 from masked scores and compute weight using original logits
        grid6 = (M,)
        select_top8_and_compute_weight_kernel[grid6](
            logits, masked_scores, topk_idx, topk_weight,
            M, num_experts,
            routed_scaling_factor,
            logits.stride(0), logits.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            num_warps=4, num_stages=2,
        )

        # Cast indices to int64 to match original return type for topk_idx
        topk_idx = topk_idx.to(torch.int64)
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
