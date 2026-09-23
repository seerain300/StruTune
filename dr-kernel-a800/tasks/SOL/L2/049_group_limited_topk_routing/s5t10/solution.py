import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Ym, stride_Yn,
                         BLOCK_N: tl.constexpr):
    # One program per row m
    m = tl.program_id(0)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask, other=0.0)
        b = tl.load(Bias_ptr + n_offsets, mask=mask, other=0.0)
        x = 1.0 / (1.0 + tl.exp(-x))  # sigmoid
        x = x + b
        tl.store(Y_ptr + m * stride_Ym + n_offsets * stride_Yn, x, mask=mask)


@triton.jit
def top2_per_group_kernel(S_ptr, GroupScores_ptr,
                          M, G, E,  # G=8, E=32
                          stride_Sm, stride_Sg, stride_Se,  # [M, G, E]
                          stride_Gm, stride_Gg,            # [M, G]
                          BLOCK_E: tl.constexpr):
    # One program per m
    m = tl.program_id(0)
    for g in range(0, G):
        # Iterate over E in blocks and compute top-2
        top1 = -float('inf')
        top2 = -float('inf')
        for e_start in range(0, E, BLOCK_E):
            e_offsets = e_start + tl.arange(0, BLOCK_E)
            mask_e = e_offsets < E
            base = m * stride_Sm + g * stride_Sg
            vals = tl.load(S_ptr + base + e_offsets * stride_Se, mask=mask_e, other=-float('inf'))
            # Reduce within the block
            block_max = tl.max(vals, axis=0)
            # Find second highest by excluding block_max
            vals2 = tl.where(vals == block_max, -float('inf'), vals)
            block_max2 = tl.max(vals2, axis=0)
            # Update global top2
            # We need to combine with existing top1/top2
            # Compare block_max
            tmp1 = block_max
            tmp2 = block_max2
            # Ensure we never overwrite existing top1 unless better, and top2 unless worse
            if top1 < tmp1:
                top2 = top1
                top1 = tmp1
            elif tmp1 > top2:
                top2 = tmp1
            # tmp2 only updates top2 if tmp2 is greater than current top2
            if tmp2 > top2:
                top2 = tmp2
        tl.store(GroupScores_ptr + m * stride_Gm + g * stride_Gg, top1 + top2)


@triton.jit
def masked_scores_kernel(S_ptr, Mask_ptr, NegInf, M, N,
                         stride_Sm, stride_Sn,
                         stride_Mm, stride_Mn):
    # One program per m
    m = tl.program_id(0)
    for n in range(0, N):
        s = tl.load(S_ptr + m * stride_Sm + n * stride_Sn)
        keep = tl.load(Mask_ptr + m * stride_Mm + n * stride_Mn) > 0.0
        new_s = tl.where(keep, s, NegInf)
        tl.store(S_ptr + m * stride_Sm + n * stride_Sn, new_s)


@triton.jit
def topk_experts_arg_kernel(Scores_ptr, TopKIdx_ptr,
                            M, N, K,  # K = 8
                            stride_Sm, stride_Sn,
                            stride_I0, stride_I1):
    # One program per m
    m = tl.program_id(0)
    # We'll iterate K times and each time pick the max, masking it to -inf for subsequent iterations.
    for k in range(0, K):
        max_val = -float('inf')
        arg = -1
        for n in range(0, N):
            val = tl.load(Scores_ptr + m * stride_Sm + n * stride_Sn)
            if val > max_val:
                max_val = val
                arg = n
        tl.store(TopKIdx_ptr + m * stride_I0 + k * stride_I1, arg)
        # Mask this index to -inf for next iterations
        for n in range(0, N):
            val = tl.load(Scores_ptr + m * stride_Sm + n * stride_Sn)
            new_val = tl.where(n == arg, -float('inf'), val)
            tl.store(Scores_ptr + m * stride_Sm + n * stride_Sn, new_val)


@triton.jit
def normalize_and_scale_kernel(Idx_ptr, Selected_ptr, Scale, Out_ptr,
                               M, K,
                               stride_I0, stride_I1,
                               stride_S0, stride_S1,
                               stride_O0, stride_O1):
    # One program per m
    m = tl.program_id(0)
    sum_vals = 0.0
    for k in range(0, K):
        idx = tl.load(Idx_ptr + m * stride_I0 + k * stride_I1)
        val = tl.load(Selected_ptr + m * stride_S0 + k * stride_S1)
        sum_vals += val
    inv = 1.0 / (sum_vals + 1e-20)
    for k in range(0, K):
        idx = tl.load(Idx_ptr + m * stride_I0 + k * stride_I1)
        val = tl.load(Selected_ptr + m * stride_S0 + k * stride_S1)
        out_val = val * inv * Scale
        tl.store(Out_ptr + m * stride_O0 + k * stride_O1, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-only implementation:
        - Input logits: [M, N], N=256 (from hidden_states @ weight.T outside)
        - expert_bias: [N], learnable bias added after sigmoid
        - routed_scaling_factor: float scaling for final weights
        Returns:
        - topk_idx: [M, 8], indices of selected experts per token
        - topk_weight: [M, 8], normalized and scaled weights
        """
        M, N = logits.shape
        device = logits.device

        # 1) Sigmoid + expert bias via Triton
        scores = torch.empty_like(logits)
        grid_sigmoid = (M,)
        sigmoid_bias_kernel[grid_sigmoid](
            logits, expert_bias.to(torch.float32), scores,
            M, N,
            scores.stride(0), scores.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_N=128
        )

        # 2) Compute top-2 per group (G=8, E=32) via Triton
        G = 8
        E = N // G  # 32
        group_scores = torch.empty((M, G), device=device, dtype=torch.float32)
        grid_top2 = (M,)
        top2_per_group_kernel[grid_top2](
            scores, group_scores,
            M, G, E,
            scores.stride(0), scores.stride(1), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            BLOCK_E=32
        )

        # 3) Per-token top-4 groups via PyTorch topk (small and fast)
        _, group_idx = torch.topk(group_scores, k=4, dim=1, sorted=False)  # [M, 4]

        # 4) Build group_mask [M, G]: selected groups = 1, others = 0
        group_mask = torch.zeros((M, G), device=device, dtype=torch.float32)
        for m in range(M):
            for j in range(4):
                g = int(group_idx[m, j].item())
                group_mask[m, g] = 1.0

        # 5) Expand group_mask to [M, N]: expand groups into experts
        expanded_mask = torch.empty((M, N), device=device, dtype=torch.float32)
        for m in range(M):
            for n in range(N):
                group_id = (n // E)  # maps expert n to group
                if group_id < G and (group_mask[m, group_id] > 0.0):
                    expanded_mask[m, n] = 1.0
                else:
                    expanded_mask[m, n] = 0.0

        # 6) Mask scores: non-selected groups set to -inf via Triton
        masked_scores = torch.empty_like(scores)
        grid_mask = (M,)
        masked_scores_kernel[grid_mask](
            scores, expanded_mask, float('-inf'), M, N,
            scores.stride(0), scores.stride(1),
            expanded_mask.stride(0), expanded_mask.stride(1)
        )

        # 7) Per-token top-8 experts from masked scores via Triton
        topk_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        grid_topk_exp = (M,)
        topk_experts_arg_kernel[grid_topk_exp](
            masked_scores, topk_idx,
            M, N, 8,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1)
        )

        # 8) Gather selected scores from original scores for normalization
        selected_scores = torch.empty((M, 8), device=device, dtype=torch.float32)
        for m in range(M):
            for k in range(8):
                idx = int(topk_idx[m, k].item())
                selected_scores[m, k] = scores[m, idx]

        # 9) Normalize and scale via Triton
        topk_weight = torch.empty((M, 8), device=device, dtype=torch.float32)
        grid_norm = (M,)
        normalize_and_scale_kernel[grid_norm](
            topk_idx, selected_scores, routed_scaling_factor,
            M, 8,
            topk_idx.stride(0), topk_idx.stride(1),
            selected_scores.stride(0), selected_scores.stride(1),
            topk_weight.stride(0), topk_weight.stride(1)
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
