import torch
import triton
import triton.language as tl


# Triton kernel: elementwise sigmoid on logits and add expert bias
@triton.jit
def sigmoid_add_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                             M, N,
                             stride_Xm, stride_Xn,
                             stride_Bn,
                             stride_Ym, stride_Yn,
                             BLOCK_N: tl.constexpr):
    m = tl.program_id(0)  # one program per token row
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask, other=0.0)
        b = tl.load(Bias_ptr + n_offsets * stride_Bn, mask=mask, other=0.0)
        y = 1.0 / (1.0 + tl.exp(-x)) + b
        tl.store(Y_ptr + m * stride_Ym + n_offsets * stride_Yn, y, mask=mask)


# Triton kernel: given scores_for_routing shaped [M, G, E], compute group_scores [M, G] = sum of top-2 within each group
# We pass S as [M, N] and remap to [M, G, E] via indices.
@triton.jit
def top2_per_group_kernel(S_ptr, GroupScores_ptr,
                          M, N, G, E,
                          stride_Sm, stride_Sn,
                          stride_GSm, stride_GSn,
                          BLOCK_N: tl.constexpr):
    m = tl.program_id(0)  # per token
    # We process all groups in a loop; N must be divisible by E and equal G*E. Here N=256, G=8, E=32.
    # Strategy: for each group g in [0..G), compute top-2 in the range g*E : (g+1)*E.
    for g in range(0, G):
        start = g * E
        for n_start in range(0, E, BLOCK_N):
            n_offsets = start + n_start + tl.arange(0, BLOCK_N)
            mask = n_offsets < (start + E)
            s = tl.load(S_ptr + m * stride_Sm + n_offsets * stride_Sn, mask=mask, other=-float('inf'))
            # local top-2
            # Use a simple two-pass method: find max, remove it, find second max
            max1 = tl.max(s, axis=0)
            # mask out positions equal to max1 (set to -inf)
            s_masked = tl.where(s == max1, -float('inf'), s)
            max2 = tl.max(s_masked, axis=0)
            group_scores = max1 + max2
            # store scalar at [m, g]
            tl.store(GroupScores_ptr + m * stride_GSm + g * stride_GSn, group_scores)


# Triton kernel: per-token top-4 group selection using arg-topk logic. Output per_token_group_idx [M, 4].
# We use a simple insertion approach per token (M programs), selecting top-4 groups and writing indices.
@triton.jit
def select_top4_groups_kernel(GroupScores_ptr, GroupIdx_ptr,
                              M, G,
                              stride_GSm, stride_GSn,
                              stride_GIm, stride_GIg,
                              BLOCK_G: tl.constexpr):
    m = tl.program_id(0)
    gs = tl.load(GroupScores_ptr + m * stride_GSm + tl.arange(0, G) * stride_GSn)  # [G]
    # Initialize top4 arrays with -inf
    top1 = tl.full([1], -float('inf'), tl.float32)
    top2 = tl.full([1], -float('inf'), tl.float32)
    top3 = tl.full([1], -float('inf'), tl.float32)
    top4 = tl.full([1], -float('inf'), tl.float32)
    idx1 = tl.full([1], -1, tl.int32)
    idx2 = tl.full([1], -1, tl.int32)
    idx3 = tl.full([1], -1, tl.int32)
    idx4 = tl.full([1], -1, tl.int32)

    # Compare each group and update top-4
    for g in range(0, G):
        v = gs[g]
        # update top1
        cond1 = v > top1
        idx1_new = tl.full([1], g, tl.int32)
        top1_new = tl.where(cond1, v, top1)
        idx1 = tl.where(cond1, idx1_new, idx1)
        top1 = tl.where(cond1, v, top1)

        # update top2
        cond2 = (v > top2) & (~cond1)
        idx2_new = tl.full([1], g, tl.int32)
        top2_new = tl.where(cond2, v, top2)
        idx2 = tl.where(cond2, idx2_new, idx2)
        top2 = tl.where(cond2, v, top2)

        # update top3
        cond3 = (v > top3) & (~cond1) & (~cond2)
        idx3_new = tl.full([1], g, tl.int32)
        top3_new = tl.where(cond3, v, top3)
        idx3 = tl.where(cond3, idx3_new, idx3)
        top3 = tl.where(cond3, v, top3)

        # update top4
        cond4 = (v > top4) & (~cond1) & (~cond2) & (~cond3)
        idx4_new = tl.full([1], g, tl.int32)
        top4_new = tl.where(cond4, v, top4)
        idx4 = tl.where(cond4, idx4_new, idx4)
        top4 = tl.where(cond4, v, top4)

    # write back to GroupIdx_ptr [M, 4]
    out_ptrs = [GroupIdx_ptr + m * stride_GIm + i * stride_GIg for i in range(4)]
    tl.store(out_ptrs[0], idx1)
    tl.store(out_ptrs[1], idx2)
    tl.store(out_ptrs[2], idx3)
    tl.store(out_ptrs[3], idx4)


# Triton kernel: set group_mask [M, G] to 1.0 at selected groups per token
@triton.jit
def set_group_mask_kernel(GroupIdx_ptr, GroupMask_ptr,
                          M, G,
                          stride_GIm, stride_GIg,
                          stride_GMm, stride_GMg,
                          BLOCK_G: tl.constexpr):
    m = tl.program_id(0)
    idx_ptrs = [GroupIdx_ptr + m * stride_GIm + i * stride_GIg for i in range(4)]
    idxs = [tl.load(ptr) for ptr in idx_ptrs]
    for i in range(4):
        g = idxs[i]
        # write 1.0 at group g for token m
        tl.store(GroupMask_ptr + m * stride_GMm + g * stride_GMg, 1.0)


# Triton kernel: expand group_mask [M, G] to expert mask [M, N] where N=G*E
@triton.jit
def expand_group_mask_kernel(GroupMask_ptr, ExpandedMask_ptr,
                             M, G, E,
                             stride_GMm, stride_GMg,
                             stride_EMm, stride_EMn,
                             BLOCK_N: tl.constexpr):
    m = tl.program_id(0)
    for g in range(0, G):
        # load mask value at [m, g]
        mask_g = tl.load(GroupMask_ptr + m * stride_GMm + g * stride_GMg)
        # For each expert in this group: set ExpandedMask[m, g*E + e] = mask_g
        for e_start in range(0, E, BLOCK_N):
            e_offsets = e_start + tl.arange(0, BLOCK_N)
            n_offsets = g * E + e_offsets
            # mask valid e positions
            e_mask = e_offsets < E
            # expand mask_g to [BLOCK_N] vector
            mask_vec = mask_g + tl.zeros([BLOCK_N], dtype=tl.float32)
            tl.store(ExpandedMask_ptr + m * stride_EMm + n_offsets * stride_EMn,
                     mask_vec, mask=e_mask)


# Triton kernel: apply mask to scores_for_routing, setting non-selected groups to -inf
@triton.jit
def mask_scores_kernel(ExpandedMask_ptr, Scores_ptr, MaskedScores_ptr,
                        M, N,
                        stride_EMm, stride_EMn,
                        stride_Sm, stride_Sn,
                        stride_MSm, stride_MSn,
                        BLOCK_N: tl.constexpr):
    m = tl.program_id(0)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        # load mask
        mask_vals = tl.load(ExpandedMask_ptr + m * stride_EMm + n_offsets * stride_EMn, mask=mask, other=1.0)
        # load scores
        s = tl.load(Scores_ptr + m * stride_Sm + n_offsets * stride_Sn, mask=mask, other=0.0)
        s_masked = tl.where(mask_vals > 0.0, s, -float('inf'))
        tl.store(MaskedScores_ptr + m * stride_MSm + n_offsets * stride_MSn, s_masked, mask=mask)


# Triton kernel: per-token top-8 expert selection from masked scores
@triton.jit
def topk_experts_kernel(Scores_ptr, TopKIdx_ptr,
                         M, N,
                         stride_Sm, stride_Sn,
                         stride_TIm, stride_TIk,
                         BLOCK_N: tl.constexpr):
    m = tl.program_id(0)
    # Select top-8 indices for this token row
    # We implement an insertion approach: maintain top8 values and indices and update
    top1 = tl.full([1], -float('inf'), tl.float32)
    top2 = tl.full([1], -float('inf'), tl.float32)
    top3 = tl.full([1], -float('inf'), tl.float32)
    top4 = tl.full([1], -float('inf'), tl.float32)
    top5 = tl.full([1], -float('inf'), tl.float32)
    top6 = tl.full([1], -float('inf'), tl.float32)
    top7 = tl.full([1], -float('inf'), tl.float32)
    top8 = tl.full([1], -float('inf'), tl.float32)

    idx1 = tl.full([1], -1, tl.int32)
    idx2 = tl.full([1], -1, tl.int32)
    idx3 = tl.full([1], -1, tl.int32)
    idx4 = tl.full([1], -1, tl.int32)
    idx5 = tl.full([1], -1, tl.int32)
    idx6 = tl.full([1], -1, tl.int32)
    idx7 = tl.full([1], -1, tl.int32)
    idx8 = tl.full([1], -1, tl.int32)

    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        s = tl.load(Scores_ptr + m * stride_Sm + n_offsets * stride_Sn, mask=mask, other=-float('inf'))
        for i in range(0, BLOCK_N):
            v = s[i]
            # Compare against current top8 and update accordingly
            # Update top1
            cond1 = v > top1
            idx1_new = tl.full([1], n_offsets[i], tl.int32)
            top1_new = tl.where(cond1, v, top1)
            idx1 = tl.where(cond1, idx1_new, idx1)
            top1 = tl.where(cond1, v, top1)

            cond2 = (v > top2) & (~cond1)
            idx2_new = tl.full([1], n_offsets[i], tl.int32)
            top2_new = tl.where(cond2, v, top2)
            idx2 = tl.where(cond2, idx2_new, idx2)
            top2 = tl.where(cond2, v, top2)

            cond3 = (v > top3) & (~cond1) & (~cond2)
            idx3_new = tl.full([1], n_offsets[i], tl.int32)
            top3_new = tl.where(cond3, v, top3)
            idx3 = tl.where(cond3, idx3_new, idx3)
            top3 = tl.where(cond3, v, top3)

            cond4 = (v > top4) & (~cond1) & (~cond2) & (~cond3)
            idx4_new = tl.full([1], n_offsets[i], tl.int32)
            top4_new = tl.where(cond4, v, top4)
            idx4 = tl.where(cond4, idx4_new, idx4)
            top4 = tl.where(cond4, v, top4)

            cond5 = (v > top5) & (~cond1) & (~cond2) & (~cond3) & (~cond4)
            idx5_new = tl.full([1], n_offsets[i], tl.int32)
            top5_new = tl.where(cond5, v, top5)
            idx5 = tl.where(cond5, idx5_new, idx5)
            top5 = tl.where(cond5, v, top5)

            cond6 = (v > top6) & (~cond1) & (~cond2) & (~cond3) & (~cond4) & (~cond5)
            idx6_new = tl.full([1], n_offsets[i], tl.int32)
            top6_new = tl.where(cond6, v, top6)
            idx6 = tl.where(cond6, idx6_new, idx6)
            top6 = tl.where(cond6, v, top6)

            cond7 = (v > top7) & (~cond1) & (~cond2) & (~cond3) & (~cond4) & (~cond5) & (~cond6)
            idx7_new = tl.full([1], n_offsets[i], tl.int32)
            top7_new = tl.where(cond7, v, top7)
            idx7 = tl.where(cond7, idx7_new, idx7)
            top7 = tl.where(cond7, v, top7)

            cond8 = (v > top8) & (~cond1) & (~cond2) & (~cond3) & (~cond4) & (~cond5) & (~cond6) & (~cond7)
            idx8_new = tl.full([1], n_offsets[i], tl.int32)
            top8_new = tl.where(cond8, v, top8)
            idx8 = tl.where(cond8, idx8_new, idx8)
            top8 = tl.where(cond8, v, top8)

    # Write top8 indices to TopKIdx_ptr [M, 8]
    out_ptrs = [TopKIdx_ptr + m * stride_TIm + i * stride_TIk for i in range(8)]
    tl.store(out_ptrs[0], idx1)
    tl.store(out_ptrs[1], idx2)
    tl.store(out_ptrs[2], idx3)
    tl.store(out_ptrs[3], idx4)
    tl.store(out_ptrs[4], idx5)
    tl.store(out_ptrs[5], idx6)
    tl.store(out_ptrs[6], idx7)
    tl.store(out_ptrs[7], idx8)


# Triton kernel: normalize and scale selected weights using selected_scores per token
@triton.jit
def normalize_scale_kernel(Scores_ptr, SelectedIdx_ptr, Weight_ptr, Scaled_ptr,
                           M, K,
                           stride_Sm, stride_Sk,
                           stride_SMm, stride_SMk,
                           stride_WMm, stride_WNn,
                           stride_SMm_out, stride_SMn_out,
                           scaling: tl.float32,
                           BLOCK_K: tl.constexpr):
    m = tl.program_id(0)
    # Load selected indices [8]
    for i in range(8):
        idx = tl.load(SelectedIdx_ptr + m * stride_SMm + i * stride_SMk)
        # Gather selected_scores from scores_ptr at [m, idx]
        score = tl.load(Scores_ptr + m * stride_Sm + idx * stride_Sk)
        # Load corresponding weight from Weight_ptr [M, K] at column idx
        w = tl.load(Weight_ptr + m * stride_WMm + idx * stride_WNn)
        # Normalize by sum of selected scores (compute sum over 8)
        sum_scores = tl.zeros([1], dtype=tl.float32)
        for j in range(8):
            s_j = tl.load(Scores_ptr + m * stride_Sm + tl.load(SelectedIdx_ptr + m * stride_SMm + j * stride_SMk) * stride_Sk)
            sum_scores += s_j
        norm = 1.0 / (sum_scores + 1e-20)
        scaled = w * norm * scaling
        tl.store(Scaled_ptr + m * stride_SMm_out + idx * stride_SMn_out, scaled)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                weight: torch.Tensor,
                expert_bias: torch.Tensor,
                routed_scaling_factor: float):
        # hidden_states: [M, K], weight: [E, K], expert_bias: [E], routed_scaling_factor: float
        M, K = hidden_states.shape
        E = weight.shape[0]
        assert E == 256, "Expected 256 experts"
        assert K == weight.shape[1], "Incompatible hidden_size and weight"

        # 1) Compute logits via PyTorch (robust and matches original)
        # Use float32 for numerical stability
        logits = torch.nn.functional.linear(hidden_states.to(torch.float32), weight.to(torch.float32))

        # 2) Triton: sigmoid + expert bias
        scores = torch.empty_like(logits, dtype=torch.float32, device=logits.device)
        BLOCK_N = 128
        grid_sigmoid = (M,)
        sigmoid_add_bias_kernel[grid_sigmoid](logits, expert_bias.to(torch.float32), scores,
                                              M, E,
                                              logits.stride(0), logits.stride(1),
                                              expert_bias.stride(0),
                                              scores.stride(0), scores.stride(1),
                                              BLOCK_N=BLOCK_N)

        # 3) Triton: per-group top-2 aggregation to get group_scores [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        grid_g_top2 = (M,)
        top2_per_group_kernel[grid_g_top2](scores, group_scores,
                                           M, E, 8, 32,
                                           scores.stride(0), scores.stride(1),
                                           group_scores.stride(0), group_scores.stride(1),
                                           BLOCK_N=32)  # E=32, so one pass

        # 4) Triton: per-token top-4 group selection (arg-topk-like), output [M, 4] int32
        per_token_group_idx = torch.empty((M, 4), dtype=torch.int32, device=scores.device)
        grid_g_top4 = (M,)
        select_top4_groups_kernel[grid_g_top4](group_scores, per_token_group_idx,
                                               M, 8,
                                               group_scores.stride(0), group_scores.stride(1),
                                               per_token_group_idx.stride(0), per_token_group_idx.stride(1),
                                               BLOCK_G=8)

        # 5) Triton: set group_mask [M, 8] to 1.0 at selected groups
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        grid_set_mask = (M,)
        set_group_mask_kernel[grid_set_mask](per_token_group_idx, group_mask,
                                             M, 8,
                                             per_token_group_idx.stride(0), per_token_group_idx.stride(1),
                                             group_mask.stride(0), group_mask.stride(1),
                                             BLOCK_G=8)

        # 6) Triton: expand group_mask to [M, 256]
        expanded_mask = torch.empty((M, E), dtype=torch.float32, device=scores.device)
        grid_expand = (M,)
        expand_group_mask_kernel[grid_expand](group_mask, expanded_mask,
                                              M, 8, 32,
                                              group_mask.stride(0), group_mask.stride(1),
                                              expanded_mask.stride(0), expanded_mask.stride(1),
                                              BLOCK_N=32)

        # 7) Triton: mask non-selected group scores to -inf
        masked_scores = torch.empty_like(scores, dtype=torch.float32, device=scores.device)
        grid_mask_scores = (M,)
        mask_scores_kernel[grid_mask_scores](expanded_mask, scores, masked_scores,
                                             M, E,
                                             expanded_mask.stride(0), expanded_mask.stride(1),
                                             scores.stride(0), scores.stride(1),
                                             masked_scores.stride(0), masked_scores.stride(1),
                                             BLOCK_N=128)

        # 8) Triton: per-token top-8 expert selection from masked scores, output [M, 8] int32
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=scores.device)
        grid_topk = (M,)
        topk_experts_kernel[grid_topk](masked_scores, topk_idx,
                                       M, E,
                                       masked_scores.stride(0), masked_scores.stride(1),
                                       topk_idx.stride(0), topk_idx.stride(1),
                                       BLOCK_N=128)

        # 9) Triton: normalize and scale selected weights using selected_scores and routed_scaling_factor
        # selected_scores can be gathered from 'scores' using topk_idx
        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        for i in range(8):
            # gather scores per token at selected expert indices
            selected_scores[:, i] = scores.index_select(dim=1, index=topk_idx[:, i])

        # normalized weights scaled
        scaled_weights = torch.empty_like(hidden_states, dtype=torch.float32, device=hidden_states.device)
        # Using Triton for the final normalization and scaling
        grid_norm = (M,)
        normalize_scale_kernel[grid_norm](selected_scores, topk_idx, weight.to(torch.float32),
                                          scaled_weights,
                                          M, K,
                                          selected_scores.stride(0), selected_scores.stride(1),
                                          topk_idx.stride(0), topk_idx.stride(1),
                                          weight.stride(0), weight.stride(1),
                                          scaled_weights.stride(0), scaled_weights.stride(1),
                                          scaling=routed_scaling_factor,
                                          BLOCK_K=128)

        return topk_idx, scaled_weights


def run(*args):
    return ModelNew()(*args)
