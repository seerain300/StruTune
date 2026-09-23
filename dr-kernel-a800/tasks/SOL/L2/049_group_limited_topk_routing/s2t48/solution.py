import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32
        # weight: [num_experts, hidden_dim], float32 (num_experts=256)
        # expert_bias: [num_experts], float32
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Allocate outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Single Triton kernel performing all steps
        # Grid: one program per token
        grid = (num_tokens,)

        # Launch Triton kernel
        triton.runtime.jit(
            r"""
            # Define a Triton kernel (inlined) that:
            # - computes per-token logits for 256 experts
            # - applies sigmoid + expert_bias
            # - groups 256 -> 8 groups of 32, computes top-2 per group and sums
            # - selects top-4 groups
            # - builds group_mask and expands to per-expert mask; set non-selected groups to -inf
            # - selects top-8 from masked scores
            # - normalizes by sum(original selected scores + 1e-20) and applies routed_scaling_factor
            # - writes out topk_idx and topk_weight
            def kernel(hidden_ptr, weight_ptr, bias_ptr, out_idx_ptr, out_weight_ptr, num_tokens, hidden_dim, num_experts, routed_scaling):
                pid = tl.program_id(axis=0)  # one program per token
                if pid >= num_tokens:
                    return

                # Load hidden vector for this token
                h = [0.0] * hidden_dim
                for d in range(0, hidden_dim):
                    h[d] = tl.load(hidden_ptr + pid * hidden_dim + d)

                # Compute logits for all num_experts via dot-product
                scores = [0.0] * num_experts
                for e in range(0, num_experts):
                    acc = 0.0
                    for d in range(0, hidden_dim):
                        w = tl.load(weight_ptr + e * hidden_dim + d)
                        acc += w * h[d]
                    scores[e] = 1.0 / (1.0 + tl.exp(-acc))  # sigmoid

                # Add expert bias
                for e in range(0, num_experts):
                    b = tl.load(bias_ptr + e)
                    scores[e] += b

                # Per-group top-2 and group scores (N_GROUPS=8, EXPERTS_PER_GROUP=32)
                group_top2 = [0.0] * 8
                for g in range(0, 8):
                    top1 = -float("inf")
                    top2 = -float("inf")
                    start = g * 32
                    for i in range(0, 32):
                        e_idx = start + i
                        v = scores[e_idx]
                        if v > top2:
                            top2 = v
                        if v > top1:
                            top2 = top1
                            top1 = v
                    group_top2[g] = top1 + top2

                # Select top-4 groups (descending): store indices in top4_group[0..3]
                top4_group = [-1] * 4
                top4_score = [0.0] * 4
                for g in range(0, 8):
                    include = True
                    for m in range(0, 4):
                        if (top4_score[m] > 0.0) and (top4_score[m] > group_top2[g]):
                            include = False
                            break
                    if include:
                        min_idx = 0
                        min_score = top4_score[0]
                        for m in range(1, 4):
                            if (top4_score[m] == 0.0) or ((top4_score[m] < min_score) and (top4_group[m] != -1)):
                                min_idx = m
                                min_score = top4_score[m]
                        top4_group[min_idx] = g
                        top4_score[min_idx] = group_top2[g]

                # Build group_mask [8] (1.0 for selected groups, 0.0 otherwise)
                group_mask = [0.0] * 8
                for m in range(0, 4):
                    group_mask[top4_group[m]] = 1.0

                # Expand group_mask to per-expert mask and create masked_scores
                masked_scores = [0.0] * num_experts
                for e in range(0, num_experts):
                    e_group = e // 32
                    keep = group_mask[e_group]  # 1.0 or 0.0
                    if keep != 1.0:
                        masked_scores[e] = -float("inf")
                    else:
                        masked_scores[e] = scores[e]

                # Select top-8 from masked_scores (duplicates allowed)
                top8_vals = [0.0] * 8
                top8_idx = [0] * 8
                for m in range(0, 8):
                    best_val = -float("inf")
                    best_idx = 0
                    for e in range(0, num_experts):
                        if masked_scores[e] > best_val:
                            best_val = masked_scores[e]
                            best_idx = e
                    top8_vals[m] = best_val
                    top8_idx[m] = best_idx

                # Normalize using original routed scores (pre-mask) for selected indices: sum(selected_original + 1e-20)
                denom = 0.0
                selected_original = [0.0] * 8
                for m in range(0, 8):
                    e = top8_idx[m]
                    selected_original[m] = scores[e]
                    denom += selected_original[m]
                for m in range(0, 8):
                    selected_original[m] /= (denom + 1e-20)
                    tl.store(out_weight_ptr + pid * 8 + m, selected_original[m] * routed_scaling)

                # Store topk_idx
                for m in range(0, 8):
                    tl.store(out_idx_ptr + pid * 8 + m, top8_idx[m])
            """
        )(hidden_states, weight, expert_bias, topk_idx, topk_weight, num_tokens, hidden_dim, num_experts, routed_scaling_factor, num_warps=4)

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
