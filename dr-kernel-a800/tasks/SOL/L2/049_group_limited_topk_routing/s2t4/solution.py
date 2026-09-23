import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Inputs:
        # hidden_states: [num_tokens, hidden_dim]
        # weight: [num_experts, hidden_dim], num_experts = 256
        # expert_bias: [num_experts]
        # routed_scaling_factor: scalar float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Allocate outputs (int32 for indices, float32 for weights)
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _group_limited_topk_kernel[grid](
            hidden_states, weight, expert_bias,
            topk_idx, topk_weight,
            num_tokens, hidden_dim, routed_scaling_factor,
            num_experts=num_experts, n_groups=8, experts_per_group=32
        )

        return topk_idx, topk_weight


@triton.jit
def _group_limited_topk_kernel(
    hidden_ptr,         # *f32, [num_tokens, hidden_dim]
    weight_ptr,         # *f32, [num_experts, hidden_dim]
    expert_bias_ptr,    # *f32, [num_experts]
    topk_idx_ptr,       # *i32, [num_tokens, 8]
    topk_weight_ptr,    # *f32, [num_tokens, 8]
    num_tokens,         # int32
    hidden_dim,         # int32
    routed_scaling_factor,  # f32
    num_experts: tl.constexpr,      # 256
    n_groups: tl.constexpr,         # 8
    experts_per_group: tl.constexpr # 32
):
    token = tl.program_id(0)

    # 1) Compute logits per expert for this token: scores[e] = dot(hidden[token, :], weight[e, :])
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    for e in range(0, num_experts):
        score = 0.0
        for j in range(0, hidden_dim):
            h = tl.load(hidden_ptr + token * hidden_dim + j)
            w = tl.load(weight_ptr + e * hidden_dim + j)
            score += h * w
        scores[e] = score

    # 2) Apply sigmoid and add expert bias
    # sigmoid(x) = 1 / (1 + exp(-x))
    scores = 1.0 / (1.0 + tl.exp(-scores))
    bias = tl.load(expert_bias_ptr + tl.arange(0, num_experts))
    scores_for_routing = scores + bias  # broadcasted add

    # 3) Compute top-2 per group and group scores (sum of top-2)
    group_top1 = tl.full((n_groups,), -1.0e30, dtype=tl.float32)
    group_top2 = tl.full((n_groups,), -1.0e30, dtype=tl.float32)
    for g in range(0, n_groups):
        start = g * experts_per_group
        top1 = -1.0e30
        top2 = -1.0e30
        for i in range(0, experts_per_group):
            idx = start + i
            s = scores_for_routing[idx]
            if s > top1:
                top2 = top1
                top1 = s
            elif s > top2:
                top2 = s
        group_top1[g] = top1
        group_top2[g] = top2

    # 4) Select top-4 groups based on group scores (sum of top-2)
    selected_groups = tl.zeros((4,), dtype=tl.int32)
    for r in range(0, 4):
        best_sum = -1.0e30
        best_g = 0
        for g in range(0, n_groups):
            s = group_top1[g] + group_top2[g]
            # Ensure uniqueness among selected
            unique = True
            for k in range(0, r):
                if selected_groups[k] == g:
                    unique = False
                    break
            if unique and s > best_sum:
                best_sum = s
                best_g = g
        selected_groups[r] = best_g

    # 5) Build per-expert mask: 1 for selected groups, 0 otherwise
    per_exp_mask = tl.zeros((num_experts,), dtype=tl.int32)
    for g in selected_groups:
        group = g
        start = group * experts_per_group
        for i in range(0, experts_per_group):
            idx = start + i
            per_exp_mask[idx] = 1

    # 6) Mask scores_for_routing: non-selected groups -> -inf
    masked_scores = tl.full((num_experts,), -1.0e30, dtype=tl.float32)
    for e in range(0, num_experts):
        if per_exp_mask[e] == 1:
            masked_scores[e] = scores_for_routing[e]
        else:
            masked_scores[e] = -1.0e30

    # 7) Select top-8 from masked_scores (allow duplicates)
    selected_idx = tl.zeros((8,), dtype=tl.int32)
    selected_val = tl.zeros((8,), dtype=tl.float32)
    for r in range(0, 8):
        best_val = -1.0e30
        best_e = 0
        for e in range(0, num_experts):
            if masked_scores[e] > best_val:
                best_val = masked_scores[e]
                best_e = e
        selected_idx[r] = best_e
        selected_val[r] = best_val

    # 8) Normalize using original logits for selected indices, then apply scaling factor
    original_selected = tl.zeros((8,), dtype=tl.float32)
    sum_original = 0.0
    for r in range(0, 8):
        e = selected_idx[r]
        original_selected[r] = scores[e]
        sum_original += original_selected[r]
    # Add small epsilon to avoid division by zero
    sum_original = tl.maximum(sum_original, 1e-20)
    normalized = selected_val * routed_scaling_factor / sum_original

    # 9) Store outputs: one row per token
    base_out = topk_idx_ptr + token * 8
    for r in range(0, 8):
        tl.store(base_out + r, selected_idx[r])
        tl.store(topk_weight_ptr + token * 8 + r, normalized[r])


def run(*args):
    return ModelNew()(*args)
