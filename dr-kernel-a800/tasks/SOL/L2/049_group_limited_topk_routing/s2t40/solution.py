import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32, on device
        # weight: [num_experts, hidden_dim], float32, on device (num_experts=256)
        # expert_bias: [num_experts], float32, on device
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Outputs (int32 indices and float32 weights)
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Ensure inputs are contiguous
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()
        expert_bias = expert_bias.contiguous()

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        triton_kernel[grid](
            hidden_states, weight, expert_bias, routed_scaling_factor,
            topk_idx, topk_weight,
            num_tokens, hidden_dim, num_experts
        )

        return topk_idx, topk_weight


@triton.jit
def triton_kernel(
    hidden_states,  # *ptr* [num_tokens, hidden_dim]
    weight,         # *ptr* [num_experts, hidden_dim]
    expert_bias,    # *ptr* [num_experts]
    routed_scaling_factor,  # float
    topk_idx,       # *ptr* [num_tokens, 8] int32
    topk_weight,    # *ptr* [num_tokens, 8] float32
    num_tokens, hidden_dim, num_experts
):
    pid = tl.program_id(0)  # token id

    # 1) Compute scores[token, e] = dot(hidden_states[token, :], weight[e, :])
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    for j in range(0, hidden_dim):
        hs_val = tl.load(hidden_states + pid * hidden_dim + j)
        for e in range(0, num_experts):
            w_val = tl.load(weight + e * hidden_dim + j)
            scores[e] += hs_val * w_val

    # 2) Apply sigmoid
    scores = 1.0 / (1.0 + tl.exp(-scores))

    # 3) Add expert bias
    for e in range(0, num_experts):
        scores[e] += tl.load(expert_bias + e)

    # 4) Grouping: 8 groups of 32, compute top-2 per group, sum to form group_scores
    group_scores = tl.zeros((8,), dtype=tl.float32)
    for g in range(0, 8):
        top1 = -float('inf')
        # First pass: find top-1 in group g
        for e in range(0, num_experts):
            group = e // 32
            if group == g:
                val = scores[e]
                top1 = tl.maximum(top1, val)
        # Second pass: find top-2 among those < top1
        top2 = -float('inf')
        for e in range(0, num_experts):
            group = e // 32
            if group == g and scores[e] < top1 and scores[e] > top2:
                top2 = scores[e]
        group_scores[g] = top1 + top2

    # 5) Select top-4 groups (indices)
    selected_groups = tl.zeros((4,), dtype=tl.int32)
    for r in range(0, 4):
        best_val = -float('inf')
        for g in range(0, 8):
            if group_scores[g] > best_val:
                best_val = group_scores[g]
        found = 0
        best_idx = -1
        for g in range(0, 8):
            if group_scores[g] == best_val:
                best_idx = g
                found = 1
                break
        selected_groups[r] = best_idx
        # Remove this group from future consideration
        group_scores[best_idx] = -float('inf')

    # 6) Build group mask [8], 1 for selected groups, 0 otherwise
    group_mask = tl.zeros((8,), dtype=tl.float32)
    for r in range(0, 4):
        g = selected_groups[r]
        group_mask[g] = 1.0

    # 7) Expand group mask to per-expert mask and apply to scores_for_routing:
    #    For each expert e, if e belongs to a non-selected group, set its score to -inf.
    scores_expanded = scores  # float32
    for e in range(0, num_experts):
        group = e // 32
        scores_expanded[e] = scores_expanded[e] * group_mask[group]

    # 8) Select top-8 from masked scores
    for r in range(0, 8):
        best_val = -float('inf')
        best_idx = -1
        for e in range(0, num_experts):
            if scores_expanded[e] > best_val:
                best_val = scores_expanded[e]
                best_idx = e
        # Record index
        tl.store(topk_idx + pid * 8 + r, best_idx)
        # Normalize using original scores (before masking)
        original_score = scores[best_idx]
        denom = 0.0
        for e2 in range(0, num_experts):
            denom += scores[e2]
        norm = original_score / (denom + 1e-20)
        tl.store(topk_weight + pid * 8 + r, norm * routed_scaling_factor)
        # Remove best by setting to -inf
        scores_expanded[best_idx] = -float('inf')


# Example usage in evaluator:
# model = ModelNew().cuda()
# hidden_states = torch.randn(2048, 128, device='cuda', dtype=torch.float32)
# weight = torch.randn(256, 128, device='cuda', dtype=torch.float32)
# expert_bias = torch.randn(256, device='cuda', dtype=torch.float32)
# routed_scaling_factor = 1.0
# topk_idx, topk_weight = model(hidden_states, weight, expert_bias, routed_scaling_factor)


def run(*args):
    return ModelNew()(*args)
