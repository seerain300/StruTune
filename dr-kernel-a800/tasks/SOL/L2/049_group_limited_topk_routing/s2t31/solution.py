import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim]
        # weight: [num_experts, hidden_dim] with num_experts == 256
        # expert_bias: [num_experts]
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel
        grid = (num_tokens,)
        run_kernel[grid](
            hidden_states, weight, expert_bias, routed_scaling_factor,
            num_tokens, hidden_dim, num_experts,
            topk_idx, topk_weight
        )

        return topk_idx, topk_weight


@triton.jit
def run_kernel(
    hidden_ptr,          # *f32, [num_tokens, hidden_dim]
    weight_ptr,          # *f32, [num_experts, hidden_dim]
    bias_ptr,            # *f32, [num_experts]
    scale,               # f32
    num_tokens,          # int32
    hidden_dim,          # int32
    num_experts,         # int32
    out_idx_ptr,         # *i32, [num_tokens, 8]
    out_weight_ptr,      # *f32, [num_tokens, 8]
):
    pid = tl.program_id(0)  # token index

    EXPERTS_PER_GROUP = 32
    NUM_GROUPS = 8
    TOPK_GROUP = 4
    TOPK = 8

    # Compute logits: scores[token, e] = dot(hidden[token, :], weight[e, :])
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    hs_ptr = hidden_ptr + pid * hidden_dim
    for e in range(0, num_experts):
        w_ptr = weight_ptr + e * hidden_dim
        sum_val = 0.0
        for j in range(0, hidden_dim):
            sum_val += tl.load(hs_ptr + j) * tl.load(w_ptr + j)
        # Apply sigmoid
        scores[e] = 1.0 / (1.0 + tl.exp(-sum_val))

    # Add expert bias
    for e in range(0, num_experts):
        scores[e] += tl.load(bias_ptr + e)

    # Compute group_scores: top-2 per group sum
    group_scores = tl.zeros((NUM_GROUPS,), dtype=tl.float32)
    for g in range(0, NUM_GROUPS):
        group_start = g * EXPERTS_PER_GROUP
        top1_val = -float('inf')
        top2_val = -float('inf')
        for j in range(0, EXPERTS_PER_GROUP):
            e = group_start + j
            val = scores[e]
            if val > top1_val:
                top2_val = top1_val
                top1_val = val
            elif val > top2_val:
                top2_val = val
        group_scores[g] = top1_val + top2_val

    # Select top-4 groups (descending group_scores)
    selected_groups = tl.zeros((TOPK_GROUP,), dtype=tl.int32)
    for g in range(0, TOPK_GROUP):
        best_gs = -float('inf')
        best_idx = -1
        for g2 in range(0, NUM_GROUPS):
            if group_scores[g2] > best_gs:
                best_gs = group_scores[g2]
                best_idx = g2
        selected_groups[g] = best_idx
        # zero it out for future selections
        group_scores[best_idx] = -float('inf')

    # Build group_mask: 1 for selected groups, 0 otherwise
    group_mask = tl.zeros((NUM_GROUPS,), dtype=tl.int32)
    for g in range(0, TOPK_GROUP):
        group_mask[selected_groups[g]] = 1

    # Apply mask: set non-selected groups to -inf
    masked_scores = scores
    for g in range(0, NUM_GROUPS):
        if group_mask[g] == 0:
            group_start = g * EXPERTS_PER_GROUP
            for j in range(0, EXPERTS_PER_GROUP):
                e = group_start + j
                masked_scores[e] = -float('inf')

    # Select top-8 from masked scores
    selected_idx = tl.zeros((TOPK,), dtype=tl.int32)
    selected_val = tl.zeros((TOPK,), dtype=tl.float32)
    for r in range(0, TOPK):
        best_val = -float('inf')
        best_e = -1
        for e in range(0, num_experts):
            val = masked_scores[e]
            if val > best_val:
                best_val = val
                best_e = e
        selected_idx[r] = best_e
        selected_val[r] = best_val
        masked_scores[best_e] = -float('inf')

    # Normalize using original logits: gather original scores and normalize
    original_scores = scores
    total = 0.0
    for r in range(0, TOPK):
        total += original_scores[selected_idx[r]]
    total = total + 1e-20
    for r in range(0, TOPK):
        out_weight_ptr[pid * TOPK + r] = (original_scores[selected_idx[r]] / total) * scale
        out_idx_ptr[pid * TOPK + r] = selected_idx[r]


def run(*args):
    return ModelNew()(*args)
