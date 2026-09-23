import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32, CUDA
        # weight: [num_experts, hidden_dim], float32, CUDA (num_experts must be 256)
        # expert_bias: [num_experts], float32, CUDA
        # routed_scaling_factor: float

        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be on CUDA"
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Allocate outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: 1D grid over tokens
        grid = (num_tokens,)

        compute_routed_kernel[grid](
            hidden_states, weight, expert_bias,
            num_tokens, hidden_dim, num_experts, routed_scaling_factor,
            topk_idx, topk_weight
        )

        return topk_idx, topk_weight


# Triton kernel: performs all computations in one pass per token
@triton.jit
def compute_routed_kernel(
    hidden_states, weight, expert_bias,
    num_tokens, hidden_dim, num_experts, routed_scaling_factor,
    out_idx_ptr, out_weight_ptr
):
    token_id = tl.program_id(0)

    # Compute logits for this token: [num_experts]
    logits = tl.zeros((num_experts,), dtype=tl.float32)
    # Dot product: sum_j hidden_states[token_id, j] * weight[e, j]
    for j in range(0, hidden_dim):
        h = tl.load(hidden_states + token_id * hidden_dim + j)
        for e in range(0, num_experts):
            w = tl.load(weight + e * hidden_dim + j)
            logits[e] += h * w

    # Apply sigmoid
    for e in range(0, num_experts):
        logits[e] = 1.0 / (1.0 + tl.exp(-logits[e]))

    # Add expert bias
    for e in range(0, num_experts):
        bias_e = tl.load(expert_bias + e)
        logits[e] += bias_e

    # Group-limited top-k logic
    experts_per_group = 32
    n_group = 8

    # Compute group scores: for each group g, find top-2 within [g*32 : (g+1)*32], sum them
    group_scores = tl.zeros((n_group,), dtype=tl.float32)
    for g in range(0, n_group):
        top1 = -float('inf')
        top1_idx = -1
        top2 = -float('inf')
        top2_idx = -1
        group_start = g * experts_per_group
        for e in range(group_start, group_start + experts_per_group):
            e_val = logits[e]
            if e_val > top1:
                top2 = top1
                top2_idx = top1_idx
                top1 = e_val
                top1_idx = e
            elif e_val > top2:
                top2 = e_val
                top2_idx = e
        group_scores[g] = top1 + top2

    # Select top-4 groups per token
    group_idx = tl.zeros((4,), dtype=tl.int32)
    group_scores_vec = group_scores
    for i in range(0, 4):
        max_val = -float('inf')
        max_idx = -1
        for g in range(0, n_group):
            if group_scores_vec[g] > max_val:
                max_val = group_scores_vec[g]
                max_idx = g
        group_idx[i] = max_idx
        # Invalidate selected group score
        group_scores_vec[max_idx] = -float('inf')

    # Build group_mask [n_group] (1 for selected, 0 otherwise)
    group_mask = tl.zeros((n_group,), dtype=tl.int32)
    for i in range(0, 4):
        group_mask[group_idx[i]] = 1

    # Expand group_mask to per-expert mask: if group_mask[g] == 0, set masked_score[e] = -inf for e in this group
    masked_scores = tl.zeros((num_experts,), dtype=tl.float32)
    for e in range(0, num_experts):
        g = e // experts_per_group  # group index for this expert
        if group_mask[g] == 0:
            masked_scores[e] = -float('inf')
        else:
            masked_scores[e] = logits[e]

    # Select top-8 from masked_scores (duplicates allowed)
    selected = tl.zeros((8,), dtype=tl.int32)
    selected_scores = tl.zeros((8,), dtype=tl.float32)
    for i in range(0, 8):
        max_val = -float('inf')
        max_idx = -1
        for e in range(0, num_experts):
            if masked_scores[e] > max_val:
                max_val = masked_scores[e]
                max_idx = e
        selected[i] = max_idx
        selected_scores[i] = max_val

    # Normalize using original selected_scores (gather from logits)
    sum_selected = 0.0
    for i in range(0, 8):
        sum_selected += selected_scores[i]
    eps = 1e-20
    norm = 1.0 / (sum_selected + eps)

    # Store outputs: indices and normalized scores * routed_scaling_factor
    for i in range(0, 8):
        tl.store(out_idx_ptr + token_id * 8 + i, selected[i])
        weight_val = (selected_scores[i] * norm) * routed_scaling_factor
        tl.store(out_weight_ptr + token_id * 8 + i, weight_val)


def run(*args):
    return ModelNew()(*args)
