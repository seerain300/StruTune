import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors."
        assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_dim]"
        assert weight.dim() == 2, "weight must be [num_experts, hidden_dim]"
        assert expert_bias.dim() == 1, "expert_bias must be [num_experts]"
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Allocate outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _group_topk_routing_kernel[grid](
            hidden_states, weight, expert_bias, topk_idx, topk_weight, routed_scaling_factor, num_tokens, hidden_dim, 256
        )

        return topk_idx, topk_weight


@triton.jit
def _group_topk_routing_kernel(
    hidden_ptr,     # *float32, [num_tokens, hidden_dim]
    weight_ptr,     # *float32, [num_experts, hidden_dim]
    bias_ptr,       # *float32, [num_experts]
    out_idx_ptr,    # *int32,   [num_tokens, 8]
    out_wgt_ptr,    # *float32, [num_tokens, 8]
    scale,          # float32
    N_TOKENS,       # int32
    HIDDEN_DIM,     # int32
    NUM_EXPERTS     # int32 (256)
):
    pid = tl.program_id(axis=0)  # token index

    # Compute logits: scores[e] = sum_j hidden[pid, j] * weight[e, j]
    scores = tl.zeros((NUM_EXPERTS,), dtype=tl.float32)
    for j in range(0, HIDDEN_DIM):
        h = tl.load(hidden_ptr + pid * HIDDEN_DIM + j)  # scalar
        for e in range(0, NUM_EXPERTS):
            w = tl.load(weight_ptr + e * HIDDEN_DIM + j)  # scalar
            scores[e] += h * w

    # Apply sigmoid
    scores = 1.0 / (1.0 + tl.exp(-scores))

    # Add expert bias
    for e in range(0, NUM_EXPERTS):
        b = tl.load(bias_ptr + e)
        scores[e] += b

    # Reshape into 8 groups of 32 and compute top-2 per group
    GROUPS = 8
    EXPERTS_PER_GROUP = NUM_EXPERTS // GROUPS  # 32
    group_scores = tl.zeros((GROUPS,), dtype=tl.float32)

    for g in range(0, GROUPS):
        start = g * EXPERTS_PER_GROUP
        # top-2 selection via simple loop
        top1 = -float('inf')
        top2 = -float('inf')
        for e in range(0, EXPERTS_PER_GROUP):
            v = scores[start + e]
            if v > top1:
                top2 = top1
                top1 = v
            elif v > top2:
                top2 = v
        group_scores[g] = top1 + top2

    # Select top-4 groups per token (store as indices)
    top_group_idx = tl.zeros((4,), dtype=tl.int32)
    for r in range(0, 4):
        max_val = -float('inf')
        chosen_group = -1
        for g in range(0, GROUPS):
            if group_scores[g] > max_val:
                max_val = group_scores[g]
                chosen_group = g
        top_group_idx[r] = chosen_group
        # avoid re-selection
        group_scores[chosen_group] = -float('inf')

    # Build group_mask: 1 for selected groups, 0 otherwise
    group_mask = tl.zeros((GROUPS,), dtype=tl.int32)
    for r in range(0, 4):
        group_mask[top_group_idx[r]] = 1

    # Expand group_mask to per-expert mask and apply: non-selected groups' scores -> -inf
    scores_masked = scores
    for g in range(0, GROUPS):
        if group_mask[g] == 0:
            start = g * EXPERTS_PER_GROUP
            # set those 32 experts' scores to -inf
            for e in range(0, EXPERTS_PER_GROUP):
                scores_masked[start + e] = -float('inf')

    # Select top-8 from masked scores (duplicates allowed)
    selected_idx = tl.zeros((8,), dtype=tl.int32)
    for r in range(0, 8):
        max_val = -float('inf')
        chosen_e = -1
        for e in range(0, NUM_EXPERTS):
            v = scores_masked[e]
            if v > max_val:
                max_val = v
                chosen_e = e
        selected_idx[r] = chosen_e
        # avoid re-selection
        scores_masked[chosen_e] = -float('inf')

    # Gather original logits for selected indices (pre-mask), normalize, apply scale
    orig_scores = scores
    chosen_vals = tl.zeros((8,), dtype=tl.float32)
    for r in range(0, 8):
        e = selected_idx[r]
        v = orig_scores[e]
        chosen_vals[r] = v

    denom = tl.sum(chosen_vals + 1e-20, axis=0)
    normalized = (chosen_vals / denom) * scale

    # Store outputs: [num_tokens, 8]
    for r in range(0, 8):
        tl.store(out_idx_ptr + pid * 8 + r, selected_idx[r])
        tl.store(out_wgt_ptr + pid * 8 + r, normalized[r])


def run(*args):
    return ModelNew()(*args)
