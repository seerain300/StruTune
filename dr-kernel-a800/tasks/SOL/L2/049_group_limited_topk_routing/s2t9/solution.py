import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Inputs:
        # hidden_states: [num_tokens, hidden_dim], float32
        # weight: [num_experts, hidden_dim], float32 (num_experts=256)
        # expert_bias: [num_experts], float32
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"
        assert weight.shape[1] == hidden_dim, "weight's second dim must match hidden_states' second dim"

        # Allocate outputs (will be stored by Triton kernel)
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _group_limited_topk_kernel[grid](
            hidden_states,
            weight,
            expert_bias,
            topk_idx,
            topk_weight,
            num_tokens,
            hidden_dim,
            num_experts,
            routed_scaling_factor,
        )

        return topk_idx, topk_weight


@triton.jit
def _group_limited_topk_kernel(
    hidden_ptr,         # *f32, [num_tokens, hidden_dim]
    weight_ptr,         # *f32, [num_experts, hidden_dim]
    expert_bias_ptr,    # *f32, [num_experts]
    topk_idx_ptr,       # *i32, [num_tokens, 8]
    topk_weight_ptr,    # *f32, [num_tokens, 8]
    num_tokens,         # i32
    hidden_dim,         # i32
    num_experts,        # i32 (256)
    routed_scaling_factor,  # f32
):
    token = tl.program_id(0)

    # 1) Compute logits: scores[token, e] = dot(hidden[token, :], weight[e, :])
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    for j in range(0, hidden_dim):
        h = tl.load(hidden_ptr + token * hidden_dim + j)
        for e in range(0, num_experts):
            w = tl.load(weight_ptr + e * hidden_dim + j)
            scores[e] += h * w

    # 2) Apply sigmoid
    for e in range(0, num_experts):
        s = 1.0 / (1.0 + tl.exp(-scores[e]))
        scores[e] = s

    # 3) Add expert bias
    for e in range(0, num_experts):
        scores[e] += tl.load(expert_bias_ptr + e)

    # 4) Reshape to groups [8, 32] and compute top-2 per group, sum to get group scores
    group_top2_sum = tl.zeros((8,), dtype=tl.float32)
    # Loop over groups g, and within each group, loop over 32 experts
    for g in range(0, 8):
        base_e = g * 32
        vals = tl.zeros((32,), dtype=tl.float32)
        for i in range(0, 32):
            e = base_e + i
            vals[i] = scores[e]
        # top-2 within group
        # First, get max
        max1 = vals[0]
        idx1 = 0
        for i in range(1, 32):
            if vals[i] > max1:
                max1 = vals[i]
                idx1 = i
        # Second max (not equal to max1)
        max2 = -1.0e30
        for i in range(0, 32):
            if vals[i] > max2 and vals[i] != max1:
                max2 = vals[i]
        group_top2_sum[g] = max1 + max2

    # 5) Select top-4 groups
    # Scan groups and pick top-4 indices
    group_sel = tl.zeros((8,), dtype=tl.int32)
    for r in range(0, 4):
        best_val = -1.0e30
        best_g = 0
        for g in range(0, 8):
            if group_top2_sum[g] > best_val:
                best_val = group_top2_sum[g]
                best_g = g
        group_sel[r] = best_g

    # 6) Build group_mask [8] (1 for selected groups, 0 otherwise)
    group_mask = tl.zeros((8,), dtype=tl.int32)
    for r in range(0, 4):
        group_mask[group_sel[r]] = 1

    # 7) Apply masking to scores_for_routing: non-selected groups -> -inf
    masked_scores = scores
    for g in range(0, 8):
        if group_mask[g] == 0:
            base_e = g * 32
            for i in range(0, 32):
                e = base_e + i
                masked_scores[e] = -1.0e30

    # 8) Select top-8 from masked_scores (duplicates allowed)
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

    # 9) Gather original logits for normalization and compute normalized weights
    original_selected = tl.zeros((8,), dtype=tl.float32)
    sum_original = 0.0
    for r in range(0, 8):
        e = selected_idx[r]
        original_selected[r] = scores[e]
        sum_original += original_selected[r]
    sum_original = tl.maximum(sum_original, 1e-20)
    normalized = selected_val * routed_scaling_factor / sum_original

    # 10) Store outputs
    base_out = topk_idx_ptr + token * 8
    for r in range(0, 8):
        tl.store(base_out + r, selected_idx[r])
        tl.store(topk_weight_ptr + token * 8 + r, normalized[r])


def run(*args):
    return ModelNew()(*args)
