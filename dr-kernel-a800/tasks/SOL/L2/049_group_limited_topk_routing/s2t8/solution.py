import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim]
        # weight: [num_experts, hidden_dim] (num_experts = 256)
        # expert_bias: [num_experts]
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"
        assert weight.shape[1] == hidden_dim, "weight's second dim must match hidden_states' second dim"
        assert expert_bias.shape[0] == num_experts, "expert_bias size must match num_experts"

        # Allocate outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _group_limited_topk_kernel[grid](
            hidden_ptr=hidden_states,
            weight_ptr=weight,
            bias_ptr=expert_bias,
            routed_scale=routed_scaling_factor,
            N=num_tokens,
            D=hidden_dim,
            E=num_experts,
            topk_idx_ptr=topk_idx,
            topk_weight_ptr=topk_weight,
        )

        # Return results
        return topk_idx, topk_weight


@triton.jit
def _group_limited_topk_kernel(
    hidden_ptr,          # *f32, [N, D]
    weight_ptr,          # *f32, [E, D]
    bias_ptr,            # *f32, [E]
    routed_scale,        # f32 scalar
    N,                   # int32: num_tokens
    D,                   # int32: hidden_dim
    E,                   # int32: num_experts (256)
    topk_idx_ptr,        # *i32, [N, 8]
    topk_weight_ptr,     # *f32, [N, 8]
):
    # One Triton program per token
    token = tl.program_id(0)

    # 1) Compute logits via dot product: scores[e] = sum_j hidden[token, j] * weight[e, j]
    scores = tl.zeros((E,), dtype=tl.float32)
    for j in range(0, D):
        h_j = tl.load(hidden_ptr + token * D + j)
        for e in range(0, E):
            w_ej = tl.load(weight_ptr + e * D + j)
            scores[e] += h_j * w_ej

    # 2) Apply sigmoid
    for e in range(0, E):
        scores[e] = 1.0 / (1.0 + tl.exp(-scores[e]))

    # 3) Add expert bias
    for e in range(0, E):
        b_e = tl.load(bias_ptr + e)
        scores[e] += b_e  # scores_for_routing

    # 4) Reshape into groups and compute top-2 per group, sum to get group scores
    EPG = E // 8  # 32
    group_top2_sum = tl.zeros((8,), dtype=tl.float32)
    # We will track the two best per group in two passes (simple scalar loop)
    for g in range(0, 8):
        start = g * EPG
        best1 = -1.0e30
        best2 = -1.0e30
        for e in range(start, start + EPG):
            v = scores[e]
            if v > best1:
                best2 = best1
                best1 = v
            elif v > best2:
                best2 = v
        group_top2_sum[g] = best1 + best2

    # 5) Select top-4 groups per token
    group_idx = tl.zeros((4,), dtype=tl.int32)  # indices 0..7
    group_scores = group_top2_sum  # [8]
    # Selection via scalar loop (repeated top extraction)
    for r in range(0, 4):
        best_val = -1.0e30
        best_g = -1
        for g in range(0, 8):
            if group_scores[g] > best_val:
                best_val = group_scores[g]
                best_g = g
        group_idx[r] = best_g
        # Mark selected by setting its score to -inf
        group_scores[best_g] = -1.0e30

    # 6) Build group mask [8] (1 for selected groups, 0 otherwise)
    group_mask = tl.zeros((8,), dtype=tl.int32)
    for r in range(0, 4):
        group_mask[group_idx[r]] = 1

    # Now apply mask to scores_for_routing: non-selected groups -> -inf
    masked_scores = scores  # copy to modify
    for g in range(0, 8):
        if group_mask[g] == 0:
            start = g * EPG
            for e in range(start, start + EPG):
                masked_scores[e] = -1.0e30

    # 7) Select top-8 from masked scores (duplicates allowed)
    selected_idx = tl.zeros((8,), dtype=tl.int32)
    selected_val = tl.zeros((8,), dtype=tl.float32)
    for r in range(0, 8):
        best_val = -1.0e30
        best_e = 0
        # Scan all experts to find max
        for e in range(0, E):
            if masked_scores[e] > best_val:
                best_val = masked_scores[e]
                best_e = e
        selected_idx[r] = best_e
        selected_val[r] = best_val

    # 8) Gather original logits for normalization and compute normalized weights
    original_selected = tl.zeros((8,), dtype=tl.float32)
    sum_original = 0.0
    for r in range(0, 8):
        e = selected_idx[r]
        original_selected[r] = scores[e]
        sum_original += original_selected[r]
    # Avoid division by zero
    sum_original = tl.maximum(sum_original, 1e-20)
    normalized = original_selected * routed_scale / sum_original

    # 9) Store outputs
    base_out = topk_idx_ptr + token * 8
    for r in range(0, 8):
        tl.store(base_out + r, selected_idx[r])
        tl.store(topk_weight_ptr + token * 8 + r, normalized[r])


def run(*args):
    return ModelNew()(*args)
