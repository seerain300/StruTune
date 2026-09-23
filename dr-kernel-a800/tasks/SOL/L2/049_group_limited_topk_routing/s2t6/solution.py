import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-only implementation of the original routing logic.
        Inputs:
          - hidden_states: [num_tokens, hidden_dim], float32, device
          - weight: [num_experts, hidden_dim], float32, device (num_experts=256)
          - expert_bias: [num_experts], float32, device
          - routed_scaling_factor: float scalar
        Outputs:
          - topk_idx: [num_tokens, 8], int32
          - topk_weight: [num_tokens, 8], float32
        """
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Allocate outputs on device
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _group_limited_topk_kernel[grid](
            hidden_states, weight, expert_bias,
            topk_idx, topk_weight,
            num_tokens, hidden_dim, num_experts,
            routed_scaling_factor
        )

        return topk_idx, topk_weight


@triton.jit
def _group_limited_topk_kernel(
    hidden_ptr,         # *f32, [num_tokens, hidden_dim]
    weight_ptr,         # *f32, [num_experts, hidden_dim]
    expert_bias_ptr,    # *f32, [num_experts]
    topk_idx_ptr,       # *i32, [num_tokens, 8]
    topk_weight_ptr,    # *f32, [num_tokens, 8]
    num_tokens,          # int32 (unused, but kept for clarity)
    hidden_dim,          # int32
    num_experts,         # int32 (256)
    routed_scaling_factor,  # f32
):
    token = tl.program_id(0)

    # 1) Compute logits = sigmoid(dot(hidden[token, :], weight[e, :])) for all e
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    for e in range(0, num_experts):
        s = 0.0
        # dot product over hidden_dim
        for d in range(0, hidden_dim):
            h = tl.load(hidden_ptr + token * hidden_dim + d)
            w = tl.load(weight_ptr + e * hidden_dim + d)
            s += h * w
        scores[e] = 1.0 / (1.0 + tl.exp(-s))

    # 2) Add expert bias to form scores_for_routing
    for e in range(0, num_experts):
        scores[e] += tl.load(expert_bias_ptr + e)

    # 3) Group-limited top-k: compute top-2 per group and sum to form group scores [8]
    group_top2_sum = tl.zeros((8,), dtype=tl.float32)
    for g in range(0, 8):
        start = g * 32
        # scan this group's 32 experts and find top-2
        best0 = -1.0e30
        best1 = -1.0e30
        for i in range(0, 32):
            idx = start + i
            val = scores[idx]
            if val > best0:
                best1 = best0
                best0 = val
            elif val > best1:
                best1 = val
        group_top2_sum[g] = best0 + best1

    # 4) Select top-4 groups based on group_top2_sum
    group_idx = tl.zeros((4,), dtype=tl.int32)
    group_scores_vec = group_top2_sum
    for r in range(0, 4):
        best_val = -1.0e30
        best_g = 0
        for g in range(0, 8):
            if group_scores_vec[g] > best_val:
                best_val = group_scores_vec[g]
                best_g = g
        # mark selected
        group_scores_vec = tl.where(tl.arange(0, 8) == best_g, -1.0e30, group_scores_vec)
        group_idx[r] = best_g

    # 5) Build score_mask [num_experts]: 1 for selected groups, 0 otherwise
    score_mask = tl.zeros((num_experts,), dtype=tl.float32)
    for r in range(0, 4):
        g = group_idx[r]
        start = g * 32
        for i in range(0, 32):
            idx = start + i
            score_mask[idx] = 1.0

    # 6) Apply mask to scores_for_routing: non-selected -> -inf
    masked_scores = scores
    for e in range(0, num_experts):
        if score_mask[e] == 0.0:
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

    # 8) Gather original logits for normalization and compute normalized weights
    original_selected = tl.zeros((8,), dtype=tl.float32)
    sum_original = 0.0
    for r in range(0, 8):
        e = selected_idx[r]
        original_selected[r] = scores[e]
        sum_original += original_selected[r]
    sum_original = tl.maximum(sum_original, 1e-20)
    normalized = original_selected * routed_scaling_factor / sum_original

    # 9) Store outputs
    base_out = topk_idx_ptr + token * 8
    for r in range(0, 8):
        tl.store(base_out + r, selected_idx[r])
        tl.store(topk_weight_ptr + token * 8 + r, normalized[r])


def run(*args):
    return ModelNew()(*args)
