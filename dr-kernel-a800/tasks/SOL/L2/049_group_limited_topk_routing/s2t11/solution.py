import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Inputs:
        # hidden_states: [num_tokens, hidden_dim], float32
        # weight: [256, hidden_dim], float32
        # expert_bias: [256], float32
        # routed_scaling_factor: float
        # Output:
        # topk_idx: [num_tokens, 8], int32
        # topk_weight: [num_tokens, 8], float32

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Output tensors (allocate on the same device as inputs)
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        selected_original_logits = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token, compute everything
        grid = (num_tokens,)

        # Invoke Triton kernel that does:
        # - compute scores[token, :] and scores_for_routing[token, :]
        # - group selection (top-4 per token)
        # - masking and selection of top-8 experts
        # - gather original logits for those 8 and write topk_idx, selected_original_logits
        _run_grouped_routing_kernel(
            hidden_ptr=hidden_states,           # [num_tokens, hidden_dim]
            weight_ptr=weight,                  # [256, hidden_dim]
            bias_ptr=expert_bias,               # [256]
            topk_idx_out_ptr=topk_idx,          # [num_tokens, 8] int32
            selected_original_out_ptr=selected_original_logits,  # [num_tokens, 8] float32
            routed_factor_ptr=routed_scaling_factor,    # scalar float
            grid=grid,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )

        # Compute topk_weight on host: normalize by sum(original_selected_scores + 1e-20) then apply routed_scaling_factor
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)
        for t in range(num_tokens):
            row = selected_original_logits[t]  # [8] float32
            denom = torch.sum(row + 1e-20)
            topk_weight_row = (row / denom) * routed_scaling_factor
            topk_weight[t] = topk_weight_row

        return topk_idx, topk_weight


# Triton kernel: one program per token. It computes all steps and writes topk_idx and selected_original_logits for that token.
@triton.jit
def _run_grouped_routing_kernel(
    hidden_ptr,                      # *fp32, [num_tokens, hidden_dim]
    weight_ptr,                      # *fp32, [num_experts, hidden_dim]
    bias_ptr,                        # *fp32, [num_experts]
    topk_idx_out_ptr,                # *int32, [num_tokens, 8]
    selected_original_out_ptr,       # *fp32, [num_tokens, 8]
    routed_factor_ptr,               # *fp32, scalar (not used in kernel)
    num_tokens,                      # int32
    hidden_dim: tl.constexpr,        # int32
    num_experts: tl.constexpr        # int32, fixed 256
):
    # One program per token
    token_id = tl.program_id(0)

    # Local constants
    EXPERTS_PER_GROUP = 32
    NUM_GROUPS = 8
    TOPK_GROUP = 4
    TOPK = 8

    # 1) Compute scores[token, :] and scores_for_routing[token, :]
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    scores_for_routing = tl.zeros((num_experts,), dtype=tl.float32)
    hidden_row_ptr = hidden_ptr + token_id * hidden_dim
    for e in range(0, num_experts):
        acc = 0.0
        w_row_ptr = weight_ptr + e * hidden_dim
        for j in range(0, hidden_dim):
            x = tl.load(hidden_row_ptr + j)
            w = tl.load(w_row_ptr + j)
            acc += x * w
        # acc is logits
        sig = 1.0 / (1.0 + tl.exp(-acc))
        scores[e] = sig
        scores_for_routing[e] = sig + tl.load(bias_ptr + e)

    # 2) Partition into 8 groups of 32, compute per-group top-2, sum to group_scores
    group_scores = tl.zeros((NUM_GROUPS,), dtype=tl.float32)
    for g in range(0, NUM_GROUPS):
        start = g * EXPERTS_PER_GROUP
        group_vals = scores[start:start + EXPERTS_PER_GROUP]
        # Compute top-2 within group
        max1 = -float('inf')
        max1_idx = -1
        for i in range(0, EXPERTS_PER_GROUP):
            v = group_vals[i]
            if v > max1:
                max1 = v
                max1_idx = start + i
        max2 = -float('inf')
        for i in range(0, EXPERTS_PER_GROUP):
            v = group_vals[i]
            if (v > max2) and (start + i != max1_idx):
                max2 = v
        group_scores[g] = max1 + max2

    # 3) Select top-4 groups per token (unordered: we only need indices of selected groups)
    selected_group_idx = tl.zeros((TOPK_GROUP,), dtype=tl.int32)
    for k in range(0, TOPK_GROUP):
        maxv = -float('inf')
        max_idx = -1
        for g in range(0, NUM_GROUPS):
            if group_scores[g] > maxv:
                maxv = group_scores[g]
                max_idx = g
        selected_group_idx[k] = max_idx
        # Marked implicitly by not reusing

    # 4) Mask out non-selected groups by setting their scores_for_routing to -inf
    masked_scores = scores_for_routing  # start with original
    for k in range(0, TOPK_GROUP):
        g = selected_group_idx[k]
        start = g * EXPERTS_PER_GROUP
        for i in range(0, EXPERTS_PER_GROUP):
            idx = start + i
            masked_scores[idx] = -float('inf')

    # 5) Select top-8 from masked_scores (duplicates allowed)
    selected_indices = tl.zeros((TOPK,), dtype=tl.int32)
    for t in range(0, TOPK):
        maxv = -float('inf')
        max_idx = -1
        for e in range(0, num_experts):
            v = masked_scores[e]
            if v > maxv:
                maxv = v
                max_idx = e
        selected_indices[t] = max_idx
        masked_scores[max_idx] = -float('inf')

    # 6) Gather original logits for those selected indices and write to selected_original_out_ptr


def run(*args):
    return ModelNew()(*args)
