import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: one program per token
# It computes:
# - scores_for_routing[e] = sigmoid(dot(hidden[token, :], weight[e, :])) + expert_bias[e]
# - per-token group top-2 sums
# - per-token top-4 groups
# - mask out non-selected groups by setting their scores to -inf
# - select top-8 from masked scores
# - recompute original logits for selected indices and write normalized topk_weight
@triton.jit
def compute_token_kernel(
    hidden_ptr,           # *float32, [num_tokens, hidden_dim]
    weight_ptr,           # *float32, [num_experts, hidden_dim]
    bias_ptr,             # *float32, [num_experts]
    num_tokens,           # int32
    hidden_dim,           # int32
    num_experts,          # int32
    routed_scaling_factor,  # float32
    topk_idx_ptr,         # *int32, [num_tokens, 8]
    topk_weight_ptr,      # *float32, [num_tokens, 8]
    selected_out_ptr      # *float32, [num_tokens, 8] (we won't use this in kernel; it exists for API symmetry)
):
    # Each program handles one token
    token_id = tl.program_id(0)

    # 1) Compute scores_for_routing[e] = sigmoid(dot) + bias
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    for e in range(0, num_experts):
        acc = 0.0
        hidden_row_ptr = hidden_ptr + token_id * hidden_dim
        w_row_ptr = weight_ptr + e * hidden_dim
        for j in range(0, hidden_dim):
            x = tl.load(hidden_row_ptr + j)
            w = tl.load(w_row_ptr + j)
            acc += x * w
        s = 1.0 / (1.0 + tl.exp(-acc))  # sigmoid
        b = tl.load(bias_ptr + e)
        scores[e] = s + b

    # 2) Group top-2 per group
    EXPERTS_PER_GROUP = 32
    NUM_GROUPS = 8
    group_scores = tl.zeros((NUM_GROUPS,), dtype=tl.float32)
    for g in range(0, NUM_GROUPS):
        start = g * EXPERTS_PER_GROUP
        group_vals = scores[start:start + EXPERTS_PER_GROUP]
        # Find top-2 in this group
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

    # 3) Select top-4 groups (unordered: we only need indices of selected groups)
    TOPK_GROUP = 4
    selected_group_idx = tl.zeros((TOPK_GROUP,), dtype=tl.int32)
    for k in range(0, TOPK_GROUP):
        maxv = -float('inf')
        max_idx = -1
        for g in range(0, NUM_GROUPS):
            if group_scores[g] > maxv:
                maxv = group_scores[g]
                max_idx = g
        selected_group_idx[k] = max_idx
        # Mark implicitly by not reusing

    # 4) Mask out non-selected groups by setting their scores_for_routing to -inf
    masked_scores = scores
    for k in range(0, TOPK_GROUP):
        g = selected_group_idx[k]
        start = g * EXPERTS_PER_GROUP
        for i in range(0, EXPERTS_PER_GROUP):
            idx = start + i
            masked_scores[idx] = -float('inf')

    # 5) Select top-8 from masked_scores (duplicates allowed)
    TOPK = 8
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

    # 6) Recompute original logits for those selected indices and write normalized topk_weight
    original_vals = tl.zeros((TOPK,), dtype=tl.float32)
    for t in range(0, TOPK):
        e = selected_indices[t]
        acc = 0.0
        hidden_row_ptr = hidden_ptr + token_id * hidden_dim
        w_row_ptr = weight_ptr + e * hidden_dim
        for j in range(0, hidden_dim):
            x = tl.load(hidden_row_ptr + j)
            w = tl.load(w_row_ptr + j)
            acc += x * w
        original_vals[t] = acc

    # Normalize by sum(original_vals + 1e-20) and apply routed_scaling_factor
    eps = 1e-20
    denom = 0.0
    for t in range(0, TOPK):
        denom += original_vals[t] + eps
    # Store normalized weights
    for t in range(0, TOPK):
        w = original_vals[t] / denom
        w = w * routed_scaling_factor
        # Write to topk_weight_ptr[token_id * TOPK + t]
        tl.store(topk_weight_ptr + token_id * TOPK + t, w)

    # 7) Write topk_idx
    # Write selected_indices to topk_idx_ptr[token_id * TOPK + t]
    for t in range(0, TOPK):
        tl.store(topk_idx_ptr + token_id * TOPK + t, selected_indices[t])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32, device
        # weight: [num_experts, hidden_dim], float32, device, num_experts=256
        # expert_bias: [num_experts], float32, device
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Allocate outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        triton.run(
            compute_token_kernel,
            grid=grid,
            hidden_ptr=hidden_states,
            weight_ptr=weight,
            bias_ptr=expert_bias,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            routed_scaling_factor=routed_scaling_factor,
            topk_idx_ptr=topk_idx,
            topk_weight_ptr=topk_weight,
            selected_out_ptr=topk_weight  # placeholder; not used in kernel
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
