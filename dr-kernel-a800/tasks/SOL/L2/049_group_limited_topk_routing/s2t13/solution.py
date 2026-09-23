import torch
import torch.nn as nn
import triton
import triton.language as tl


# Constants used in the Triton kernel
NUM_GROUPS = 8
EXPERTS_PER_GROUP = 32
NUM_EXPERTS = 256
TOPK_GROUP = 4
TOPK = 8


@triton.jit
def _compute_routing_one_token(
    hidden_ptr,                      # *f32, [num_tokens, hidden_dim]
    weight_ptr,                      # *f32, [num_experts, hidden_dim]
    expert_bias_ptr,                 # *f32, [num_experts]
    topk_idx_ptr,                    # *i32, [num_tokens, TOPK] flattened row-major
    selected_original_logits_ptr,    # *f32, [num_tokens, TOPK] flattened row-major
    num_tokens: tl.constexpr,        # int
    num_experts: tl.constexpr,       # int, 256
    hidden_dim: tl.constexpr,        # int
    token_id: tl.constexpr,          # int in [0, num_tokens)
):
    # Load hidden vector for this token
    hidden_row_ptr = hidden_ptr + token_id * hidden_dim

    # 1) Compute scores_for_routing for all experts
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    for e in range(0, num_experts):
        acc = 0.0
        w_row_ptr = weight_ptr + e * hidden_dim
        for j in range(0, hidden_dim):
            x = tl.load(hidden_row_ptr + j)
            w = tl.load(w_row_ptr + j)
            acc += x * w
        sig = 1.0 / (1.0 + tl.exp(-acc))
        bias = tl.load(expert_bias_ptr + e)
        scores[e] = sig + bias

    # 2) Compute group_scores: sum of top-2 scores per group
    group_scores = tl.zeros((NUM_GROUPS,), dtype=tl.float32)
    for g in range(0, NUM_GROUPS):
        start = g * EXPERTS_PER_GROUP
        group_vals = tl.zeros((EXPERTS_PER_GROUP,), dtype=tl.float32)
        for i in range(0, EXPERTS_PER_GROUP):
            e_idx = start + i
            group_vals[i] = scores[e_idx]
        # Find top-2
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

    # 3) Select top-4 groups per token (unordered)
    selected_group_idx = tl.zeros((TOPK_GROUP,), dtype=tl.int32)
    for k in range(0, TOPK_GROUP):
        maxv = -float('inf')
        max_idx = -1
        for g in range(0, NUM_GROUPS):
            if group_scores[g] > maxv:
                maxv = group_scores[g]
                max_idx = g
        selected_group_idx[k] = max_idx

    # 4) Expand group mask to per-expert and set non-selected group elements to -inf
    masked_scores = scores  # start with original scores_for_routing
    for k in range(0, TOPK_GROUP):
        g = selected_group_idx[k]
        start = g * EXPERTS_PER_GROUP
        for i in range(0, EXPERTS_PER_GROUP):
            e_idx = start + i
            # For non-selected groups, set masked_scores[e_idx] to -inf
            pass

    # Explicit masking
    for e in range(0, num_experts):
        keep = 0
        for k in range(0, TOPK_GROUP):
            g = selected_group_idx[k]
            start = g * EXPERTS_PER_GROUP
            if (e >= start) and (e < start + EXPERTS_PER_GROUP):
                keep = 1
                break
        if keep == 0:
            masked_scores[e] = -float('inf')

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
        # Mark selected as -inf to avoid reselecting (not necessary here as we choose exactly TOPK, but harmless)
        masked_scores[max_idx] = -float('inf')

    # 6) Recompute original logits for selected indices and write outputs
    # Original logits = dot(hidden[token, :], weight[e, :])
    for t in range(0, TOPK):
        e = selected_indices[t]
        acc = 0.0
        w_row_ptr = weight_ptr + e * hidden_dim
        hidden_row_ptr = hidden_ptr + token_id * hidden_dim
        for j in range(0, hidden_dim):
            x = tl.load(hidden_row_ptr + j)
            w = tl.load(w_row_ptr + j)
            acc += x * w
        out_base = token_id * TOPK
        tl.store(selected_original_logits_ptr + out_base + t, acc)
        tl.store(topk_idx_ptr + out_base + t, e)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32
        # weight: [num_experts, hidden_dim], float32, num_experts=256
        # expert_bias: [num_experts], float32
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"
        assert weight.shape[1] == hidden_dim, "weight's second dimension must match hidden_dim"

        # Allocate outputs
        topk_idx = torch.empty((num_tokens, TOPK), dtype=torch.int32, device=hidden_states.device)
        selected_original_logits = torch.empty((num_tokens, TOPK), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _compute_routing_one_token[grid](
            hidden_states,                      # hidden_ptr
            weight,                             # weight_ptr
            expert_bias,                        # expert_bias_ptr
            topk_idx,                           # topk_idx_ptr
            selected_original_logits,           # selected_original_logits_ptr
            num_tokens=num_tokens,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
        )

        # Compute topk_weight on host: normalize original logits by sum(original_selected + 1e-20) and apply scaling factor
        denom = selected_original_logits + 1e-20            # [num_tokens, 8]
        denom = denom.sum(dim=1, keepdim=True)              # [num_tokens, 1]
        topk_weight = (selected_original_logits / denom) * routed_scaling_factor  # [num_tokens, 8]

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
