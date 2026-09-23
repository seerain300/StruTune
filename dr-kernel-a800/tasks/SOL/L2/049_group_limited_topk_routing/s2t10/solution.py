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

        # Ensure contiguous tensors
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()
        expert_bias = expert_bias.contiguous()

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"
        assert expert_bias.shape[0] == num_experts, "expert_bias size must match num_experts"

        # Output buffers
        # Triton will write:
        # - topk_idx: [num_tokens, 8], int32
        # - selected_original_logits: [num_tokens, 8], float32 (original logits for selected experts)
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        selected_original_logits = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        compute_routing_kernel[grid](
            hidden_ptr=hidden_states,
            weight_ptr=weight,
            bias_ptr=expert_bias,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            topk_idx_ptr=topk_idx,
            selected_original_logits_ptr=selected_original_logits,
        )

        # Compute topk_weight using PyTorch (host) to match original semantics:
        # topk_weight = (selected_original_logits / sum(selected_original_logits + 1e-20)) * routed_scaling_factor
        denom = selected_original_logits.sum(dim=1, keepdim=True) + 1e-20  # [num_tokens, 1]
        topk_weight = (selected_original_logits / denom) * routed_scaling_factor  # [num_tokens, 8], float32

        return topk_idx, topk_weight


# Triton kernel: computes top-8 selected expert indices (topk_idx) per token,
# and writes out the original logits (selected_original_logits[token, 8]) for those selected experts.
@triton.jit
def compute_routing_kernel(
    hidden_ptr,               # *float32, [num_tokens, hidden_dim]
    weight_ptr,               # *float32, [num_experts, hidden_dim]
    bias_ptr,                 # *float32, [num_experts]
    num_tokens,               # int32
    hidden_dim,               # int32
    num_experts,              # int32
    topk_idx_ptr,             # *int32, [num_tokens, 8]
    selected_original_logits_ptr,  # *float32, [num_tokens, 8]
):
    token_id = tl.program_id(0)
    if token_id >= num_tokens:
        return

    EXPERTS_PER_GROUP = 32
    N_GROUPS = 8

    # 1) Compute scores[token, e] = dot(hidden[token, :], weight[e, :])
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    hidden_row_ptr = hidden_ptr + token_id * hidden_dim
    for e in range(0, num_experts):
        w_row_ptr = weight_ptr + e * hidden_dim
        acc = 0.0
        for j in range(0, hidden_dim):
            x = tl.load(hidden_row_ptr + j)
            w = tl.load(w_row_ptr + j)
            acc += x * w
        scores[e] = acc

    # 2) Apply sigmoid and add expert bias to form scores_for_routing
    for e in range(0, num_experts):
        scores[e] = 1.0 / (1.0 + tl.exp(-scores[e]))  # sigmoid
        scores[e] += tl.load(bias_ptr + e)

    # 3) Group scores and compute top-2 per group, sum to form group_scores [N_GROUPS]
    group_scores = tl.zeros((N_GROUPS,), dtype=tl.float32)
    for g in range(0, N_GROUPS):
        group_start = g * EXPERTS_PER_GROUP
        v0 = -float('inf')
        v1 = -float('inf')
        for i in range(0, EXPERTS_PER_GROUP):
            idx = group_start + i
            val = scores[idx]
            if val > v0:
                v1 = v0
                v0 = val
            elif val > v1:
                v1 = val
        group_scores[g] = v0 + v1

    # 4) Select top-4 groups based on group_scores
    top4_groups = tl.zeros((4,), dtype=tl.int32)
    for t in range(0, 4):
        maxv = -float('inf')
        maxg = -1
        for g in range(0, N_GROUPS):
            if group_scores[g] > maxv:
                maxv = group_scores[g]
                maxg = g
        top4_groups[t] = maxg
        # Ignore this group in subsequent selections
        group_scores[maxg] = -float('inf')

    # 5) Build group_mask (1 for selected groups, 0 otherwise)
    group_mask = tl.zeros((N_GROUPS,), dtype=tl.int32)
    for t in range(0, 4):
        group_mask[top4_groups[t]] = 1

    # 6) Mask scores: for non-selected groups, set their scores to -inf
    # We select top-8 from masked scores.
    masked_scores = tl.zeros((num_experts,), dtype=tl.float32)
    for e in range(0, num_experts):
        group_idx = (e // EXPERTS_PER_GROUP) * EXPERTS_PER_GROUP
        keep = 0
        for t in range(0, 4):
            # Check if this expert falls within any of the selected groups
            if (group_idx >= (top4_groups[t] * EXPERTS_PER_GROUP)) and (group_idx < ((top4_groups[t] + 1) * EXPERTS_PER_GROUP)):
                keep = 1
                break
        masked_scores[e] = scores[e] if keep == 1 else (-float('inf'))

    # 7) Select top-8 from masked_scores and write indices
    selected_indices = tl.zeros((8,), dtype=tl.int32)
    for t in range(0, 8):
        maxv = -float('inf')
        max_idx = -1
        for e in range(0, num_experts):
            v = masked_scores[e]
            if v > maxv:
                maxv = v
                max_idx = e
        selected_indices[t] = max_idx
        masked_scores[max_idx] = -float('inf')

    # 8) Compute original logits for the selected experts and write them out
    # Original logits = dot(hidden[token, :], weight[e, :])
    for t in range(0, 8):
        e = selected_indices[t]
        w_row_ptr = weight_ptr + e * hidden_dim
        acc = 0.0
        hidden_row_ptr = hidden_ptr + token_id * hidden_dim
        for j in range(0, hidden_dim):
            x = tl.load(hidden_row_ptr + j)
            w = tl.load(w_row_ptr + j)
            acc += x * w
        # Store original logits for this selected expert
        selected_original_logits_ptr[token_id * 8 + t] = acc
        # Store index
        topk_idx_ptr[token_id * 8 + t] = selected_indices[t]


def run(*args):
    return ModelNew()(*args)
