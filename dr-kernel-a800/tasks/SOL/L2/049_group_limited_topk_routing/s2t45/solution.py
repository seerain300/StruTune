import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: per-token routing computation
# Each program handles one token. It computes:
# - logits per expert (dot product),
# - sigmoid + bias,
# - group top-2 and selection,
# - mask application,
# - final top-8 selection and normalized weights.
@triton.jit
def routing_kernel(
    hidden_ptr,        # *float32, [num_tokens, hidden_dim]
    weight_ptr,        # *float32, [num_experts, hidden_dim]
    bias_ptr,          # *float32, [num_experts]
    out_idx_ptr,       # *int32,   [num_tokens, 8]
    out_weight_ptr,    # *float32, [num_tokens, 8]
    num_tokens,        # int32
    hidden_dim,        # int32
    num_experts,       # int32 (must be 256)
    routed_scaling,    # float32
):
    pid = tl.program_id(axis=0)
    if pid >= num_tokens:
        return

    # Compute logits per expert: scores[token, e] = dot(hidden[token,:], weight[e,:])
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    for e in range(0, num_experts):
        total = tl.zeros((), dtype=tl.float32)
        for j in range(0, hidden_dim):
            h = tl.load(hidden_ptr + pid * hidden_dim + j)
            w = tl.load(weight_ptr + e * hidden_dim + j)
            total += h * w
        scores[e] = tl.sigmoid(total)

    # Add expert bias
    for e in range(0, num_experts):
        scores[e] += tl.load(bias_ptr + e)

    # Group top-2 per group (8 groups of 32)
    group_top2 = tl.zeros((8,), dtype=tl.float32)
    for g in range(0, 8):
        top1 = -float("inf")
        top2 = -float("inf")
        for kk in range(0, 32):
            e_idx = g * 32 + kk
            v = scores[e_idx]
            if v > top1:
                top2 = top1
                top1 = v
            elif v > top2:
                top2 = v
        group_top2[g] = top1 + top2

    # Select top-4 groups (descending)
    top4_group = tl.zeros((4,), dtype=tl.int32)  # indices in [0..7]
    top4_score = tl.zeros((4,), dtype=tl.float32)
    for g in range(0, 8):
        include = True
        for m in range(0, 4):
            # If any existing top4 entry is strictly greater than current group score, skip
            if (top4_score[m] > 0.0) and (top4_score[m] > group_top2[g]):
                include = False
                break
        if include:
            # Replace the smallest current score slot
            min_idx = 0
            min_score = top4_score[0]
            for m in range(1, 4):
                if (top4_score[m] == 0.0) or ((top4_score[m] < min_score) and (top4_group[m] != -1)):
                    min_idx = m
                    min_score = top4_score[m]
            top4_group[min_idx] = g
            top4_score[min_idx] = group_top2[g]

    # Build group_mask: 1.0 for selected groups, else 0.0
    group_mask = tl.zeros((8,), dtype=tl.float32)
    for m in range(0, 4):
        group_mask[top4_group[m]] = 1.0

    # Expand group_mask to per-expert mask: for each expert e, check if e//32 in selected groups
    masked_scores = tl.full((num_experts,), -float("inf"), dtype=tl.float32)
    for e in range(0, num_experts):
        e_group = e // 32
        keep = group_mask[e_group]  # float32 1.0 or 0.0
        if keep == 1.0:
            masked_scores[e] = scores[e]

    # Select top-8 from masked_scores (duplicates allowed)
    top8_vals = tl.zeros((8,), dtype=tl.float32)
    top8_idx = tl.zeros((8,), dtype=tl.int32)
    for m in range(0, 8):
        best_val = -float("inf")
        best_idx = -1
        for e in range(0, num_experts):
            if masked_scores[e] > best_val:
                best_val = masked_scores[e]
                best_idx = e
        top8_vals[m] = best_val
        top8_idx[m] = best_idx

    # Normalize by sum(original selected scores + 1e-20), then apply routed_scaling
    sum_sel = 0.0
    for m in range(0, 8):
        sum_sel += top8_vals[m]
    norm = sum_sel + 1e-20
    for m in range(0, 8):
        w = top8_vals[m] / norm * routed_scaling
        tl.store(out_weight_ptr + pid * 8 + m, w)
        tl.store(out_idx_ptr + pid * 8 + m, top8_idx[m])


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
        assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32

        # Ensure contiguous tensors
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()
        expert_bias = expert_bias.contiguous()

        # Allocate outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        routing_kernel[grid](
            hidden_states, weight, expert_bias,
            topk_idx, topk_weight,
            num_tokens, hidden_dim, num_experts,
            routed_scaling_factor,
            num_warps=4,  # reasonable default for scalar-heavy loops
        )
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
