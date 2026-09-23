import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Inputs:
        # hidden_states: [num_tokens, hidden_dim], float32, on device
        # weight: [num_experts, hidden_dim], float32, on device (num_experts=256)
        # expert_bias: [num_experts], float32, on device
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Allocate outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: each program handles one token and computes everything
        grid = (num_tokens,)
        run_kernel[grid](hidden_states, weight, expert_bias, routed_scaling_factor, topk_idx, topk_weight)

        return topk_idx, topk_weight


@triton.jit
def run_kernel(
    hidden_ptr,      # *float32, [num_tokens, hidden_dim]
    weight_ptr,      # *float32, [num_experts, hidden_dim]
    bias_ptr,        # *float32, [num_experts]
    scaling,         # float32
    out_idx_ptr,     # *int32,   [num_tokens, 8]
    out_weight_ptr,  # *float32, [num_tokens, 8]
    hidden_dim: tl.constexpr,
    num_experts: tl.constexpr,
):
    pid_token = tl.program_id(0)
    hidden_row_ptr = hidden_ptr + pid_token * hidden_dim

    # Compute scores[token, e] for all e: logits
    scores = tl.zeros([num_experts], dtype=tl.float32)
    for e in range(0, num_experts):
        total = 0.0
        for j in range(0, hidden_dim):
            h = tl.load(hidden_row_ptr + j)
            w = tl.load(weight_ptr + e * hidden_dim + j)
            total += h * w
        scores[e] = total

    # Apply sigmoid
    for e in range(0, num_experts):
        scores[e] = 1.0 / (1.0 + tl.exp(-scores[e]))

    # Add expert bias
    bias_vals = tl.load(bias_ptr + tl.arange(0, num_experts))
    scores = scores + bias_vals

    # Reshape into groups [8, 32] and compute top-2 per group
    group_scores = tl.zeros([8], dtype=tl.float32)
    for g in range(0, 8):
        start = g * 32
        end = start + 32
        group = scores[start:end]
        top1 = tl.max(group, axis=0)
        tmp = tl.where(group == top1, -1e30, group)
        top2 = tl.max(tmp, axis=0)
        group_scores[g] = top1 + top2

    # Select top-4 groups per token
    top4_idx = tl.zeros([4], dtype=tl.int32) - 1
    top4_vals = tl.full([4], -1e30, dtype=tl.float32)
    for i in range(0, 8):
        if group_scores[i] > top4_vals[0]:
            for k in range(3, -1, -1):
                if k == 0:
                    top4_vals[3] = group_scores[i]
                    top4_idx[3] = i
                    break
                elif top4_vals[k] > group_scores[i]:
                    top4_vals[k + 1] = top4_vals[k]
                    top4_idx[k + 1] = top4_idx[k]
                else:
                    top4_vals[k] = group_scores[i]
                    top4_idx[k] = i
                    break

    # Build group mask [num_tokens, 8] with 1 for selected groups, 0 otherwise
    group_mask = tl.zeros([8], dtype=tl.int32)
    for i in range(0, 4):
        group_mask[top4_idx[i]] = 1

    # Expand to per-expert mask: mask_exp[e] = 1 if e belongs to any selected group, else 0
    mask_exp = tl.zeros([num_experts], dtype=tl.int32)
    for e in range(0, num_experts):
        group_id = e // 32
        if group_id >= 0 and group_id < 8 and group_mask[group_id] == 1:
            mask_exp[e] = 1

    # Apply mask to scores_for_routing: non-selected group elements become -inf
    scores_for_routing = scores  # recompute exact scores for mask
    masked_scores = tl.where(mask_exp == 1, scores_for_routing, -1e30)

    # Select top-8 from masked scores
    top8_idx = tl.zeros([8], dtype=tl.int32) - 1
    top8_vals = tl.full([8], -1e30, dtype=tl.float32)
    for e in range(0, num_experts):
        if masked_scores[e] > top8_vals[0]:
            val = masked_scores[e]
            idx = e
            for k in range(7, -1, -1):
                if k == 0:
                    top8_vals[7] = val
                    top8_idx[7] = idx
                    break
                elif top8_vals[k] > val:
                    top8_vals[k + 1] = top8_vals[k]
                    top8_idx[k + 1] = top8_idx[k]
                else:
                    top8_vals[k] = val
                    top8_idx[k] = idx
                    break

    # Gather original logits for those selected indices and normalize by sum(original_selected_scores + 1e-20)
    original_logits = tl.zeros([num_experts], dtype=tl.float32)
    for e in range(0, num_experts):
        total = 0.0
        for j in range(0, hidden_dim):
            h = tl.load(hidden_row_ptr + j)
            w = tl.load(weight_ptr + e * hidden_dim + j)
            total += h * w
        original_logits[e] = total

    # Compute normalized weights
    denom = 0.0
    for k in range(0, 8):
        if top8_idx[k] >= 0:
            denom += original_logits[top8_idx[k]] + 1e-20

    # Store outputs
    base = pid_token * 8
    for k in range(0, 8):
        out_idx_off = out_idx_ptr + base + k
        out_w_off = out_weight_ptr + base + k
        tl.store(out_idx_off, top8_idx[k])
        contrib = original_logits[top8_idx[k]] + 1e-20
        weight = (contrib / denom) * scaling
        tl.store(out_w_off, weight)


# This Triton kernel is invoked from ModelNew.forward. It performs all computations and writes outputs.
# Outputs: topk_idx [num_tokens, 8] int32 and topk_weight [num_tokens, 8] float32, matching the original logic.


def run(*args):
    return ModelNew()(*args)
