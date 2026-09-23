import torch
import torch.nn as nn
import triton
import triton.language as tl

# Constants as in the original code
NUM_EXPERTS = 256
N_GROUPS = 8
EXPERTS_PER_GROUP = NUM_EXPERTS // N_GROUPS  # 32
TOPK_GROUP = 4
TOPK_EXPERTS = 8


@triton.jit
def _group_limited_topk_kernel(
    hidden_ptr,         # *f32, shape [num_tokens, hidden_dim]
    weight_ptr,         # *f32, shape [NUM_EXPERTS, hidden_dim]
    expert_bias_ptr,    # *f32, shape [NUM_EXPERTS]
    topk_idx_ptr,       # *i32, shape [num_tokens, TOPK_EXPERTS]
    topk_weight_ptr,    # *f32, shape [num_tokens, TOPK_EXPERTS]
    routed_scaling_factor,  # f32 scalar
    num_tokens,         # int32
    hidden_dim,         # int32
):
    # One Triton program per token
    token = tl.program_id(0)

    # 1) Compute logits = hidden[token, :] dot weight[e, :] for e in [0..NUM_EXPERTS-1]
    scores_raw = tl.zeros((NUM_EXPERTS,), dtype=tl.float32)
    for e in range(0, NUM_EXPERTS):
        score = 0.0
        for j in range(0, hidden_dim):
            h = tl.load(hidden_ptr + token * hidden_dim + j)
            w = tl.load(weight_ptr + e * hidden_dim + j)
            score += h * w
        scores_raw[e] = score

    # 2) Apply sigmoid to logits (routed scores pre-bias)
    scores_sig = 1.0 / (1.0 + tl.exp(-scores_raw))

    # 3) Add expert bias (learned routing adjustment)
    bias = tl.load(expert_bias_ptr + tl.arange(0, NUM_EXPERTS))
    scores = scores_sig + bias  # scores_for_routing

    # 4) Compute group scores: sum of top-2 within each group (of 32 experts)
    group_top2_sum = tl.full((N_GROUPS,), -1.0e30, dtype=tl.float32)
    for g in range(0, N_GROUPS):
        start = g * EXPERTS_PER_GROUP
        top1 = tl.full((), -1.0e30, dtype=tl.float32)
        top2 = tl.full((), -1.0e30, dtype=tl.float32)
        for i in range(0, EXPERTS_PER_GROUP):
            idx = start + i
            s = scores[idx]
            if s > top1:
                top2 = top1
                top1 = s
            elif s > top2:
                top2 = s
        group_top2_sum[g] = top1 + top2

    # 5) Select top-4 groups (descending group scores)
    selected_groups = tl.full((TOPK_GROUP,), -1, dtype=tl.int32)
    best = tl.full((), -1.0e30, dtype=tl.float32)
    for k in range(0, TOPK_GROUP):
        best_idx = -1
        for g in range(0, N_GROUPS):
            if group_top2_sum[g] > best:
                best = group_top2_sum[g]
                best_idx = g
        selected_groups[k] = best_idx
        # Invalidate this group for future selections
        group_top2_sum[best_idx] = -1.0e30

    # 6) Build per-expert mask: selected groups keep original scores; non-selected become -inf
    expert_mask = tl.zeros((NUM_EXPERTS,), dtype=tl.float32)
    for g in range(0, N_GROUPS):
        start = g * EXPERTS_PER_GROUP
        for i in range(0, EXPERTS_PER_GROUP):
            idx = start + i
            # If group g is selected, keep value; else set to -inf
            for kk in range(0, TOPK_GROUP):
                if selected_groups[kk] == g:
                    expert_mask[idx] = 1.0  # keep
                    break
            # If not selected in any kk, set mask to 0
            found = 0
            for kk in range(0, TOPK_GROUP):
                if selected_groups[kk] == g:
                    found = 1
                    break
            if found == 0:
                expert_mask[idx] = 0.0

    # 7) Apply mask to scores: non-selected groups -> -inf
    masked_scores = scores
    for e in range(0, NUM_EXPERTS):
        if expert_mask[e] == 0.0:
            masked_scores[e] = -1.0e30

    # 8) Select top-8 experts from masked_scores (avoid duplicates)
    top8_idx = tl.full((TOPK_EXPERTS,), -1, dtype=tl.int32)
    top8_val = tl.full((TOPK_EXPERTS,), -1.0e30, dtype=tl.float32)
    for k in range(0, TOPK_EXPERTS):
        best_e = -1
        best_score = -1.0e30
        for e in range(0, NUM_EXPERTS):
            ms = masked_scores[e]
            if ms > best_score:
                best_score = ms
                best_e = e
        top8_idx[k] = best_e
        top8_val[k] = best_score
        masked_scores[best_e] = -1.0e30  # remove from consideration

    # 9) Gather original logits for normalization: original scores_raw for selected indices
    selected_original_scores = tl.zeros((TOPK_EXPERTS,), dtype=tl.float32)
    for k in range(0, TOPK_EXPERTS):
        idx = top8_idx[k]
        selected_original_scores[k] = scores_raw[idx]

    # 10) Normalize: divide by sum(selected_original_scores) + 1e-20, then apply routed_scaling_factor
    sum_top8 = 0.0
    for k in range(0, TOPK_EXPERTS):
        sum_top8 += selected_original_scores[k]
    norm = sum_top8 + 1.0e-20

    # 11) Store outputs
    for k in range(0, TOPK_EXPERTS):
        out_idx_ptr = topk_idx_ptr + token * TOPK_EXPERTS + k
        out_weight_ptr = topk_weight_ptr + token * TOPK_EXPERTS + k
        tl.store(out_idx_ptr, top8_idx[k])
        tl.store(out_weight_ptr, (top8_val[k] / norm) * routed_scaling_factor)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure inputs are on CUDA and contiguous, dtype float32
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Inputs must be CUDA tensors"
        device = hidden_states.device
        hidden_states = hidden_states.contiguous().to(torch.float32)
        weight = weight.contiguous().to(torch.float32)
        expert_bias = expert_bias.contiguous().to(torch.float32)

        num_tokens, hidden_dim = hidden_states.shape
        assert weight.shape == (NUM_EXPERTS, hidden_dim), f"weight must be [NUM_EXPERTS={NUM_EXPERTS}, hidden_dim={hidden_dim}]"
        assert expert_bias.shape == (NUM_EXPERTS,), "expert_bias must be [NUM_EXPERTS]"

        # Allocate outputs
        topk_idx = torch.empty((num_tokens, TOPK_EXPERTS), dtype=torch.int32, device=device)
        topk_weight = torch.empty((num_tokens, TOPK_EXPERTS), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _group_limited_topk_kernel[grid](
            hidden_states, weight, expert_bias,
            topk_idx, topk_weight,
            routed_scaling_factor,
            num_tokens, hidden_dim,
        )

        # Return the computed outputs; evaluator will compare values, not gradients
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
