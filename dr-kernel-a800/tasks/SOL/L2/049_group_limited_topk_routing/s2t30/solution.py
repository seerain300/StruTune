import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _grouped_routing_kernel(
    hidden_ptr,          # *f32, [num_tokens, hidden_dim]
    weight_ptr,          # *f32, [num_experts, hidden_dim]
    bias_ptr,            # *f32, [num_experts]
    out_idx_ptr,         # *i32, [num_tokens * 8] (flattened)
    out_weight_ptr,      # *f32, [num_tokens * 8] (flattened)
    num_tokens: tl.int32,
    hidden_dim: tl.int32,
    num_experts: tl.int32,             # must be 256
    scaling: tl.float32,
    EXPERTS_PER_GROUP: tl.constexpr,   # 32
    NUM_GROUPS: tl.constexpr,          # 8
    TOPK_GROUP: tl.constexpr,          # 4
    TOPK: tl.constexpr,                # 8
):
    # One program instance per token
    token = tl.program_id(0)

    # 1) Compute logits scores[token, e] = dot(hidden[token, :], weight[e, :]) for all e
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    hs = tl.load(hidden_ptr + token * hidden_dim + tl.arange(0, hidden_dim), mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)

    for e in range(0, num_experts):
        w = tl.load(weight_ptr + e * hidden_dim + tl.arange(0, hidden_dim), mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
        scores[e] = tl.sum(hs * w, axis=0)

    # 2) Sigmoid and add bias
    scores = 1.0 / (1.0 + tl.exp(-scores))
    bias = tl.load(bias_ptr + tl.arange(0, num_experts))
    scores = scores + bias

    # 3) Partition into groups, compute top-2 per group and sum
    group_scores = tl.zeros((NUM_GROUPS,), dtype=tl.float32)

    for g in range(0, NUM_GROUPS):
        max1 = -float("inf")
        max2 = -float("inf")
        start = g * EXPERTS_PER_GROUP
        for ee in range(0, EXPERTS_PER_GROUP):
            idx = start + ee
            v = scores[idx]
            if v > max1:
                max2 = max1
                max1 = v
            elif v > max2:
                max2 = v
        group_scores[g] = max1 + max2

    # 4) Select top-4 groups (rank them)
    top_group_idx = tl.zeros((TOPK_GROUP,), dtype=tl.int32)
    for rk in range(0, TOPK_GROUP):
        best_val = -float("inf")
        best_g = -1
        for g in range(0, NUM_GROUPS):
            if group_scores[g] > best_val:
                best_val = group_scores[g]
                best_g = g
        top_group_idx[rk] = best_g
        # Remove selected group from further consideration
        group_scores[best_g] = -float("inf")

    # 5) Build per-expert mask and apply masking: set non-selected groups' scores to -inf
    mask_e = tl.zeros((num_experts,), dtype=tl.int32)
    for rk in range(0, TOPK_GROUP):
        g_sel = top_group_idx[rk]
        start = g_sel * EXPERTS_PER_GROUP
        for ee in range(0, EXPERTS_PER_GROUP):
            idx = start + ee
            mask_e[idx] = 1
    # Invert mask: non-selected set to -inf
    for e in range(0, num_experts):
        if mask_e[e] == 0:
            scores[e] = -float("inf")

    # 6) Select top-8 from masked scores (duplicates allowed)
    selected_idx = tl.zeros((TOPK,), dtype=tl.int32)
    for rk in range(0, TOPK):
        best_val = -float("inf")
        best_e = -1
        for e in range(0, num_experts):
            if scores[e] > best_val:
                best_val = scores[e]
                best_e = e
        selected_idx[rk] = best_e
        scores[best_e] = -float("inf")  # remove from consideration

    # 7) Gather original logits for those selected indices: original_logit = dot(hidden, weight[e]) without bias
    selected_logit = tl.zeros((TOPK,), dtype=tl.float32)
    for rk in range(0, TOPK):
        e = selected_idx[rk]
        w = tl.load(weight_ptr + e * hidden_dim + tl.arange(0, hidden_dim), mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
        hs = tl.load(hidden_ptr + token * hidden_dim + tl.arange(0, hidden_dim), mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
        orig = tl.sum(hs * w, axis=0)
        selected_logit[rk] = orig

    # 8) Normalize by sum(original_selected + 1e-20), then apply scaling
    denom = 0.0
    for rk in range(0, TOPK):
        denom += selected_logit[rk] + 1e-20
    inv_denom = 1.0 / denom
    topk_weight_out = selected_logit * scaling * inv_denom

    # Store outputs as flattened arrays: out_idx[token*TOPK + rk], out_weight[token*TOPK + rk]
    base = token * TOPK
    for rk in range(0, TOPK):
        out_idx_ptr[base + rk] = selected_idx[rk]
        out_weight_ptr[base + rk] = topk_weight_out[rk]


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32
        # weight: [num_experts, hidden_dim], float32 (num_experts must be 256)
        # expert_bias: [num_experts], float32
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"
        assert expert_bias.shape[0] == num_experts, "expert_bias must match num_experts"

        # Allocate outputs flattened
        out_idx = torch.empty(num_tokens * 8, dtype=torch.int32, device=hidden_states.device)
        out_weight = torch.empty(num_tokens * 8, dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _grouped_routing_kernel[grid](
            hidden_states, weight, expert_bias,
            out_idx, out_weight,
            num_tokens, hidden_dim, num_experts,
            routed_scaling_factor,
            EXPERTS_PER_GROUP=32, NUM_GROUPS=8, TOPK_GROUP=4, TOPK=8
        )

        # Reshape back to [num_tokens, 8]
        topk_idx = out_idx.view(num_tokens, 8)
        topk_weight = out_weight.view(num_tokens, 8)

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
