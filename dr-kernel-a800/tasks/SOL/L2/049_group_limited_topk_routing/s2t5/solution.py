import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Constants fixed as in the original
        self.num_experts = 256
        self.n_groups = 8
        self.experts_per_group = self.num_experts // self.n_groups  # 32
        self.topk_group = 4
        self.topk_experts = 8

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-only implementation that matches the original Model.run outputs:
        - topk_idx: [num_tokens, 8] int64
        - topk_weight: [num_tokens, 8] float32
        """
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == self.num_experts, "num_experts must be 256"

        # Allocate outputs (Triton will write int32; we cast to int64 after)
        topk_idx_i32 = torch.empty((num_tokens, self.topk_experts), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, self.topk_experts), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _group_limited_topk_kernel[grid](
            hidden_states, weight, expert_bias,
            routed_scaling_factor,
            num_tokens, hidden_dim, num_experts,
            topk_idx_i32, topk_weight
        )

        # Return outputs with exact dtypes as original
        topk_idx = topk_idx_i32.to(torch.int64)
        return topk_idx, topk_weight


@triton.jit
def _group_limited_topk_kernel(
    hidden_ptr,         # *f32, [num_tokens, hidden_dim]
    weight_ptr,         # *f32, [num_experts, hidden_dim]
    expert_bias_ptr,    # *f32, [num_experts]
    routed_scale,       # f32, scalar
    num_tokens,         # i32
    hidden_dim,         # i32
    num_experts,        # i32
    topk_idx_ptr,       # *i32, [num_tokens, 8]
    topk_weight_ptr,    # *f32, [num_tokens, 8]
):
    token = tl.program_id(0)

    # 1) Compute logits: scores[token, e] = dot(hidden[token, :], weight[e, :])
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    hs = hidden_ptr + token * hidden_dim
    for e in range(0, num_experts):
        w = weight_ptr + e * hidden_dim
        acc = 0.0
        for d in range(0, hidden_dim):
            acc += tl.load(hs + d) * tl.load(w + d)
        scores[e] = acc

    # 2) Sigmoid + expert bias -> scores_for_routing
    scores = 1.0 / (1.0 + tl.exp(-scores))
    # Add expert bias per-expert
    for e in range(0, num_experts):
        scores[e] += tl.load(expert_bias_ptr + e)

    # 3) Reshape into groups and compute top-2 per group
    num_groups = self.n_groups
    group_scores = tl.zeros((num_groups,), dtype=tl.float32)
    for g in range(0, num_groups):
        start = g * self.experts_per_group
        sub = scores[start:start + self.experts_per_group]
        top1_val = -1.0e30
        top1_idx = 0
        top2_val = -1.0e30
        top2_idx = 0
        for j in range(0, self.experts_per_group):
            v = sub[j]
            idx = start + j
            if v > top1_val:
                top2_val = top1_val
                top2_idx = top1_idx
                top1_val = v
                top1_idx = idx
            elif v > top2_val:
                top2_val = v
                top2_idx = idx
        group_scores[g] = top1_val + top2_val

    # 4) Select top-4 groups
    top4_groups = tl.zeros((self.topk_group,), dtype=tl.int32)  # indices in [0,7]
    for r in range(0, self.topk_group):
        best_val = -1.0e30
        best_g = 0
        for g in range(0, num_groups):
            if group_scores[g] > best_val:
                best_val = group_scores[g]
                best_g = g
        group_scores[best_g] = -1.0e30  # prevent reuse
        top4_groups[r] = best_g

    # 5) Build per-expert mask based on selected groups: if group not in top4, set score to -inf
    masked_scores = tl.zeros((num_experts,), dtype=tl.float32)
    for e in range(0, num_experts):
        found = 0
        for r in range(0, self.topk_group):
            g = top4_groups[r]
            if (e // self.experts_per_group) == g:
                found = 1
                break
        masked_scores[e] = scores[e] if found == 1 else -1.0e30

    # 6) Select top-8 from masked_scores (allow duplicates)
    selected_idx = tl.zeros((self.topk_experts,), dtype=tl.int32)
    selected_val = tl.zeros((self.topk_experts,), dtype=tl.float32)
    for r in range(0, self.topk_experts):
        best_val = -1.0e30
        best_e = 0
        for e in range(0, num_experts):
            if masked_scores[e] > best_val:
                best_val = masked_scores[e]
                best_e = e
        selected_idx[r] = best_e
        selected_val[r] = best_val

    # 7) Normalize using original logits for selected indices and apply routed_scaling_factor
    original_selected = tl.zeros((self.topk_experts,), dtype=tl.float32)
    sum_original = 0.0
    for r in range(0, self.topk_experts):
        e = selected_idx[r]
        original_selected[r] = scores[e]
        sum_original += original_selected[r]
    sum_original = tl.maximum(sum_original, 1e-20)
    normalized = selected_val * routed_scale / sum_original

    # Store outputs
    base_out = topk_idx_ptr + token * self.topk_experts
    base_weight = topk_weight_ptr + token * self.topk_experts
    for r in range(0, self.topk_experts):
        tl.store(base_out + r, selected_idx[r])
        tl.store(base_weight + r, normalized[r])


def run(*args):
    return ModelNew()(*args)
