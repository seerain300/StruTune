import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_logits(hidden_ptr, weight_ptr, scores_ptr,
                   num_tokens, hidden_dim, num_experts,
                   BLOCK_H: tl.constexpr):
    # Each program handles one token; iterate hidden_dim in chunks of BLOCK_H
    pid = tl.program_id(0)
    # Compute dot product for all experts: scores_ptr layout [num_tokens, num_experts]
    for h in range(0, hidden_dim, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask_h = offs < hidden_dim
        # Load hidden vector for this token
        hidden = tl.load(hidden_ptr + pid * hidden_dim + offs, mask=mask_h, other=0.0).to(tl.float32)
        # Accumulate dot product for each expert
        total = tl.zeros((num_experts,), dtype=tl.float32)
        for e in range(0, num_experts):
            w = tl.load(weight_ptr + e * hidden_dim + offs, mask=mask_h, other=0.0).to(tl.float32)
            total[e] = tl.sum(hidden * w, axis=0)
        # Store scores for this token
        tl.store(scores_ptr + pid * num_experts + tl.arange(0, num_experts), total)


@triton.jit
def group_top2_sum(scores_ptr, group_scores_ptr,
                   num_tokens, n_group, experts_per_group):
    # For each token, compute sum of top-2 per group (groups of 32 from 256)
    pid = tl.program_id(0)
    for g in range(8):
        start = g * 32
        idxs = start + tl.arange(0, 32)
        vals = tl.load(scores_ptr + pid * 256 + idxs)
        # Compute top-2 via two reductions
        max1 = tl.max(vals, axis=0)
        top2 = -float('inf')
        # Re-scan vals and track top-2, excluding max1
        for i in range(32):
            val = vals[i]
            if val > top2 and val != max1:
                top2 = val
        sum_top2 = max1 + top2
        tl.store(group_scores_ptr + pid * 8 + g, sum_top2)


@triton.jit
def apply_group_mask(scores_ptr, group_mask_ptr, masked_ptr,
                     num_tokens, num_experts):
    # For each token, set non-selected groups to -inf
    for t in range(0, num_tokens):
        for g in range(8):
            m = tl.load(group_mask_ptr + t * 8 + g)  # scalar float
            if m == 0.0:
                start = g * 32
                idxs = start + tl.arange(0, 32)
                vals = tl.load(scores_ptr + t * 256 + idxs)
                vals = -float('inf')
                tl.store(masked_ptr + t * 256 + idxs, vals)
        # Copy remaining groups (masked_ptr initialized with scores_ptr, non-selected groups already set)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim]
        # weight: [num_experts, hidden_dim] (num_experts=256)
        # expert_bias: [num_experts]
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # 1) Compute logits via Triton: scores_ptr [num_tokens, 256] float32
        scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=hidden_states.device)
        compute_logits[(num_tokens,)](
            hidden_states, weight, scores,
            num_tokens, hidden_dim, num_experts,
            BLOCK_H=128
        )

        # 2) Sigmoid and add expert bias in torch (elementwise)
        scores = torch.sigmoid(scores)
        # Ensure expert_bias is broadcasted over tokens
        scores_for_routing = scores + expert_bias.to(torch.float32).view(1, -1)

        # 3) Group top-2 sum via Triton: group_scores [num_tokens, 8] float32
        group_scores = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)
        group_top2_sum[(num_tokens,)](
            scores_for_routing.reshape(num_tokens, 256),
            group_scores,
            num_tokens, 8, 32
        )

        # 4) Select top-4 groups per token using torch.topk
        _, group_idx = torch.topk(group_scores, k=4, dim=1, sorted=False)  # [num_tokens, 4], int64
        # Build group_mask [num_tokens, 8], 1 for selected groups, 0 otherwise
        group_mask = torch.zeros((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)
        group_mask.scatter_(1, group_idx.to(torch.int64), 1.0)

        # 5) Apply group mask to scores_for_routing using Triton: set non-selected groups to -inf
        masked_scores = torch.empty_like(scores_for_routing)
        apply_group_mask[(num_tokens,)](
            scores_for_routing.reshape(num_tokens, 256),
            group_mask,
            masked_scores.reshape(num_tokens, 256),
            num_tokens, 256
        )

        # 6) Final selection and normalization using torch (ensure correctness)
        # masked_scores: [num_tokens, 256], select top-8
        _, topk_idx = torch.topk(masked_scores, k=8, dim=1, sorted=False)  # [num_tokens, 8], int64

        # 7) Recompute original logits (pre-sigmoid) to normalize properly
        # linear = F.linear(hidden_states, weight) -> [num_tokens, 256]
        linear = torch.nn.functional.linear(hidden_states.to(torch.float32), weight.to(torch.float32))
        scores_pre = torch.sigmoid(linear)  # [num_tokens, 256]
        selected_scores_pre = torch.gather(scores_pre, dim=1, index=topk_idx.to(torch.int64))  # [num_tokens, 8]

        # 7.1 Normalize by sum(selected_scores_pre + 1e-20) and apply scaling
        selected_scores_pre = selected_scores_pre + 1e-20
        weight_selected = selected_scores_pre / selected_scores_pre.sum(dim=1, keepdim=True)  # [num_tokens, 8]
        weight_selected = weight_selected * routed_scaling_factor

        # Return indices (int64) and normalized weights (float32)
        return topk_idx, weight_selected


def run(*args):
    return ModelNew()(*args)
