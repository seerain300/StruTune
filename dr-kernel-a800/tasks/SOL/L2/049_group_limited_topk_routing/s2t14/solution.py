import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32
        # weight: [num_experts, hidden_dim], float32 (num_experts=256)
        # expert_bias: [num_experts], float32
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"
        assert expert_bias.shape[0] == num_experts, "expert_bias must match num_experts"

        # Outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        selected_original_logits = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        @triton.jit
        def _routing_compute_one_token(
            hidden_ptr,                 # *float32, [num_tokens, hidden_dim]
            weight_ptr,                 # *float32, [256, hidden_dim]
            expert_bias_ptr,            # *float32, [256]
            out_idx_ptr,                # *int32,   [num_tokens*8]
            out_log_ptr,                # *float32, [num_tokens*8]
            num_tokens,                 # int32
            hidden_dim,                 # int32
            routed_scaling_factor,      # float32
        ):
            token_id = tl.program_id(axis=0)

            # 1) Compute logits per expert: scores[e] = dot(hidden[token, :], weight[e, :])
            scores = tl.zeros((num_experts,), dtype=tl.float32)
            for e in range(0, num_experts):
                acc = 0.0
                hidden_row_ptr = hidden_ptr + token_id * hidden_dim
                w_row_ptr = weight_ptr + e * hidden_dim
                for j in range(0, hidden_dim):
                    x = tl.load(hidden_row_ptr + j)
                    w = tl.load(w_row_ptr + j)
                    acc += x * w
                scores[e] = acc

            # 2) Apply sigmoid and add expert_bias -> scores_for_routing
            scores_for_routing = tl.zeros((num_experts,), dtype=tl.float32)
            for e in range(0, num_experts):
                v = 1.0 / (1.0 + tl.exp(-scores[e]))  # sigmoid
                b = tl.load(expert_bias_ptr + e)
                scores_for_routing[e] = v + b

            # 3) Group into 8 groups of 32; compute top-2 per group, sum -> group_scores [8]
            EXPERTS_PER_GROUP = 32
            NUM_GROUPS = 8
            group_scores = tl.zeros((NUM_GROUPS,), dtype=tl.float32)
            for g in range(0, NUM_GROUPS):
                start = g * EXPERTS_PER_GROUP
                group_vals = tl.zeros((EXPERTS_PER_GROUP,), dtype=tl.float32)
                for i in range(0, EXPERTS_PER_GROUP):
                    idx = start + i
                    group_vals[i] = scores_for_routing[idx]
                # Top-2 within group
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

            # 4) Select top-4 groups (unordered) and mark others with -inf
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
                # Mark other groups by setting scores_for_routing to -inf
                for g2 in range(0, NUM_GROUPS):
                    if g2 != max_idx:
                        start2 = g2 * EXPERTS_PER_GROUP
                        for i in range(0, EXPERTS_PER_GROUP):
                            idx2 = start2 + i
                            scores_for_routing[idx2] = -float('inf')

            # 5) Select top-8 from masked scores (duplicates allowed) and record indices
            TOPK = 8
            selected_indices = tl.zeros((TOPK,), dtype=tl.int32)
            for t in range(0, TOPK):
                maxv = -float('inf')
                max_idx = -1
                for e in range(0, num_experts):
                    v = scores_for_routing[e]
                    if v > maxv:
                        maxv = v
                        max_idx = e
                selected_indices[t] = max_idx
                scores_for_routing[max_idx] = -float('inf')

            # 6) Recompute original logits for selected indices and write to out buffers
            for t in range(0, TOPK):
                e = selected_indices[t]
                # Original logits = dot(hidden[token, :], weight[e, :])
                acc = 0.0
                hidden_row_ptr = hidden_ptr + token_id * hidden_dim
                w_row_ptr = weight_ptr + e * hidden_dim
                for j in range(0, hidden_dim):
                    x = tl.load(hidden_row_ptr + j)
                    w = tl.load(w_row_ptr + j)
                    acc += x * w
                # Write original logits for normalization on host
                out_log_ptr[token_id * TOPK + t] = acc
                # Write selected index
                out_idx_ptr[token_id * TOPK + t] = e

        # Launch kernel
        _routing_compute_one_token[grid](
            hidden_states, weight, expert_bias,
            topk_idx.view(-1), selected_original_logits.view(-1),
            num_tokens, hidden_dim, routed_scaling_factor,
            num_warps=1, num_stages=1
        )

        # 7) Host computes normalized weights: normalize by sum(original_selected_scores + 1e-20), then scale
        sum_vals = selected_original_logits + 1e-20  # [num_tokens, 8]
        row_sums = sum_vals.sum(dim=1)               # [num_tokens]
        topk_weight = (sum_vals / row_sums.unsqueeze(1)) * routed_scaling_factor  # [num_tokens, 8]

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
