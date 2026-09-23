import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Triton-only implementation: no torch ops in forward, only allocations and kernel launch.
        # Inputs:
        # hidden_states: [num_tokens, hidden_dim], device
        # weight: [num_experts, hidden_dim], device (num_experts must be 256)
        # expert_bias: [num_experts], device
        # routed_scaling_factor: float
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Prepare contiguous inputs and outputs (no torch math, only allocation)
        hidden_c = hidden_states.contiguous().to(torch.float32)
        weight_c = weight.contiguous().to(torch.float32)
        bias_c = expert_bias.contiguous().to(torch.float32)

        # Outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # One program per token
        grid = (num_tokens,)

        @triton.jit
        def _group_topk_routing_kernel(
            hidden_ptr,      # *f32, [num_tokens, hidden_dim]
            weight_ptr,      # *f32, [num_experts, hidden_dim]
            bias_ptr,        # *f32, [num_experts]
            out_idx_ptr,     # *i32, [num_tokens, 8]
            out_weight_ptr,  # *f32, [num_tokens, 8]
            routed_scale,    # f32
            hidden_dim: tl.constexpr,
            num_experts: tl.constexpr,
        ):
            token = tl.program_id(0)  # 0..num_tokens-1
            row_start = token * hidden_dim

            # 1) Compute logits scores[token, e] via dot product
            scores = tl.zeros([num_experts], dtype=tl.float32)
            # Loop over hidden_dim to compute dot per expert
            for j in range(0, hidden_dim):
                hid_val = tl.load(hidden_ptr + row_start + j)  # scalar
                for e in range(0, num_experts):
                    w_val = tl.load(weight_ptr + e * hidden_dim + j)  # scalar
                    scores[e] += hid_val * w_val

            # 2) Apply sigmoid and add expert bias
            for e in range(0, num_experts):
                scores[e] = 1.0 / (1.0 + tl.exp(-scores[e])) + tl.load(bias_ptr + e)

            # 3) Compute group scores: sum of top-2 per group of 32
            group_scores = tl.zeros([8], dtype=tl.float32)
            for g in range(0, 8):
                start = g * 32
                expert_idx = start + tl.arange(0, 32)
                valid = expert_idx < num_experts
                scores_sub = tl.zeros([32], dtype=tl.float32)
                for i in range(0, 32):
                    # For valid i, set scores_sub[i] = scores[start + i], else 0
                    idx = start + i
                    scores_sub[i] = tl.where(valid[i], tl.load(scores + idx), 0.0)
                # Find top-2 within this group
                max1 = tl.max(scores_sub, axis=0)
                # Second max: exclude positions equal to max1
                second = tl.zeros((), dtype=tl.float32)
                for i in range(0, 32):
                    v = scores_sub[i]
                    if v != max1:
                        second = tl.maximum(second, v)
                group_scores[g] = max1 + second

            # 4) Select top-4 groups
            top_group_idx = tl.zeros([4], dtype=tl.int32)
            for t in range(0, 4):
                best_val = -1.0e20
                best_idx = -1
                for g in range(0, 8):
                    if group_scores[g] > best_val:
                        best_val = group_scores[g]
                        best_idx = g
                top_group_idx[t] = best_idx
                # Zero it out for next iterations
                group_scores[best_idx] = -1.0e20

            # 5) Build group_mask and expand to per-expert mask, set non-selected groups to -inf
            group_mask = tl.zeros([8], dtype=tl.int32)
            for t in range(0, 4):
                group_mask[top_group_idx[t]] = 1
            for e in range(0, num_experts):
                g = e // 32
                if group_mask[g] == 0:
                    scores[e] = -1.0e20

            # 6) Select top-8 from masked scores (duplicates allowed)
            selected_idx = tl.zeros([8], dtype=tl.int32)
            selected_val = tl.zeros([8], dtype=tl.float32)
            for t in range(0, 8):
                best_val = -1.0e20
                best_idx = -1
                for e in range(0, num_experts):
                    if scores[e] > best_val:
                        best_val = scores[e]
                        best_idx = e
                selected_idx[t] = best_idx
                selected_val[t] = best_val
                scores[best_idx] = -1.0e20

            # 7) Gather original logits without bias for selected indices
            original_selected = tl.zeros([8], dtype=tl.float32)
            for t in range(0, 8):
                e = selected_idx[t]
                acc = 0.0
                for j in range(0, hidden_dim):
                    hid_val = tl.load(hidden_ptr + row_start + j)
                    w_val = tl.load(weight_ptr + e * hidden_dim + j)
                    acc += hid_val * w_val
                original_selected[t] = 1.0 / (1.0 + tl.exp(-acc))

            # 8) Normalize by sum(original_selected + 1e-20) and apply routed_scaling_factor
            sum_selected = tl.sum(original_selected + 1.0e-20)
            norm = (original_selected / sum_selected) * routed_scale

            # 9) Store results
            for t in range(0, 8):
                base = token * 8 + t
                tl.store(out_idx_ptr + base, selected_idx[t])
                tl.store(out_weight_ptr + base, norm[t])

        # Launch Triton kernel (forward contains no torch math beyond allocation/launch)
        _group_topk_routing_kernel[grid](
            hidden_c, weight_c, bias_c, topk_idx, topk_weight, routed_scaling_factor,
            hidden_dim=hidden_dim, num_experts=num_experts
        )

        # Return outputs matching original: topk_idx [num_tokens, 8] int32, topk_weight [num_tokens, 8] float32
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
