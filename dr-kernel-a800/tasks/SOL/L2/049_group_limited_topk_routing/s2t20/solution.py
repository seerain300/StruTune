import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Inputs:
        # hidden_states: [num_tokens, hidden_dim]
        # weight: [num_experts, hidden_dim], num_experts = 256
        # expert_bias: [num_experts]
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        device = hidden_states.device

        # Ensure contiguity and dtype for Triton
        hidden_flat = hidden_states.reshape(-1).contiguous()             # [num_tokens * hidden_dim], float32
        weight_flat = weight.reshape(-1).contiguous()                   # [num_experts * hidden_dim], float32
        expert_bias_flat = expert_bias.contiguous()                     # [num_experts], float32

        # Outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=device)

        # Triton launch: one program per token
        grid = (num_tokens,)

        # Define Triton kernel
        run_kernel = triton.jit(
            """
            #define EXPERTS_PER_GROUP 32
            #define NUM_EXPERTS 256
            #define N_GROUP 8
            #define TOPK_GROUP 4

            @triton.jit
            def routing_compute(
                hidden_ptr,         // float32* [num_tokens*hidden_dim]
                weight_ptr,          // float32* [NUM_EXPERTS*hidden_dim]
                bias_ptr,            // float32* [NUM_EXPERTS]
                out_idx_ptr,         // int32*  [num_tokens*8]
                out_weight_ptr,      // float32* [num_tokens*8]
                routed_scale,        // float32
                hidden_size,         // int32
                num_tokens,          // int32
                seed                 // int32 (unused)
            ):
                token = tl.program_id(0)

                # Compute scores per expert for this token: scores[NUM_EXPERTS]
                scores = [0.0] * NUM_EXPERTS
                for e in range(NUM_EXPERTS):
                    acc = 0.0
                    # Dot product over hidden_dim
                    for j in range(hidden_size):
                        hid_val = tl.load(hidden_ptr + token * hidden_size + j)
                        w_val = tl.load(weight_ptr + e * hidden_size + j)
                        acc += hid_val * w_val
                    sig = 1.0 / (1.0 + tl.exp(-acc))
                    scores[e] = sig + tl.load(bias_ptr + e)

                # Group scores: top-2 per group, sum
                group_scores_list = [0.0] * N_GROUP
                for g in range(N_GROUP):
                    start = g * EXPERTS_PER_GROUP
                    top1 = -1.0e20
                    top2 = -1.0e20
                    for j in range(EXPERTS_PER_GROUP):
                        idx = start + j
                        s = scores[idx]
                        if s > top1:
                            top2 = top1
                            top1 = s
                        elif s > top2:
                            top2 = s
                    group_scores_list[g] = top1 + top2

                # Select top-4 groups via scan (deterministic)
                selected_groups = [-1] * TOPK_GROUP
                used = [False] * N_GROUP
                for t in range(TOPK_GROUP):
                    maxv = -1.0e20
                    chosen = -1
                    for g in range(N_GROUP):
                        if not used[g] and group_scores_list[g] > maxv:
                            maxv = group_scores_list[g]
                            chosen = g
                    used[chosen] = True
                    selected_groups[t] = chosen

                # Build per-expert mask: 1 for selected groups, else 0
                score_mask = [0] * NUM_EXPERTS
                for t in range(TOPK_GROUP):
                    g = selected_groups[t]
                    start = g * EXPERTS_PER_GROUP
                    for j in range(EXPERTS_PER_GROUP):
                        idx = start + j
                        score_mask[idx] = 1

                # Apply masking: set non-selected group elements to a large negative value
                masked_scores = [0.0] * NUM_EXPERTS
                for e in range(NUM_EXPERTS):
                    g = e // EXPERTS_PER_GROUP
                    if score_mask[e] == 0:
                        masked_scores[e] = -1.0e20  # emulate -inf
                    else:
                        masked_scores[e] = scores[e]

                # Select top-8 from masked_scores (largest first)
                selected_exp = [-1] * 8
                used_exp = [False] * NUM_EXPERTS
                for t in range(8):
                    maxv = -1.0e20
                    chosen = -1
                    for e in range(NUM_EXPERTS):
                        if not used_exp[e] and masked_scores[e] > maxv:
                            maxv = masked_scores[e]
                            chosen = e
                    used_exp[chosen] = True
                    selected_exp[t] = chosen

                # Recompute original logits for selected experts
                original_scores = [0.0] * NUM_EXPERTS
                for e in range(NUM_EXPERTS):
                    acc = 0.0
                    for j in range(hidden_size):
                        hid_val = tl.load(hidden_ptr + token * hidden_size + j)
                        w_val = tl.load(weight_ptr + e * hidden_size + j)
                        acc += hid_val * w_val
                    sig = 1.0 / (1.0 + tl.exp(-acc))
                    original_scores[e] = sig + tl.load(bias_ptr + e)

                # Normalize routing weights using original selected scores and apply scaling
                denom = 0.0
                for t in range(8):
                    e = selected_exp[t]
                    denom += original_scores[e]
                for t in range(8):
                    e = selected_exp[t]
                    w = original_scores[e] / (denom + 1e-20)
                    out_weight_ptr[token * 8 + t] = w * routed_scale
                    out_idx_ptr[token * 8 + t] = e
            """
        )

        # Launch Triton kernel
        run_kernel[grid](
            hidden_flat, weight_flat, expert_bias_flat, topk_idx, topk_weight, routed_scaling_factor, hidden_dim, num_tokens, 0
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
