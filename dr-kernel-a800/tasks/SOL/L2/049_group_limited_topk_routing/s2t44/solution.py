import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32, on device
        # weight: [num_experts, hidden_dim], float32, on device (num_experts=256)
        # expert_bias: [num_experts], float32, on device
        # routed_scaling_factor: float

        # Ensure contiguity
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()
        bias = expert_bias.contiguous()

        num_tokens = hidden.shape[0]
        hidden_dim = hidden.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Output tensors
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)

        @triton.jit
        def _group_topk_routing(hidden_ptr, weight_ptr, bias_ptr, topk_idx_ptr, topk_w_ptr, num_tokens, hidden_dim, num_experts, scaling):
            pid_token = tl.program_id(0)
            hidden_row_ptr = hidden_ptr + pid_token * hidden_dim

            # 1) Compute logits per expert: raw_logits[num_experts]
            raw_logits = tl.zeros([num_experts], dtype=tl.float32)
            for e in range(0, num_experts):
                total = 0.0
                for j in range(0, hidden_dim):
                    h = tl.load(hidden_row_ptr + j)
                    w = tl.load(weight_ptr + e * hidden_dim + j)
                    total += h * w
                raw_logits[e] = total

            # 2) Apply sigmoid and add bias -> scores
            scores = tl.zeros([num_experts], dtype=tl.float32)
            for e in range(0, num_experts):
                s = 1.0 / (1.0 + tl.exp(-raw_logits[e]))  # sigmoid
                b = tl.load(bias_ptr + e)
                scores[e] = s + b

            # 3) Reshape to groups [8, 32] and compute top-2 per group
            group_scores = scores.view(8, 32)
            group_sums = tl.zeros([8], dtype=tl.float32)
            for g in range(0, 8):
                # Compute two largest within group g (32 dims)
                # Since we have a vector of 32, we can use argmax twice easily
                vals = group_scores[g, :]
                max_val = -float('inf')
                max_idx = -1
                for i in range(0, 32):
                    v = vals[i]
                    if v > max_val:
                        max_val = v
                        max_idx = i
                # zero out max
                for i in range(0, 32):
                    vals[i] = vals[i] - (i == max_idx) * max_val
                second_val = -float('inf')
                for i in range(0, 32):
                    if vals[i] > second_val:
                        second_val = vals[i]
                group_sums[g] = max_val + second_val

            # 4) Select top-4 groups (per token)
            selected = tl.zeros([8], dtype=tl.int32)
            for t in range(0, 4):
                best = -float('inf')
                chosen = 0
                for g in range(0, 8):
                    if selected[g] == 0 and group_sums[g] > best:
                        best = group_sums[g]
                        chosen = g
                selected[chosen] = 1

            # 5) Build group_mask and expand to expert-level mask
            group_mask = tl.zeros([8], dtype=tl.int32)
            group_mask[selected] = 1

            # Set non-selected groups' scores to -inf
            for g in range(0, 8):
                if group_mask[g] == 0:
                    base = g * 32
                    for i in range(0, 32):
                        e = base + i
                        scores[e] = -float('inf')

            # 6) Select top-8 from masked scores (duplicates allowed)
            top8 = tl.zeros([8], dtype=tl.int32)
            top8_vals = tl.zeros([8], dtype=tl.float32)
            for t in range(0, 8):
                best = -float('inf')
                chosen = 0
                for e in range(0, num_experts):
                    if scores[e] > best:
                        best = scores[e]
                        chosen = e
                top8[t] = chosen
                top8_vals[t] = best
                # avoid duplicates
                scores[chosen] = -float('inf')

            # 7) Gather original logits for selected indices and normalize
            denom = 0.0
            for k in range(0, 8):
                e = top8[k]
                orig = 0.0
                for j in range(0, hidden_dim):
                    h = tl.load(hidden_row_ptr + j)
                    w = tl.load(weight_ptr + e * hidden_dim + j)
                    orig += h * w
                denom += (orig + 1e-20)

            # 8) Store outputs
            base = pid_token * 8
            for k in range(0, 8):
                tl.store(topk_idx_ptr + base + k, top8[k])
                orig = 0.0
                for j in range(0, hidden_dim):
                    h = tl.load(hidden_row_ptr + j)
                    w = tl.load(weight_ptr + top8[k] * hidden_dim + j)
                    orig += h * w
                weight_k = (orig + 1e-20) / denom
                weight_k *= scaling
                tl.store(topk_w_ptr + base + k, weight_k)

        _group_topk_routing[grid](
            hidden, weight, bias, topk_idx, topk_weight,
            num_tokens, hidden_dim, num_experts, routed_scaling_factor
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
