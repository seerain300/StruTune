import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure inputs are on CUDA and contiguous
        device = torch.device("cuda")
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"
        assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_dim]"
        assert weight.dim() == 2, "weight must be [num_experts, hidden_dim]"
        assert expert_bias.dim() == 1 and expert_bias.shape[0] == num_experts, "expert_bias must be [num_experts]"

        hidden_states = hidden_states.to(device=device, dtype=torch.float32).contiguous()
        weight = weight.to(device=device, dtype=torch.float32).contiguous()
        expert_bias = expert_bias.to(device=device, dtype=torch.float32).contiguous()

        num_tokens = hidden_states.shape[0]

        # Outputs
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=device)

        # Launch Triton kernel (grid = (num_tokens,))
        run_kernel(num_tokens, hidden_dim, num_experts, routed_scaling_factor, hidden_states, weight, expert_bias, topk_idx, topk_weight)


@triton.jit
def run_kernel(
    num_tokens: tl.constexpr,
    hidden_dim: tl.constexpr,
    num_experts: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    hidden_states_ptr,   # [num_tokens, hidden_dim], float32
    weight_ptr,          # [num_experts, hidden_dim], float32
    expert_bias_ptr,     # [num_experts], float32
    topk_idx_ptr,        # [num_tokens, 8], int32
    topk_weight_ptr,     # [num_tokens, 8], float32
):
    token = tl.program_id(0)

    # Base pointers
    hs_base = hidden_states_ptr + token * hidden_dim

    # 1) Compute logits scores[token, e] = dot(hidden_states[token,:], weight[e,:])
    scores = tl.zeros((num_experts,), dtype=tl.float32)
    for j in range(hidden_dim):
        xj = tl.load(hs_base + j)
        for e in range(num_experts):
            we = tl.load(weight_ptr + e * hidden_dim + j)
            scores[e] += xj * we

    # 2) Apply sigmoid and add expert_bias
    scores = 1.0 / (1.0 + tl.exp(-scores))  # sigmoid
    bias = tl.load(expert_bias_ptr + tl.arange(0, num_experts))
    scores += bias

    # 3) Group top-2 per group and sum to form group_scores[token, g]
    n_group = 8
    experts_per_group = 32  # 256 / 8
    group_scores = tl.zeros((n_group,), dtype=tl.float32)
    for g in range(n_group):
        start = g * experts_per_group
        top1_val = -float("inf")
        top1_idx = 0
        top2_val = -float("inf")
        top2_idx = 0
        for i in range(experts_per_group):
            e = start + i
            val = scores[e]
            if val > top1_val:
                top2_val = top1_val
                top2_idx = top1_idx
                top1_val = val
                top1_idx = e
            elif val > top2_val:
                top2_val = val
                top2_idx = e
        group_scores[g] = top1_val + top2_val

    # 4) Select top-4 groups per token
    top4 = tl.zeros((4,), dtype=tl.int32)
    top4_vals = tl.zeros((4,), dtype=tl.float32)
    for g in range(n_group):
        val = group_scores[g]
        inserted = 0
        for i in range(4):
            if (top4_vals[i] == 0.0) or (val > top4_vals[i]):
                # bubble-insertion
                for j in range(3, i - 1, -1):
                    top4_vals[j] = top4_vals[j - 1]
                    top4[j] = top4[j - 1]
                top4_vals[i] = val
                top4[i] = g
                inserted = 1
                break
        if not inserted:
            for i in range(4):
                if top4_vals[i] == 0.0:
                    top4_vals[i] = val
                    top4[i] = g
                    inserted = 1
                    break
            if not inserted:
                # replace smallest if full
                min_pos = 0
                for i in range(1, 4):
                    if top4_vals[i] < top4_vals[min_pos]:
                        min_pos = i
                top4_vals[min_pos] = val
                top4[min_pos] = g

    # 5) Expand group_mask to per-expert and mask scores: non-selected groups -> -inf
    masked_scores = tl.zeros((num_experts,), dtype=tl.float32) - float("inf")
    for g in range(n_group):
        if g in top4:  # dynamic membership: if g appears in top4
            start = g * experts_per_group
            for i in range(experts_per_group):
                e = start + i
                masked_scores[e] = scores[e]

    # 6) Select top-8 from masked scores
    top8_idx = tl.zeros((8,), dtype=tl.int32)
    top8_vals = tl.zeros((8,), dtype=tl.float32)
    for e in range(num_experts):
        val = masked_scores[e]
        inserted = 0
        for i in range(8):
            if (top8_vals[i] == 0.0) or (val > top8_vals[i]):
                for j in range(7, i - 1, -1):
                    top8_vals[j] = top8_vals[j - 1]
                    top8[j] = top8[j - 1]
                top8_vals[i] = val
                top8[i] = e
                inserted = 1
                break
        if not inserted:
            for i in range(8):
                if top8_vals[i] == 0.0:
                    top8_vals[i] = val
                    top8[i] = e
                    inserted = 1
                    break
            if not inserted:
                min_pos = 0
                for i in range(1, 8):
                    if top8_vals[i] < top8_vals[min_pos]:
                        min_pos = i
                top8_vals[min_pos] = val
                top8[min_pos] = e

    # 7) Gather original logits for selected indices
    original_selected = tl.zeros((8,), dtype=tl.float32)
    for i in range(8):
        e = top8[i]
        orig_score = tl.zeros((), dtype=tl.float32)
        for j in range(hidden_dim):
            xj = tl.load(hs_base + j)
            we = tl.load(weight_ptr + e * hidden_dim + j)
            orig_score += xj * we
        original_selected[i] = orig_score

    # 8) Normalize by sum(original_selected + 1e-20), then apply routed_scaling_factor
    total = tl.zeros((), dtype=tl.float32)
    for i in range(8):
        total += original_selected[i] + 1e-20
    norm = 1.0 / total
    for i in range(8):
        val = original_selected[i] * norm * routed_scaling_factor
        tl.store(topk_weight_ptr + token * 8 + i, val)
        tl.store(topk_idx_ptr + token * 8 + i, top8[i].to(tl.int32))


def run(*args):
    return ModelNew()(*args)
