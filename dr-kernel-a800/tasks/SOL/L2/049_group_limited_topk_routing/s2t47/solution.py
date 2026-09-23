import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: compute logits for a batch of tokens and a block of experts in parallel.
# Inputs:
#   hidden_ptr: *float32, [num_tokens, hidden_dim], row-major
#   weight_ptr: *float32, [num_experts, hidden_dim], row-major
#   bias_ptr:   *float32, [num_experts]
#   logits_ptr: *float32, [num_tokens, num_experts] (output)
# Launch config:
#   grid = (num_tokens, num_experts // BLOCK_E)
@triton.jit
def compute_logits_and_sigmoid(
    hidden_ptr,       # *float32, [num_tokens, hidden_dim]
    weight_ptr,       # *float32, [num_experts, hidden_dim]
    bias_ptr,         # *float32, [num_experts]
    logits_ptr,       # *float32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,  # used for indexing
    hidden_dim: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,     # number of experts per program, e.g., 32
):
    pid_tok = tl.program_id(axis=0)  # token id
    pid_blk = tl.program_id(axis=1)  # block id along experts
    if pid_tok >= num_tokens:
        return

    e_start = pid_blk * BLOCK_E
    e_offsets = e_start + tl.arange(0, BLOCK_E)
    # mask for expert range
    e_mask = e_offsets < num_experts

    # load hidden vector for this token
    h = tl.zeros((hidden_dim,), dtype=tl.float32)
    for d in range(0, hidden_dim):
        h[d] = tl.load(hidden_ptr + pid_tok * hidden_dim + d)

    # compute dot-products for this block of experts
    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for d in range(0, hidden_dim):
        w = tl.load(weight_ptr + e_offsets * hidden_dim + d, mask=e_mask, other=0.0)
        acc += w * h[d]

    # sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # add expert bias
    b = tl.load(bias_ptr + e_offsets, mask=e_mask, other=0.0)
    acc += b

    # store results
    for i in range(0, BLOCK_E):
        e_idx = e_start + i
        if e_idx < num_experts:
            tl.store(logits_ptr + pid_tok * num_experts + e_idx, acc[i])


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32
        # weight: [num_experts, hidden_dim], float32, num_experts == 256
        # expert_bias: [num_experts], float32
        # routed_scaling_factor: float
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Output buffer for logits [num_tokens, num_experts]
        logits = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: compute logits and sigmoid
        grid = (num_tokens, num_experts // 32)  # 32 experts per block
        compute_logits_and_sigmoid[grid](
            hidden_states,
            weight,
            expert_bias,
            logits,
            num_tokens,
            hidden_dim,
            num_experts,
            BLOCK_E=32,
        )

        # Now perform the group-limited top-k routing in PyTorch (to keep semantics exact and code concise):
        # Step 1: Reshape scores into groups [num_tokens, 8, 32]
        scores = logits  # already sigmoid + bias
        group_scores_reshaped = scores.view(num_tokens, 8, 32)

        # Step 2: Compute top-2 per group and sum
        top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)  # [num_tokens, 8, 2]
        group_scores = top2_vals.sum(dim=-1)  # [num_tokens, 8]

        # Step 3: Select top-4 groups per token
        _, group_idx = torch.topk(group_scores, k=4, dim=-1, sorted=False)  # [num_tokens, 4], indices in [0..7]

        # Step 4: Create group mask [num_tokens, 8] with 1.0 for selected groups, else 0.0
        group_mask = torch.zeros((num_tokens, 8), dtype=torch.float32)
        group_mask.scatter_(1, group_idx, 1.0)

        # Step 5: Expand group mask to per-expert mask [num_tokens, 256]
        score_mask = group_mask.unsqueeze(-1).expand(num_tokens, 8, 32)
        score_mask = score_mask.reshape(num_tokens, num_experts)

        # Step 6: Mask out non-selected groups by setting their scores to -inf
        neg_inf = float("-inf")
        masked_scores = scores.masked_fill(~(score_mask.bool()), neg_inf)  # keep selected groups, -inf otherwise

        # Step 7: Select top-8 experts from masked scores (allow duplicates)
        _, topk_idx = torch.topk(masked_scores, k=8, dim=-1, largest=True, sorted=False)  # [num_tokens, 8]

        # Step 8: Gather original (pre-bias) logits for selected indices
        # Original logits without bias are simply logits we computed above (since we applied bias in the kernel).
        # But we need original logits (no bias). We can reconstruct by computing dot-product without bias:
        # However, for efficiency and correctness, we can obtain original logits by subtracting bias from scores:
        # original_logit[e] = scores[e] - bias[e], but we don't have per-element bias subtraction in kernel,
        # so we recompute without bias using the kernel output where bias was added. This is fine:
        # original_logits are the acc before adding bias. We don't have them explicitly; but since we computed
        # scores with bias, we can compute original_logits by subtracting bias. To avoid ambiguity, we can
        # recompute original_logits in PyTorch: original_logits = torch.bmm(hidden_states.unsqueeze(1), weight.t())[:, 0, :].
        # Given weight and hidden_states, we can simply obtain original logits as scores - bias.
        # We'll do that efficiently: original_logits = logits - expert_bias (broadcast).
        original_logits = logits - expert_bias.unsqueeze(0)  # [num_tokens, 256]

        # Gather selected original logits
        selected_logits = original_logits.gather(1, topk_idx)  # [num_tokens, 8]

        # Normalize by sum(selected_logits + eps), then apply routing scaling
        eps = 1e-20
        denom = selected_logits + eps
        sum_selected = denom.sum(dim=1, keepdim=True)  # [num_tokens, 1]
        topk_weight = (denom / sum_selected) * routed_scaling_factor  # [num_tokens, 8]

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
