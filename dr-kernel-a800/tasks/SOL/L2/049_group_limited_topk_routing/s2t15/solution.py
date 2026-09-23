import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: compute scores_for_routing = sigmoid(dot(hidden_states[:, :], weight)) + expert_bias
# hidden_states: [num_tokens, hidden_dim]
# weight: [num_experts, hidden_dim]
# expert_bias: [num_experts]
# outputs:
# - scores_for_routing: [num_tokens, num_experts] (we write to this tensor)
@triton.jit
def compute_scores_kernel(
    hidden_states_ptr,  # *f32, [num_tokens, hidden_dim]
    weight_ptr,         # *f32, [num_experts, hidden_dim]
    expert_bias_ptr,    # *f32, [num_experts]
    scores_ptr,         # *f32, [num_tokens, num_experts] output
    num_tokens,         # int32
    hidden_dim,         # int32
    num_experts,        # int32
    BLOCK_K: tl.constexpr,  # block size along hidden_dim
):
    token_id = tl.program_id(0)  # each program handles one token
    expert_id = tl.program_id(1) # each program handles one expert

    # Initialize accumulator for dot product
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over hidden_dim in chunks of BLOCK_K
    for k in range(0, hidden_dim, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < hidden_dim

        # Load hidden state for this token (vector)
        # Compute pointer: hidden_states_ptr + token_id * hidden_dim + offs
        h_ptr = hidden_states_ptr + token_id * hidden_dim + offs
        h = tl.load(h_ptr, mask=mask, other=0.0)  # [BLOCK_K]

        # Load weight row for this expert (vector)
        w_ptr = weight_ptr + expert_id * hidden_dim + offs
        w = tl.load(w_ptr, mask=mask, other=0.0)  # [BLOCK_K]

        # Accumulate dot product
        # Masked elements are 0 due to tl.load(other=0.0), so safe to multiply and sum
        acc += tl.sum(h * w, axis=0)

    # Compute sigmoid and add expert bias
    score = 1.0 / (1.0 + tl.exp(-acc))
    bias = tl.load(expert_bias_ptr + expert_id)
    score = score + bias

    # Store result into scores[token_id, expert_id]
    scores_ptr_row = scores_ptr + token_id * num_experts + expert_id
    tl.store(scores_ptr_row, score)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32, device
        # weight: [num_experts, hidden_dim], float32, device
        # expert_bias: [num_experts], float32, device
        # routed_scaling_factor: float

        assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_dim]"
        assert weight.dim() == 2, "weight must be [num_experts, hidden_dim]"
        assert expert_bias.dim() == 1 and expert_bias.shape[0] == weight.shape[0], "expert_bias must be [num_experts]"
        assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32, "All inputs must be float32"
        assert hidden_states.device == weight.device and hidden_states.device == expert_bias.device, "All inputs must be on the same device"

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Allocate output scores tensor
        scores_for_routing = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: grid over tokens and experts
        # Use a BLOCK_K that divides hidden_dim well; choose 128 for typical hidden_dim=128, loop handles others.
        BLOCK_K = 128
        grid = (num_tokens, num_experts)
        compute_scores_kernel[grid](
            hidden_states, weight, expert_bias, scores_for_routing,
            num_tokens, hidden_dim, num_experts,
            BLOCK_K=BLOCK_K,
            num_warps=4,  # reasonable default
        )

        # Continue with the original logic using torch ops on GPU tensors
        # Apply sigmoid and add expert_bias already done in kernel (redundant if we had computed manually; here scores_for_routing already has sigmoid+bias)
        # Group and top-k logic
        # Partition into groups [num_tokens, 8, 32]
        # Note: scores_for_routing is [num_tokens, 256]
        group_scores_reshaped = scores_for_routing.view(num_tokens, 8, 32)  # 256 = 8 * 32
        # Compute top-2 values within each group along last dim
        # Use torch.topk on last dimension
        top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)  # [num_tokens, 8, 2]
        group_scores = top2_vals.sum(dim=-1)  # [num_tokens, 8]

        # Select top-4 groups per token
        _, group_idx = torch.topk(group_scores, k=4, dim=-1, sorted=False)  # [num_tokens, 4]

        # Build group_mask [num_tokens, 8]
        group_mask = torch.zeros((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)
        group_mask.scatter_(1, group_idx, 1.0)  # positions corresponding to group_idx are set to 1.0

        # Expand to per-expert mask [num_tokens, 256]
        score_mask = group_mask.unsqueeze(-1).expand(num_tokens, 8, 32)
        score_mask = score_mask.reshape(num_tokens, 256)

        # Apply mask: set non-selected groups to -inf (so they are ignored in subsequent topk)
        neg_inf = torch.finfo(torch.float32).min
        masked_scores = scores_for_routing.masked_fill(score_mask == 0, neg_inf)  # [num_tokens, 256]

        # Select top-8 experts from masked scores (duplicates allowed)
        _, topk_idx = torch.topk(masked_scores, k=8, dim=-1, sorted=False)  # [num_tokens, 8], int64

        # Gather original logits for selected indices and normalize
        # We need original logits: since we computed scores = sigmoid(dot) + bias, original logits are scores_for_routing - expert_bias
        original_logits = scores_for_routing - expert_bias  # [num_tokens, 256]

        selected_logits = torch.gather(original_logits, dim=1, index=topk_idx)  # [num_tokens, 8]
        norm = selected_logits.sum(dim=-1, keepdim=True) + 1e-20  # [num_tokens, 1]
        topk_weight = selected_logits / norm  # [num_tokens, 8]
        topk_weight = topk_weight * routed_scaling_factor  # apply scaling

        # Store outputs
        topk_idx_out = topk_idx.to(torch.int64)
        topk_weight_out = topk_weight

        return topk_idx_out, topk_weight_out


def run(*args):
    return ModelNew()(*args)
