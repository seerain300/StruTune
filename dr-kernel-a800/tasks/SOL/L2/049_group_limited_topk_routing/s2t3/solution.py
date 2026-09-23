import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-optimized forward that mirrors the original logic:
        - Compute scores = sigmoid(F.linear(hidden, weight))
        - Add expert bias
        - Group-limited top-k: top-2 per group, sum to get group scores
        - Select top-4 groups
        - Mask out non-selected groups from routed scores
        - Select top-8 from masked scores
        - Normalize using original logits and apply routing scaling
        """
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert weight.shape[1] == hidden_dim, "weight's second dim must match hidden_states' second dim"
        assert expert_bias.shape[0] == num_experts, "expert_bias shape must match num_experts"

        # Allocate outputs on device
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)

        # Launch one Triton program per token
        grid = (num_tokens,)
        _group_limited_topk_token_kernel[grid](
            hidden_ptr=hidden_states.contiguous().data_ptr(),
            weight_ptr=weight.contiguous().data_ptr(),
            expert_bias_ptr=expert_bias.contiguous().data_ptr(),
            routed_scaling_factor=routed_scaling_factor,
            topk_idx_ptr=topk_idx.data_ptr(),
            topk_weight_ptr=topk_weight.data_ptr(),
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
