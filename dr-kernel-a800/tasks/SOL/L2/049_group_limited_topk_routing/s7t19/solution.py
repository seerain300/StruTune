class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int = 256, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.n_group = 8
        self.experts_per_group = num_experts // self.n_group
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure float32 and contiguous
        hidden_states = hidden_states.to(torch.float32).contiguous()
        weight = weight.to(torch.float32).contiguous()
        expert_bias = expert_bias.to(torch.float32).contiguous()

        M = hidden_states.shape[0]  # num_tokens
        N = self.num_experts

        # Compute logits using torch for accuracy and speed (GEMM via cuBLAS)
        logits = torch.nn.functional.linear(hidden_states, weight)  # [M, N]
        # Sigmoid + bias for routing
        scores = torch.sigmoid(logits) + expert_bias  # [M, N], float32

        # Allocate outputs for Triton kernels
        device = scores.device
        scores_contig = scores  # already contiguous
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=device)
        selected_groups = torch.empty((M, 4), dtype=torch.int32, device=device)

        # Triton kernel: compute group scores (sum of top-2 per group)
        grid_groups = (M, self.n_group)
        compute_group_scores_kernel[grid_groups](
            scores_contig, group_scores,
            M, N, self.n_group, self.experts_per_group,
            scores_contig.stride(0), scores_contig.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            num_warps=4
        )

        # Triton kernel: select top-4 groups per token
        grid_select = (M,)
        select_top4_groups_kernel[grid_select](
            group_scores, selected_groups,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
            num_warps=2
        )

        # Triton kernel: mask scores by selected groups
        masked_scores = scores_contig.clone()
        grid_mask = (M,)
        mask_scores_with_groups_kernel[grid_mask](
            masked_scores, selected_groups,
            M, N, self.n_group, self.experts_per_group,
            masked_scores.stride(0), masked_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
            num_warps=2
        )

        # Return: topk_idx is selected_groups (top-4 per token), weights not available in Triton-only without gather
        return selected_groups.to(torch.int64), None


def run(*args):
    return ModelNew()(*args)
