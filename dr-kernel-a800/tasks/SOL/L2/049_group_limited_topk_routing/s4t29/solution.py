class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,   # [num_experts, hidden_dim], float32
        expert_bias: torch.Tensor,  # [num_experts], float32
        routed_scaling_factor: float,
    ):
        """
        hidden_states: [M, K], float32, CUDA
        weight: [N, K], float32, CUDA (N=256, K=hidden_dim)
        expert_bias: [N], float32, CUDA
        routed_scaling_factor: float
        Returns:
        topk_idx: [M, 8], int64
        topk_weight: [M, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be CUDA"
        assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32, "Use float32"
        M, K = hidden_states.shape
        N = weight.shape[0]  # num_experts = 256
        # Ensure contiguous
        hidden_states_c = hidden_states.contiguous()
        weight_c = weight.contiguous()
        expert_bias_c = expert_bias.contiguous()

        # 1) Compute logits = hidden_states @ weight.T (B = weight.T [K, N])
        # Prepare B = weight.T
        B = weight_c.T.contiguous()  # [K, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        # Launch matmul kernel
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
        _matmul_AxB_kernel[grid](
            hidden_states_c, B, logits,
            M, N, K,
            hidden_states_c.stride(0), hidden_states_c.stride(1),
            B.stride(0), B.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # 2) scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid2 = (M, triton.cdiv(N, 128))
        _sigmoid_add_bias_kernel[grid2](
            logits, expert_bias_c, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_N=128,
            num_warps=4,
        )

        # 3) Group top-2 sum per token
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        grid3 = (M,)
        _group_top2_sum_kernel[grid3](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXPERTS_PER_GROUP=32, NUM_GROUPS=8, BLOCK=32,
            num_warps=1,
        )

        # 4) Group top-4 selection per token
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden_states.device)
        grid4 = (M,)
        _group_top4_select_kernel[grid4](
            group_scores, group_idx,
            M,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            K_GROUPS=4, BLOCK=8,
            num_warps=1,
        )

        # 5) Final top-8 selection and normalization per token
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        grid5 = (M,)
        _final_top8_and_normalize_kernel_full[grid5](
            scores, group_idx, topk_idx, topk_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_scaling_factor,
            BLOCK=128, K_TOP=8,
            num_warps=4,
        )

        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
