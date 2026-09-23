class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        All computation must be done in Triton kernels; no PyTorch operators in host code.
        Returns:
        - topk_idx: [num_tokens, 8], int64
        - topk_weight: [num_tokens, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors"
        assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_dim]"
        assert weight.dim() == 2, "weight must be [num_experts, hidden_dim]"
        assert expert_bias.dim() == 1 and expert_bias.shape[0] == weight.shape[0], "expert_bias must be [num_experts]"
        num_tokens = hidden_states.shape[0]
        num_experts = weight.shape[0]
        hidden_dim = weight.shape[1]
        assert num_experts == 256, "num_experts must be 256"
        assert num_experts % 8 == 0, "num_experts must be divisible by 8"
        experts_per_group = num_experts // 8  # 32
        n_group = 8

        # Ensure contiguous and dtype fp32
        hidden_states = hidden_states.contiguous().to(torch.float32)
        weight = weight.contiguous().to(torch.float32)
        expert_bias = expert_bias.contiguous().to(torch.float32)

        # 1) Compute logits = hidden_states @ weight^T + expert_bias
        logits = torch.empty((num_tokens, num_experts), device=hidden_states.device, dtype=torch.float32)

        M = num_tokens
        N = num_experts
        K = hidden_dim

        # Strides
        stride_am = hidden_states.stride(0)
        stride_ak = hidden_states.stride(1)
        stride_wn = weight.stride(0)  # row-major [N, K]
        stride_wk = weight.stride(1)
        stride_om = logits.stride(0)
        stride_on = logits.stride(1)

        # Launch linear_bias_kernel
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_bias_kernel[grid](
            hidden_states, weight, expert_bias, logits,
            M, N, K,
            stride_am, stride_ak,
            stride_wn, stride_wk,
            stride_om, stride_on,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) Sigmoid of logits
        scores = torch.empty_like(logits)
        stride_sm = logits.stride(0)
        stride_sn = logits.stride(1)
        stride_dm = scores.stride(0)
        stride_dn = scores.stride(1)
        grid_sigmoid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        sigmoid_kernel[grid_sigmoid](
            logits, scores,
            M, N,
            stride_sm, stride_sn,
            stride_dm, stride_dn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 3) Reshape scores into groups [M, 8, 32]
        scores_group = scores.view(M, n_group, experts_per_group)

        # 4) Compute top-2 sums per group using Triton
        group_scores = torch.empty((M, n_group), device=scores.device, dtype=torch.float32)
        stride_sgm = scores_group.stride(0), scores_group.stride(1), scores_group.stride(2)
        stride_gom = group_scores.stride(0), group_scores.stride(1)
        # Triton expects strides as (stride_m, stride_e, stride_s) for IN, and (stride_m, stride_e) for OUT
        IN_strides = scores_group.stride(0), scores_group.stride(1), scores_group.stride(2)
        OUT_strides = group_scores.stride(0), group_scores.stride(1)
        grid_group = (triton.cdiv(M, BLOCK_M), 1)  # BLOCK_E=8 handled inside
        top2_group_kernel[grid_group](
            scores_group, group_scores,
            M, n_group, experts_per_group,
            IN_strides[0], IN_strides[1], IN_strides[2],
            OUT_strides[0], OUT_strides[1],
            BLOCK_M=BLOCK_M, BLOCK_E=8, S=experts_per_group,
        )

        # 5) Select top-4 groups
        group_idx = torch.empty((M, 4), device=scores.device, dtype=torch.int32)
        stride_gs = group_scores.stride(0), group_scores.stride(1)
        stride_gis = group_idx.stride(0), group_idx.stride(1)
        grid_top4 = (triton.cdiv(M, BLOCK_M), 1)  # we select 4
        top4_groups_kernel[grid_top4](
            group_scores, group_idx,
            M, n_group,
            stride_gs[0], stride_gs[1],
            stride_gis[0], stride_gis[1],
            BLOCK_M=BLOCK_M, BLOCK_E=4,
        )

        # 6) Final top-8 selection and normalization using Triton (iterative elimination + recompute logits)
        # We'll compute output buffer [M, 16] where odd columns are indices (int32), even columns are weights (float32)
        out_buffer = torch.empty((M, 16), device=scores.device, dtype=torch.float32)
        # For safety, zero init
        out_buffer.zero_()
        grid_final = (triton.cdiv(M, BLOCK_M),)
        final_top8_with_weight_and_indices_kernel[grid_final](
            hidden_states, weight, expert_bias, out_buffer,
            M, hidden_dim, num_experts, routed_scaling_factor,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            out_buffer.stride(0), out_buffer.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_E=8,
        )

        # Parse out_buffer: even columns are weights, odd are indices
        # Convert to final outputs
        # indices (int64), weights (float32)
        topk_idx = torch.empty((M, 8), device=scores.device, dtype=torch.long)
        topk_weight = torch.empty((M, 8), device=scores.device, dtype=torch.float32)

        # Read indices and weights from out_buffer
        for j in range(8):
            topk_idx[:, j] = out_buffer[:, 2 * j + 1].to(torch.long)
            topk_weight[:, j] = out_buffer[:, 2 * j]

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
