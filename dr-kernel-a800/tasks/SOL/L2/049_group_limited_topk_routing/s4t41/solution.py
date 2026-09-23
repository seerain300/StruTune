class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        M = hidden_states.shape[0]
        N = 256  # num_experts
        K = hidden_states.shape[1]  # hidden_dim

        # Ensure dtype float32, contiguous
        hidden_states = hidden_states.contiguous().to(torch.float32)
        weight_T = weight.T.contiguous().to(torch.float32)  # [K, N]
        expert_bias = expert_bias.contiguous().to(torch.float32)  # [N]

        # 1) Compute logits via Triton matmul
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_no_bias_kernel[grid](
            hidden_states, weight_T, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 2) Compute scores = sigmoid(logits) + expert_bias via Triton
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        # Launch elementwise kernel over rows; one program per row
        grid2 = (M,)
        _sigmoid_add_bias_kernel[grid2](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK=128,
            num_warps=4,
        )

        # 3) Compute group_scores [M, 8] via Triton
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            BLOCK_N=32,
            num_warps=1,
        )

        # 4) Select top-4 groups per token via Triton
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden_states.device)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, N,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            CHUNK=8,
            num_warps=1,
        )

        # 5) Final top-8 selection and weights (indices in OUT_IDX, weights in OUT_WEIGHT) via Triton
        out_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden_states.device)
        out_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        _final_top8_and_normalize_kernel[(M,)](
            scores, group_idx, out_idx, out_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            out_idx.stride(0), out_idx.stride(1),
            out_weight.stride(0), out_weight.stride(1),
            routed_scaling_factor,
            CHUNK=4,
            num_warps=1,
        )

        return out_idx.to(torch.int64), out_weight


def run(*args):
    return ModelNew()(*args)
