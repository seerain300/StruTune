class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure device and contiguity
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be CUDA tensors"
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]  # 256
        N = weight.shape[0]         # 256

        # 1) GEMM: logits = hidden @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        A = hidden_states.contiguous().to(torch.float32)
        B = weight.contiguous().to(torch.float32)  # [N, K]
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)
        stride_bk = B.stride(1)
        stride_lm = logits.stride(0)
        stride_ln = logits.stride(1)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_proj_kernel[grid](
            A, B, logits,
            M, K, N,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_lm, stride_ln,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) scores = sigmoid(logits) + expert_bias
        scores = torch.empty_like(logits)
        Bias = expert_bias.contiguous().to(torch.float32)
        stride_b = Bias.stride(0)
        stride_xm = logits.stride(0)
        stride_xn = logits.stride(1)
        stride_ym = scores.stride(0)
        stride_yn = scores.stride(1)
        grid_elem = (M * N,)
        sigmoid_add_bias_kernel[grid_elem](
            logits, Bias, scores,
            M, N,
            stride_xm, stride_xn,
            stride_b,
            stride_ym, stride_yn,
            num_warps=1, num_stages=1,
        )

        # 3) group top-2 sum per token: [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_sm = scores.stride(0)
        stride_sn = scores.stride(1)
        stride_gm = group_scores.stride(0)
        stride_gn = group_scores.stride(1)
        group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_gm=stride_gm, stride_gn=stride_gn,
            num_warps=1, num_stages=1,
        )

        # 4) select top-4 groups per token: [M, 4], int32
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        stride_om = top4_groups.stride(0)
        stride_on = top4_groups.stride(1)
        select_top4_groups_kernel[(M,)](
            group_scores, top4_groups,
            M,
            stride_gm=stride_gm, stride_gn=stride_gn,
            stride_om=stride_om, stride_on=stride_on,
            num_warps=1, num_stages=1,
        )

        # 5) mask non-selected groups: create masked_scores
        masked_scores = torch.empty_like(scores)
        stride_msm = masked_scores.stride(0)
        stride_msn = masked_scores.stride(1)
        mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N,
            EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_om=stride_om, stride_on=stride_on,
            stride_msm=stride_msm, stride_msn=stride_msn,
            num_warps=1, num_stages=1,
        )

        # 6) select top-8 from masked_scores per token: [M, 8], int32
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        stride_om8 = top8_indices.stride(0)
        stride_on8 = top8_indices.stride(1)
        select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            stride_msm=stride_msm, stride_msn=stride_msn,
            stride_om=stride_om8, stride_on=stride_on8,
            num_warps=1, num_stages=1,
        )

        # 7) normalize and scale to produce topk_weight and topk_idx
        topk_idx = torch.empty_like(top8_indices, dtype=torch.int64)  # match original
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_omw = topk_weight.stride(0)
        stride_onw = topk_weight.stride(1)
        normalize_and_scale_kernel[(M,)](
            scores, top8_indices, topk_weight,
            M, N,
            self.routed_scaling_factor,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_om=stride_om8, stride_on=stride_on8,
            num_warps=1, num_stages=1,
        )

        # Convert indices to int64 to match original return type
        topk_idx = top8_indices.to(torch.int64)

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
