class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,  # [M, K]
        weight: torch.Tensor,         # [N, K] (num_experts=256, hidden_dim=K)
        expert_bias: torch.Tensor,    # [N]
        routed_scaling_factor: float,
    ):
        # Ensure device/dtype/contiguity
        device = hidden_states.device
        dtype = torch.float32

        # 1) GEMM via Triton: logits = hidden_states @ weight.T
        A = hidden_states.contiguous().to(dtype)  # [M, K]
        # weight is [N, K], we need B = weight.T -> [K, N]
        B = weight.T.contiguous().to(dtype)      # [K, N]
        M, K = A.shape
        N = B.shape[1]  # 256

        logits = torch.empty((M, N), dtype=dtype, device=device)

        stride_am, stride_ak = A.stride(0), A.stride(1)  # A strides
        stride_bk, stride_bn = B.stride(0), B.stride(1)  # B strides
        stride_cm, stride_cn = logits.stride(0), logits.stride(1)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_AxB_kernel[grid](
            A, B, logits,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 2) Sigmoid + bias via Triton
        scores = torch.empty((M, N), dtype=dtype, device=device)
        stride_xm, stride_xn = logits.stride(0), logits.stride(1)
        stride_ym, stride_yn = scores.stride(0), scores.stride(1)

        _sigmoid_add_bias_kernel[(M,)](
            logits, expert_bias.to(dtype), scores,
            M, N,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            num_warps=4, num_stages=1,
        )

        # 3) Group top-2 sum -> [M, 8] via Triton
        group_scores = torch.empty((M, 8), dtype=dtype, device=device)
        stride_sm, stride_sn = scores.stride(0), scores.stride(1)
        stride_gm, stride_gn = group_scores.stride(0), group_scores.stride(1)

        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            stride_sm, stride_sn,
            stride_gm, stride_gn,
            EXPERTS_PER_GROUP=32, N_GROUPS=8,
            num_warps=1, num_stages=1,
        )

        # 4) Group top-4 indices -> [M, 4] via Triton
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=device)
        stride_gm_gi, stride_gn_gi = group_scores.stride(0), group_scores.stride(1)
        stride_om, stride_on = group_idx.stride(0), group_idx.stride(1)

        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M,
            stride_gm_gi, stride_gn_gi,
            stride_om, stride_on,
            N_GROUPS=8, EXPERTS_PER_GROUP=32,
            num_warps=1, num_stages=1,
        )

        # 5) Final top-8 selection + normalize via Triton -> [M, 8] idx and weights
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        topk_weight = torch.empty((M, 8), dtype=dtype, device=device)

        stride_sm_f, stride_sn_f = scores.stride(0), scores.stride(1)
        stride_gm_f, stride_gn_f = group_idx.stride(0), group_idx.stride(1)
        stride_om_f, stride_on_f = topk_idx.stride(0), topk_idx.stride(1)
        stride_wm, stride_wn = topk_weight.stride(0), topk_weight.stride(1)

        _final_top8_and_normalize_kernel[(M,)](
            scores, group_idx, topk_idx, topk_weight,
            M, N,
            stride_sm_f, stride_sn_f,
            stride_gm_f, stride_gn_f,
            stride_om_f, stride_on_f,
            stride_wm, stride_wn,
            float(routed_scaling_factor),
            N_GROUPS=8, EXPERTS_PER_GROUP=32, FINAL_K=8,
            num_warps=4, num_stages=1,
        )

        # Return int64 indices and float32 weights
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
