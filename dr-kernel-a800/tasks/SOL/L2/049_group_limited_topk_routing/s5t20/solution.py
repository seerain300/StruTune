import torch
import triton
import triton.language as tl


@triton.jit
def gemv_linear_kernel(A_ptr, B_ptr, C_ptr,
                        M, N, K,
                        stride_Am, stride_Ak,
                        stride_Bn, stride_Bk,
                        stride_Cm, stride_Cn,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # One program per token row m
    m = tl.program_id(0)
    # Loop over N (experts) in blocks
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        # Accumulators for this block
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # Loop over hidden dimension K in blocks
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            # Load hidden vector chunk for this token: A[m, k]
            a = tl.load(
                A_ptr + m * stride_Am + k_offsets * stride_Ak,
                mask=k_offsets < K, other=0.0
            )  # [BLOCK_K]
            # Load weight block: B[n, k] for n in n_offsets, k in k_offsets
            b = tl.load(
                B_ptr + n_offsets[:, None] * stride_Bn + k_offsets[None, :] * stride_Bk,
                mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
                other=0.0
            )  # [BLOCK_N, BLOCK_K]
            # Accumulate: acc[n] += sum_k a[k] * b[n, k]
            acc += tl.sum(a[None, :] * b, axis=1)
        # Store accumulated results for this block
        tl.store(
            C_ptr + m * stride_Cm + n_offsets * stride_Cn,
            acc,
            mask=n_offsets < N
        )


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Ym, stride_Yn,
                         BLOCK_N: tl.constexpr):
    # One program per token row m
    m = tl.program_id(0)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        x = tl.load(
            X_ptr + m * stride_Xm + n_offsets * stride_Xn,
            mask=n_offsets < N, other=0.0
        )
        b = tl.load(Bias_ptr + n_offsets, mask=n_offsets < N, other=0.0)
        y = 1.0 / (1.0 + tl.exp(-x)) + b
        tl.store(
            Y_ptr + m * stride_Ym + n_offsets * stride_Yn,
            y,
            mask=n_offsets < N
        )


@triton.jit
def top2_sum_kernel(S_ptr, Out_ptr,
                     M, N, G, E,
                     stride_Sm, stride_Sn,
                     stride_Outm, stride_Outg,
                     BLOCK_N: tl.constexpr):
    # One program per token row m
    m = tl.program_id(0)
    for g in range(0, G):
        base = g * E
        offs = base + tl.arange(0, BLOCK_N)
        vals = tl.load(
            S_ptr + m * stride_Sm + offs * stride_Sn,
            mask=offs < N, other=-1e20
        )
        # Top-2 via two reductions
        v1 = tl.max(vals, axis=0)
        mask1 = vals == v1
        vals2 = tl.where(mask1, -1e20, vals)
        v2 = tl.max(vals2, axis=0)
        tl.store(
            Out_ptr + m * stride_Outm + g * stride_Outg,
            v1 + v2
        )


@triton.jit
def topk_groups_arg_kernel(GroupScores_ptr, OutIdx_ptr,
                           M, G, K_GROUPS,
                           stride_GS, stride_GN,
                           stride_Outm, stride_Outk,
                           BLOCK_G: tl.constexpr):
    # One program per token row m
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_G)
    scores = tl.load(
        GroupScores_ptr + m * stride_GS + offs * stride_GN,
        mask=offs < G, other=-1e20
    )
    for k in range(K_GROUPS):
        v = tl.max(scores, axis=0)
        idx = tl.max(tl.where(scores == v, offs, -1), axis=0)
        tl.store(OutIdx_ptr + m * stride_Outm + k * stride_Outk, idx)
        scores = tl.where(scores == v, -1e20, scores)


@triton.jit
def expand_group_mask_kernel(GroupMask_ptr, Masked_ptr,
                             M, G, N,
                             stride_GMm, stride_GMg,
                             stride_Mm, stride_Mn,
                             BLOCK_G: tl.constexpr, BLOCK_N: tl.constexpr):
    # One program per token row m
    m = tl.program_id(0)
    offs_g = tl.arange(0, BLOCK_G)
    gm = tl.load(GroupMask_ptr + m * stride_GMm + offs_g * stride_GMg,
                 mask=offs_g < G, other=0.0)  # float 0/1
    for g in range(0, G):
        group_active = gm[g]  # scalar float
        start = g * (N // G)  # 32
        n_offs = start + tl.arange(0, BLOCK_N)
        vals = tl.where(group_active == 0.0, tl.full([BLOCK_N], -1e20, tl.float32),
                        tl.full([BLOCK_N], 0.0, tl.float32))
        tl.store(Masked_ptr + m * stride_Mm + n_offs * stride_Mn,
                 vals, mask=n_offs < N)


@triton.jit
def masked_topk_experts_arg_kernel(ScoreMasked_ptr, OutIdx_ptr,
                                   M, N, TOP_K,
                                   stride_SMm, stride_SMn,
                                   stride_Outm, stride_Outk,
                                   BLOCK_N: tl.constexpr):
    # One program per token row m
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    scores = tl.load(
        ScoreMasked_ptr + m * stride_SMm + offs * stride_SMn,
        mask=offs < N, other=-1e20
    )
    for k in range(TOP_K):
        v = tl.max(scores, axis=0)
        idx = tl.max(tl.where(scores == v, offs, -1), axis=0)
        tl.store(OutIdx_ptr + m * stride_Outm + k * stride_Outk, idx)
        scores = tl.where(scores == v, -1e20, scores)


@triton.jit
def normalize_and_scale_kernel(Idx_ptr, Selected_ptr, Scale, Out_ptr,
                               M, TOP_K, OUT_K,
                               stride_Idxm, stride_Idxk,
                               stride_Selkm, stride_Selkk,
                               stride_Outm, stride_Outk):
    # One program per token row m
    m = tl.program_id(0)
    sum_scores = tl.zeros((), dtype=tl.float32)
    for k in range(TOP_K):
        idx = tl.load(Idx_ptr + m * stride_Idxm + k * stride_Idxk)
        val = tl.load(Selected_ptr + m * stride_Selkm + k * stride_Selkk)
        sum_scores += val
    eps = 1e-20
    norm = 1.0 / (sum_scores + eps)
    for k in range(OUT_K):
        idx = tl.load(Idx_ptr + m * stride_Idxm + k * stride_Idxk)
        val = tl.load(Selected_ptr + m * stride_Selkm + k * stride_Selkk)
        val = val * norm * Scale
        tl.store(Out_ptr + m * stride_Outm + k * stride_Outk, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                weight: torch.Tensor,
                expert_bias: torch.Tensor,
                routed_scaling_factor: float):
        """
        hidden_states: [num_tokens, K], float32, CUDA
        weight: [num_experts, K], float32, CUDA  (PyTorch weight is [E, K])
        expert_bias: [num_experts], float32, CUDA
        routed_scaling_factor: float
        Returns:
        topk_idx: [num_tokens, 8], int32
        topk_weight: [num_tokens, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Tensors must be on CUDA"
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]  # 256
        G = 8
        E = N // G  # 32
        TOP_K = 8

        # 1) Triton GEMV: logits = hidden_states @ weight.T -> [M, N]
        logits = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_gemv = (M,)
        gemv_linear_kernel[grid_gemv](
            hidden_states, weight, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_N=128, BLOCK_K=128,
            num_warps=4
        )

        # 2) Triton sigmoid + add expert bias -> scores
        scores = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_sigmoid = (M,)
        sigmoid_bias_kernel[grid_sigmoid](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_N=128,
            num_warps=4
        )

        # 3) Triton top-2 per group -> group_scores [M, G]
        group_scores = torch.empty((M, G), device=device, dtype=torch.float32)
        grid_top2 = (M,)
        top2_sum_kernel[grid_top2](
            scores, group_scores,
            M, N, G, E,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            BLOCK_N=128,
            num_warps=4
        )

        # 4) Triton per-token top-4 groups -> group_idx [M, 4]
        group_idx = torch.empty((M, 4), device=device, dtype=torch.int32)
        grid_top4 = (M,)
        topk_groups_arg_kernel[grid_top4](
            group_scores, group_idx,
            M, G, 4,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            BLOCK_G=8,
            num_warps=4
        )

        # 5) Compute group_mask in PyTorch: 0/1 float [M, G]
        group_mask = torch.zeros((M, G), device=device, dtype=torch.float32)
        for m in range(M):
            for j in range(4):
                idx = int(group_idx[m, j].item())
                group_mask[m, idx] = 1.0

        # 6) Triton expand group mask to [M, N] and mask non-selected groups with -inf
        masked_scores = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_mask = (M,)
        expand_group_mask_kernel[grid_mask](
            group_mask, masked_scores,
            M, G, N,
            group_mask.stride(0), group_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            BLOCK_G=8, BLOCK_N=128,
            num_warps=4
        )

        # 7) Triton per-token top-8 expert selection from masked_scores -> topk_idx [M, 8]
        topk_idx = torch.empty((M, TOP_K), device=device, dtype=torch.int32)
        grid_top8 = (M,)
        masked_topk_experts_arg_kernel[grid_top8](
            masked_scores, topk_idx,
            M, N, TOP_K,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            BLOCK_N=256,
            num_warps=4
        )

        # 8) Triton normalize + scale selected weights: gather original selected scores for normalization
        # Gather selected scores from 'scores' using topk_idx
        selected_scores = torch.empty((M, TOP_K), device=device, dtype=torch.float32)
        # This gather uses PyTorch for simplicity; it's a small step.
        for m in range(M):
            for k in range(TOP_K):
                idx = int(topk_idx[m, k].item())
                selected_scores[m, k] = scores[m, idx]

        topk_weight = torch.empty((M, TOP_K), device=device, dtype=torch.float32)
        grid_norm = (M,)
        normalize_and_scale_kernel[grid_norm](
            topk_idx, selected_scores, routed_scaling_factor, topk_weight,
            M, TOP_K, TOP_K,
            topk_idx.stride(0), topk_idx.stride(1),
            selected_scores.stride(0), selected_scores.stride(1),
            topk_weight.stride(0), topk_weight.stride(1)
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
