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
    # 2D grid: (M, ceil_div(N, BLOCK_N))
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Accumulator for this block of experts
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[m, k] chunk: shape [BLOCK_K]
        a = tl.load(A_ptr + m * stride_Am + k_offsets * stride_Ak,
                    mask=mask_k, other=0.0)

        # Load B[n, k] chunk: shape [BLOCK_N, BLOCK_K]
        b = tl.load(B_ptr + n_offsets[:, None] * stride_Bn + k_offsets[None, :] * stride_Bk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # Accumulate: acc[n] += sum_k (A[m,k] * B[n,k])
        acc += tl.sum(b * a[None, :], axis=1)

    # Store results
    tl.store(C_ptr + m * stride_Cm + n_offsets * stride_Cn,
             acc, mask=mask_n)


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Bn,
                         stride_Ym, stride_Yn,
                         BLOCK_N: tl.constexpr):
    # One program per token row
    m = tl.program_id(0)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask, other=0.0)
        b = tl.load(Bias_ptr + n_offsets * stride_Bn, mask=mask, other=0.0)
        y = 1.0 / (1.0 + tl.exp(-x)) + b
        tl.store(Y_ptr + m * stride_Ym + n_offsets * stride_Yn, y, mask=mask)


@triton.jit
def top2_per_group_kernel(S_ptr, GroupScores_ptr,
                          M, G, E,
                          stride_Sm, stride_Sg, stride_Sexp,
                          stride_Gm, stride_Gg,
                          BLOCK_E: tl.constexpr):
    # Compute top-2 per group and store sum to GroupScores [M, G]
    for m in range(0, M):
        for g in range(0, G):
            best1 = -float('inf')
            best2 = -float('inf')
            # Each group has E experts; scan in chunks of BLOCK_E
            for e in range(0, E, BLOCK_E):
                e_offsets = e + tl.arange(0, BLOCK_E)
                mask = (e_offsets < E)
                s = tl.load(
                    S_ptr + m * stride_Sm + g * stride_Sg + e_offsets * stride_Sexp,
                    mask=mask, other=-float('inf')
                )
                # Within this chunk, find top-2
                for j in range(0, BLOCK_E):
                    val = s[j]
                    if val > best1:
                        best2 = best1
                        best1 = val
                    elif val > best2:
                        best2 = val
            sum_top2 = best1 + best2
            tl.store(GroupScores_ptr + m * stride_Gm + g * stride_Gg, sum_top2)


@triton.jit
def topk_groups_arg_kernel(GroupScores_ptr, GroupIdx_ptr,
                           M, G, K_TOP,
                           stride_Gm, stride_Gg,
                           stride_Ijm, stride_Ijk,
                           BLOCK_G: tl.constexpr):
    # One program per token m; select top-K_TOP groups from G
    m = tl.program_id(0)
    scores = tl.load(GroupScores_ptr + m * stride_Gm + tl.arange(0, BLOCK_G),
                     mask=tl.arange(0, BLOCK_G) < G, other=-float('inf'))
    # Bubble selection for K_TOP
    for k in range(0, K_TOP):
        max_val = -float('inf')
        max_idx = -1
        for g in range(0, G):
            if scores[g] > max_val:
                max_val = scores[g]
                max_idx = g
        tl.store(GroupIdx_ptr + m * stride_Ijm + k * stride_Ijk, max_idx)
        scores = tl.where(tl.arange(0, BLOCK_G) == max_idx, -float('inf'), scores)


@triton.jit
def expand_group_mask_kernel(GroupIdx_ptr, GroupMask_ptr,
                             M, G, K_TOP,
                             stride_Ijm, stride_Ijk,
                             stride_Gm, stride_Gg,
                             BLOCK_G: tl.constexpr):
    # One program per token m; set group_mask[m, group_idx[m, k]] = 1.0
    m = tl.program_id(0)
    for k in range(0, K_TOP):
        idx = tl.load(GroupIdx_ptr + m * stride_Ijm + k * stride_Ijk)  # scalar
        tl.store(GroupMask_ptr + m * stride_Gm + idx * stride_Gg, 1.0)


@triton.jit
def mask_scores_kernel(Scores_ptr, GroupMask_ptr, Masked_ptr,
                        M, N, G,
                        stride_Sm, stride_Sn,
                        stride_Gm, stride_Gg,
                        stride_Mm, stride_Mn,
                        BLOCK_N: tl.constexpr):
    # One program per token m; for each expert n, if group_mask[m, g] != 1 for that expert's group g,
    # set masked score to -inf; otherwise keep original score.
    for m in range(0, M):
        for n in range(0, N, BLOCK_N):
            n_offsets = n + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < N
            s = tl.load(Scores_ptr + m * stride_Sm + n_offsets * stride_Sn, mask=mask_n, other=0.0)
            # Determine group for each n: g = n // 32
            g_offsets = (n_offsets // 32).to(tl.int32)
            gm = tl.load(GroupMask_ptr + m * stride_Gm + g_offsets * stride_Gg, mask=g_offsets < G, other=0.0)
            active = gm > 0
            s = tl.where(active, s, -float('inf'))
            tl.store(Masked_ptr + m * stride_Mm + n_offsets * stride_Mn, s, mask=mask_n)


@triton.jit
def topk_experts_arg_kernel(Masked_ptr, TopIdx_ptr,
                            M, N, K_TOP,
                            stride_Mm, stride_Mn,
                            stride_Tm, stride_Tk,
                            BLOCK_N: tl.constexpr):
    # One program per token m; arg-topk on masked scores
    m = tl.program_id(0)
    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N
        s = tl.load(Masked_ptr + m * stride_Mm + n_offsets * stride_Mn, mask=mask_n, other=-float('inf'))
        for k in range(0, K_TOP):
            max_val = -float('inf')
            max_idx = -1
            for j in range(0, BLOCK_N):
                val = s[j]
                if val > max_val:
                    max_val = val
                    max_idx = n + j
            tl.store(TopIdx_ptr + m * stride_Tm + k * stride_Tk, max_idx)
            s = tl.where(tl.arange(0, BLOCK_N) == max_idx, -float('inf'), s)


@triton.jit
def normalize_and_scale_kernel(Idx_ptr, Selected_ptr, Scale, Out_ptr,
                               M, K_TOP,
                               stride_Ijm, stride_Ijk,
                               stride_Sm, stride_Sk,
                               stride_Om, stride_Ok):
    # Normalize selected scores by their sum and apply scale
    eps = 1e-20
    for m in range(0, M):
        sum_vals = 0.0
        for k in range(0, K_TOP):
            idx = tl.load(Idx_ptr + m * stride_Ijm + k * stride_Ijk)
            val = tl.load(Selected_ptr + m * stride_Sm + idx * stride_Sk)
            sum_vals += val
        inv = Scale / (sum_vals + eps)
        for k in range(0, K_TOP):
            idx = tl.load(Idx_ptr + m * stride_Ijm + k * stride_Ijk)
            val = tl.load(Selected_ptr + m * stride_Sm + idx * stride_Sk)
            out_val = val * inv
            tl.store(Out_ptr + m * stride_Om + k * stride_Ok, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float = None):
        device = hidden_states.device
        M = hidden_states.shape[0]  # num_tokens
        K = hidden_states.shape[1]
        N = weight.shape[0]  # num_experts (256 in the original)
        G = 8  # groups
        E = N // G  # 32

        # 1) Compute logits via Triton GEMV: [M, K] @ [N, K]^T -> [M, N]
        logits = torch.empty((M, N), device=device, dtype=torch.float32)
        grid = (M, triton.cdiv(N, 128))
        gemv_linear_kernel[grid](
            hidden_states.to(torch.float32), weight.to(torch.float32), logits,
            M, N, K,
            1, K,                # strides for A (row-major)
            1, K,                # strides for B (row-major, weight [N, K])
            1, 1,                # strides for C (logits [M, N], contiguous)
            128, 32,             # BLOCK_N, BLOCK_K
            num_warps=4
        )

        # 2) Sigmoid + expert bias via Triton
        scores = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_sig = (M,)
        sigmoid_bias_kernel[grid_sig](
            logits, expert_bias.to(torch.float32), scores,
            M, N,
            1, 1,
            1,
            1, 1,
            256,
            num_warps=4
        )

        # 3) Per-group top-2 (sum) via Triton
        group_scores = torch.empty((M, G), device=device, dtype=torch.float32)
        grid_top2 = (M,)
        top2_per_group_kernel[grid_top2](
            scores, group_scores,
            M, G, E,
            1, G, 1,  # we treat scores as [M, G, E] logically; Triton loads via provided offsets
            1, 1,
            32,
            num_warps=2
        )

        # 4) Per-token top-4 groups via Triton
        group_idx = torch.empty((M, 4), device=device, dtype=torch.int32)
        grid_top4 = (M,)
        topk_groups_arg_kernel[grid_top4](
            group_scores, group_idx,
            M, G, 4,
            1, 1,
            1, 1,
            8,
            num_warps=2
        )

        # 5) Build group_mask [M, G] via Triton
        group_mask = torch.empty((M, G), device=device, dtype=torch.float32)
        grid_mask = (M,)
        expand_group_mask_kernel[grid_mask](
            group_idx, group_mask,
            M, G, 4,
            1, 1,
            1, 1,
            8,
            num_warps=2
        )

        # 6) Expand group_mask to [M, N] and mask non-selected group scores to -inf via Triton
        masked_scores = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_mask_exp = (M,)
        mask_scores_kernel[grid_mask_exp](
            scores, group_mask, masked_scores,
            M, N, G,
            1, 1,
            1, 1,
            256,
            num_warps=4
        )

        # 7) Per-token top-8 experts from masked scores via Triton
        topk_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        grid_top8 = (M,)
        topk_experts_arg_kernel[grid_top8](
            masked_scores, topk_idx,
            M, N, 8,
            1, 1,
            1, 1,
            256,
            num_warps=4
        )

        # 8) Normalize and scale selected weights via Triton
        # Gather selected scores from original scores at indices topk_idx to compute normalization
        selected_scores = torch.empty((M, 8), device=device, dtype=torch.float32)
        for m in range(M):
            for k in range(8):
                idx = int(topk_idx[m, k].item())
                selected_scores[m, k] = scores[m, idx]

        topk_weight = torch.empty((M, 8), device=device, dtype=torch.float32)
        grid_norm = (M,)
        normalize_and_scale_kernel[grid_norm](
            topk_idx, selected_scores, self.routed_scaling_factor,
            M, 8,
            1, 1,
            1, 1,
            1, 1,
            num_warps=2
        )

        # Return indices and normalized weights
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
