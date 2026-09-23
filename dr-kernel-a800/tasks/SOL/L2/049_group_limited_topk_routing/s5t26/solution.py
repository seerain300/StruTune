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
    # 2D grid: one program per (token row, expert block)
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Accumulator for this expert block
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load hidden for this token and chunk: A[m, k]
        a = tl.load(A_ptr + m * stride_Am + k_offsets * stride_Ak,
                    mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight chunk: B[n, k] -> [BLOCK_N, BLOCK_K]
        b = tl.load(B_ptr + n_offsets[:, None] * stride_Bn + k_offsets[None, :] * stride_Bk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # Accumulate: acc[n] += sum_k (A[m,k] * B[n,k])
        acc += tl.sum(b * a[None, :], axis=1)

    # Store results to C[m, n_offsets]
    tl.store(C_ptr + m * stride_Cm + n_offsets * stride_Cn, acc, mask=mask_n)


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
    # S is [M, G, E], GroupScores is [M, G]
    m = tl.program_id(0)  # token row
    for g in range(G):
        # Get per-group scores across E (32) in chunks
        best1 = -float('inf')
        best2 = -float('inf')
        # Iterate over E
        for e in range(0, E):
            val = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + e * stride_Sexp)
            if val > best1:
                best2 = best1
                best1 = val
            elif val > best2:
                best2 = val
        tl.store(GroupScores_ptr + m * stride_Gm + g * stride_Gg, best1 + best2)


@triton.jit
def topk_groups_arg_kernel(GroupScores_ptr, GroupIdx_ptr,
                           M, K,
                           stride_Gm, stride_Gg,
                           stride_Im, stride_Ik,
                           K_CONST: tl.constexpr):
    # Compute arg-top-K groups per token using iterative selection
    m = tl.program_id(0)
    gs = GroupScores_ptr + m * stride_Gm  # points to [G]
    ti = GroupIdx_ptr + m * stride_Im     # points to [K]
    # Initialize indices 0..7
    idxs = tl.arange(0, K_CONST)
    # Select K winners by removing max each time
    for t in range(K_CONST):
        # Load current group scores vector
        scores = tl.zeros([8], dtype=tl.float32)
        for g in range(8):
            scores[g] = tl.load(gs + g * stride_Gg)
        # Compute max and argmax
        maxv = -float('inf')
        argmax = 0
        for g in range(8):
            v = scores[g]
            if v > maxv:
                maxv = v
                argmax = g
        # Write argmax to idxs[t]
        tl.store(tis, argmax)


@triton.jit
def set_group_mask_kernel(GroupIdx_ptr, GroupMask_ptr,
                          M, G,
                          stride_Ik, stride_Gm, stride_Gg,
                          K_CONST: tl.constexpr, G_CONST: tl.constexpr):
    # Given GroupIdx [M, K], set GroupMask [M, G] to 1 at selected groups
    m = tl.program_id(0)
    idxs = GroupIdx_ptr + m * stride_Ik  # points to [K]
    mask = GroupMask_ptr + m * stride_Gm # points to [G]
    # Set 1 at indices idxs[0..K_CONST-1]
    for t in range(K_CONST):
        g = tl.load(idxs + t)  # group index 0..7
        tl.store(mask + g * stride_Gg, 1.0)


@triton.jit
def expand_group_mask_kernel(GroupMask_ptr, ExpandedMask_ptr,
                             M, G, E,
                             stride_Gm, stride_Gg,
                             stride_Em, stride_En,
                             K_CONST: tl.constexpr, E_CONST: tl.constexpr):
    # GroupMask: [M, G], ExpandedMask: [M, N] where N=G*E
    m = tl.program_id(0)
    g = tl.program_id(1)  # group index 0..7
    # For each token m, write mask[g] across E experts in that group
    mask_val = tl.load(GroupMask_ptr + m * stride_Gm + g * stride_Gg)
    for e in range(0, E_CONST):
        n = g * E_CONST + e
        tl.store(ExpandedMask_ptr + m * stride_Em + n * stride_En, mask_val)


@triton.jit
def mask_scores_kernel(ExpandedMask_ptr, Scores_ptr, MaskedScores_ptr,
                       M, N,
                       stride_Em, stride_En,
                       stride_Sm, stride_Sn,
                       stride_Msm, stride_Msn,
                       BLOCK_N: tl.constexpr):
    # One program per token row
    m = tl.program_id(0)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N
        # Load mask and scores
        mask = tl.load(ExpandedMask_ptr + m * stride_Em + n_offsets * stride_En, mask=mask_n, other=0.0)
        score = tl.load(Scores_ptr + m * stride_Sm + n_offsets * stride_Sn, mask=mask_n, other=0.0)
        # Set -inf where mask == 0
        neg_inf = -1.0e20
        score = tl.where(mask > 0.0, score, neg_inf)
        tl.store(MaskedScores_ptr + m * stride_Msm + n_offsets * stride_Msn, score, mask=mask_n)


@triton.jit
def topk_experts_arg_kernel(MaskedScores_ptr, TopIdx_ptr,
                            M, N, K,
                            stride_Sm, stride_Sn,
                            stride_Im, stride_Ik,
                            K_CONST: tl.constexpr):
    # Per-token top-K (descending) indices on MaskedScores [M, N]
    m = tl.program_id(0)
    top = TopIdx_ptr + m * stride_Im  # [K]
    # Initialize idxs 0..N-1 and iteratively select max K times
    for k in range(K_CONST):
        # Find current max
        maxv = -float('inf')
        argmax = 0
        for n in range(N):
            val = tl.load(MaskedScores_ptr + m * stride_Sm + n * stride_Sn)
            if val > maxv:
                maxv = val
                argmax = n
        # Write argmax to top[k]
        tl.store(top + k * stride_Ik, argmax)


@triton.jit
def normalize_scale_kernel(TopIdx_ptr, MaskedScores_ptr, TopWeights_ptr,
                           M, N, K,
                           stride_Sm, stride_Sn,
                           stride_Im, stride_Ik,
                           stride_Wm, stride_Wk,
                           routed_factor: tl.float32,
                           K_CONST: tl.constexpr):
    # For each token, gather selected scores at TopIdx, normalize, scale, and store
    m = tl.program_id(0)
    top = TopIdx_ptr + m * stride_Im  # [K]
    scores = MaskedScores_ptr + m * stride_Sm  # [N]
    weights = TopWeights_ptr + m * stride_Wm   # [K]
    # Gather selected scores
    selected = tl.zeros([K_CONST], dtype=tl.float32)
    for k in range(K_CONST):
        idx = tl.load(top + k * stride_Ik)
        selected[k] = tl.load(scores + idx * stride_Sn)
    # Normalize: sum then divide
    total = 0.0
    for k in range(K_CONST):
        total += selected[k]
    inv_total = 1.0 / (total + 1e-20)
    # Write normalized and scaled weights
    for k in range(K_CONST):
        tl.store(weights + k * stride_Wk, selected[k] * routed_factor * inv_total)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # hidden_states: [M, K], weight: [N, K], expert_bias: [N]
        M, K = hidden_states.shape
        N = self.num_experts
        G = 8
        E = N // G  # 32

        # 1) Compute logits via Triton GEMV: [M, N]
        logits = torch.empty((M, N), device=hidden_states.device, dtype=hidden_states.dtype)
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (M, triton.cdiv(N, BLOCK_N))
        gemv_linear_kernel[grid](
            hidden_states, weight, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # 2) Triton: sigmoid + expert bias -> scores [M, N]
        scores = torch.empty_like(logits)
        sigmoid_bias_kernel[(M,)](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), scores.stride(1),  # stride_Xm, stride_Yn
            expert_bias.stride(0),               # stride_Bn
            scores.stride(0), scores.stride(1),
            BLOCK_N=128
        )

        # 3) Reshape scores to [M, G, E] and compute per-group top-2 sums -> group_scores [M, G]
        # We need to load scores[m, n] into S[m, g, e]. Implement by recomputing from scores.
        group_scores = torch.empty((M, G), device=scores.device, dtype=scores.dtype)
        S = scores  # view as [M, G, E] via stride tricks: group = n // 32, expert = n % 32
        # Launch per-group kernel
        top2_per_group_kernel[(M,)](
            S, group_scores,
            M, G, E,
            S.stride(0), S.stride(1), S.stride(2),
            group_scores.stride(0), group_scores.stride(1),
            BLOCK_E=E  # process all 32 in one go
        )

        # 4) Per-token top-4 groups (arg-topk) -> group_idx [M, 4]
        group_idx = torch.empty((M, 4), device=scores.device, dtype=torch.int32)
        topk_groups_arg_kernel[(M,)](
            group_scores, group_idx,
            M, 4,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            K_CONST=4
        )

        # 5) Set group_mask [M, 8] to 1 at selected groups
        group_mask = torch.empty((M, G), device=scores.device, dtype=torch.float32)
        set_group_mask_kernel[(M,)](
            group_idx, group_mask,
            M, G,
            group_idx.stride(1), group_mask.stride(0), group_mask.stride(1),
            K_CONST=4, G_CONST=G
        )

        # 6) Expand group_mask to expanded_mask [M, N] (32 per group)
        expanded_mask = torch.empty((M, N), device=scores.device, dtype=torch.float32)
        expand_group_mask_kernel[(M, G)](
            group_mask, expanded_mask,
            M, G, E,
            group_mask.stride(0), group_mask.stride(1),
            expanded_mask.stride(0), expanded_mask.stride(1),
            K_CONST=G, E_CONST=E
        )

        # 7) Mask non-selected group scores to -inf in masked_scores [M, N]
        masked_scores = torch.empty_like(scores)
        mask_scores_kernel[(M,)](
            expanded_mask, scores, masked_scores,
            M, N,
            expanded_mask.stride(0), expanded_mask.stride(1),
            scores.stride(0), scores.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            BLOCK_N=256
        )

        # 8) Per-token top-8 experts from masked_scores -> topk_idx [M, 8]
        topk_idx = torch.empty((M, 8), device=scores.device, dtype=torch.int32)
        topk_experts_arg_kernel[(M,)](
            masked_scores, topk_idx,
            M, N, 8,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            K_CONST=8
        )

        # 9) Normalize and scale selected weights -> topk_weight [M, 8]
        topk_weight = torch.empty((M, 8), device=scores.device, dtype=scores.dtype)
        normalize_scale_kernel[(M,)](
            topk_idx, masked_scores, topk_weight,
            M, N, 8,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_factor=self.routed_scaling_factor,
            K_CONST=8
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
