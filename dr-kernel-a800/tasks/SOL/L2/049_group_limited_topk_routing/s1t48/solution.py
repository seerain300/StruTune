import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Matmul for logits = hidden @ weight^T
# hidden: [M, K], weight: [N, K], out: [M, N]
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Pointers for current tile
        A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B_tile_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        A_tile = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)
        B_tile = tl.load(B_tile_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store result
    C_tile_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=c_mask)


# Kernel 2: Sigmoid elementwise
@triton.jit
def _sigmoid_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptrs, y, mask=mask)


# Kernel 3: Add bias (vector of length N) to each column
@triton.jit
def _add_bias_kernel(
    S_ptr, B_ptr, Out_ptr,
    M, N,
    stride_sm, stride_sn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    s_ptrs = S_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    out_ptrs = Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    s = tl.load(s_ptrs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0)
    out = s + b[None, :]
    tl.store(out_ptrs, out, mask=mask)


# Kernel 4: Compute per-group top-2 sum for groups of 32 experts
@triton.jit
def _group_top2_sum_kernel(
    S_ptr, GroupScores_ptr,
    M, N, n_group, experts_per_group,
    stride_sm, stride_sn,
    stride_gs_m, stride_gs_n,  # stride_gs_n is not used because GroupScores is 1D per row
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    # Each program processes one row (token)
    offs_m = pid_m
    # Loop over groups
    for g in range(0, n_group):
        base = g * experts_per_group
        # Initialize top-1 and top-2 for this group
        top1 = -1.0e20
        top2 = -1.0e20
        # Iterate over 32 experts in the group
        for e in range(0, experts_per_group):
            idx = base + e
            val = tl.load(S_ptr + offs_m * stride_sm + idx * stride_sn)
            # Update top-2
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        group_score = top1 + top2
        tl.store(GroupScores_ptr + offs_m * stride_gs_m + g * stride_gs_n, group_score)


# Kernel 5: Select top-4 groups per token (iterative argmax)
@triton.jit
def _select_top4_groups_kernel(
    GroupScores_ptr, GroupIdx_ptr,
    M, n_group,
    stride_gs_m, stride_gs_n,
    BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    top = -1.0e20
    # indices are ints, but we keep values in fp32; we just pick argmax 4 times without duplicates
    for t in range(0, 4):
        maxv = -1.0e20
        max_idx = 0
        # Scan groups and find the next maximum (excluding previously selected)
        for g in range(0, n_group):
            v = tl.load(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_n)
            if v > maxv:
                maxv = v
                max_idx = g
        # Record index
        tl.store(GroupIdx_ptr + pid_m * 4 + t, max_idx)
        # Set it to -inf so it won’t be selected again
        tl.store(GroupScores_ptr + pid_m * stride_gs_m + max_idx * stride_gs_n, -1.0e20)


# Kernel 6: Build expert-level mask: for each token, set 1 for selected groups’ 32 experts, 0 otherwise
@triton.jit
def _build_group_mask_kernel(
    GroupIdx_ptr, ScoreMask_ptr,
    M, n_group, experts_per_group, topk_group,
    stride_sm_m, stride_sm_n,
    BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    # topk_group is 4 in this setup
    for t in range(0, topk_group):
        g = tl.load(GroupIdx_ptr + pid_m * topk_group + t)  # int32
        base = g * experts_per_group
        for e in range(0, experts_per_group):
            idx = base + e
            tl.store(ScoreMask_ptr + pid_m * stride_sm_m + idx * stride_sm_n, 1)
    # Zero out remaining entries (in case of any overrun, though we know exactly 32*4 set)
    total_set = topk_group * experts_per_group
    for e in range(0, n_group * experts_per_group):
        if e < total_set:
            continue
        tl.store(ScoreMask_ptr + pid_m * stride_sm_m + e * stride_sm_n, 0)


# Kernel 7: Masked fill: set masked_scores[i, e] = -inf if score_mask[i, e] == 0, else keep scores_for_routing[i, e]
@triton.jit
def _masked_fill_kernel(
    S_ptr, ScoreMask_ptr, Masked_ptr,
    M, N,
    stride_sm_m, stride_sm_n,
    stride_ms_m, stride_ms_n,
    NEG_INF: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    s_ptrs = S_ptr + offs_m[:, None] * stride_sm_m + offs_n[None, :] * stride_sm_n
    mask_ptrs = ScoreMask_ptr + offs_m[:, None] * stride_sm_m + offs_n[None, :] * stride_sm_n
    ms_ptrs = Masked_ptr + offs_m[:, None] * stride_ms_m + offs_n[None, :] * stride_ms_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    s = tl.load(s_ptrs, mask=mask, other=0.0)
    sm = tl.load(mask_ptrs, mask=mask, other=0)  # 0 or 1
    # -inf if mask==0 else s
    out = tl.where(sm == 0, NEG_INF, s)
    tl.store(ms_ptrs, out, mask=mask)


# Kernel 8: Final top-8 selection from masked_scores (iterative argmax)
@triton.jit
def _final_top8_selection_kernel(
    Masked_ptr, TopIdx_ptr, TopVals_ptr,
    M, N,
    stride_ms_m, stride_ms_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    for r in range(0, 8):
        maxv = -1.0e20
        max_idx = 0
        for n in range(0, N):
            val = tl.load(Masked_ptr + pid_m * stride_ms_m + n * stride_ms_n)
            if val > maxv:
                maxv = val
                max_idx = n
        tl.store(TopIdx_ptr + pid_m * 8 + r, max_idx)
        tl.store(TopVals_ptr + pid_m * 8 + r, maxv)
        # Set selected to -inf to avoid re-selection
        tl.store(Masked_ptr + pid_m * stride_ms_m + max_idx * stride_ms_n, -1.0e20)


# Kernel 9: Normalize and scale top-8 weights
@triton.jit
def _normalize_and_scale_kernel(
    TopVals_ptr, TopWeight_ptr,
    M, top_k,
    SCALE,
    BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    total = 0.0
    for r in range(0, top_k):
        val = tl.load(TopVals_ptr + pid_m * top_k + r)
        total += val
    for r in range(0, top_k):
        val = tl.load(TopVals_ptr + pid_m * top_k + r)
        w = val / (total + 1e-20) * SCALE
        tl.store(TopWeight_ptr + pid_m * top_k + r, w)


class ModelNew(nn.Module):
    # Fixed constants from the original code
    def __init__(self, hidden_dim: int, routed_scaling_factor: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = 32
        self.top_k = 8
        self.topk_group = 4
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward: allocate and launch kernels
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        # Ensure contiguity and dtype FP32
        hidden = hidden_states.contiguous().to(torch.float32)       # [M, K]
        weight = weight.contiguous().to(torch.float32)              # [N, K] (N=256, K=hidden_dim)
        bias = expert_bias.contiguous().to(torch.float32)           # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch: weight must be [256, hidden_dim] and hidden [M, hidden_dim]."

        # Intermediate and output tensors
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)        # [M, N]
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)        # [M, N]
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)  # [M, N]
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)  # [M, 8]
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden.device)    # [M, 4]
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)          # [M, N]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)      # [M, N]
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)     # [M, 8]
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)   # [M, 8]
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device) # [M, 8]

        # Launch Triton kernels
        # 1) Matmul for logits = hidden @ weight^T
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            hidden, weight, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight.stride(1), weight.stride(0),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) Sigmoid
        _sigmoid_kernel[(M, N)](
            logits, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 3) Add bias
        _add_bias_kernel[(M, N)](
            scores, bias, scores_for_routing,
            M, N,
            scores.stride(0), scores.stride(1),
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 4) Group top-2 sum: [M, 8]
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, self.n_group, self.experts_per_group,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 5) Select top-4 groups per token
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            BLOCK_M=BLOCK_M,
        )

        # 6) Build expert-level mask
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, self.n_group, self.experts_per_group, self.topk_group,
            score_mask.stride(0), score_mask.stride(1),
            BLOCK_M=BLOCK_M,
        )

        # 7) Masked fill: set non-selected experts to -inf
        NEG_INF = -1.0e20
        _masked_fill_kernel[(M, N)](
            scores_for_routing, score_mask, masked_scores,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            NEG_INF=NEG_INF,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 8) Final top-8 selection from masked_scores
        _final_top8_selection_kernel[(M, N)](
            masked_scores, top8_idx, top8_vals,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 9) Normalize and scale top-8 weights
        _normalize_and_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, self.top_k,
            self.routed_scaling_factor,
            BLOCK_M=BLOCK_M,
        )

        # Return indices and weights
        # top8_idx shape [M, 8], topk_weight shape [M, 8]
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
