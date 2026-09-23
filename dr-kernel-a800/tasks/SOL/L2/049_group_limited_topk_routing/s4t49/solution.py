import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_AxB_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [K, N], float32 (we pass weight.T with appropriate strides)
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,  # B[k, n]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        a_mask = (offs_m < M)[:, None] & (offs_k < K)[None, :]
        b_mask = (offs_k < K)[:, None] & (offs_n < N)[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,  # [M, N], float32
    B_ptr,  # [N], float32
    Y_ptr,  # [M, N], float32
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_bn,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    # One program per row segment
    row = pid_m
    if row >= M:
        return
    j = 0
    while j < N:
        x = tl.load(X_ptr + row * stride_xm + j * stride_xn)
        b = tl.load(B_ptr + j * stride_bn)
        y = tl.sigmoid(x) + b
        tl.store(Y_ptr + row * stride_ym + j * stride_yn, y)
        j += 1


@triton.jit
def _group_top2_sum_kernel(
    X_ptr,  # [M, N], float32
    GroupScores_ptr,  # [M, 8], float32
    M, N,
    stride_xm, stride_xn,
    stride_gs_m, stride_gs_n,
    EXP_PER_GRP: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # One program per token (row)
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Load scores row
    scores_row = tl.load(X_ptr + pid_m * stride_xm + tl.arange(0, N) * stride_xn)  # [N]

    # For each group g in [0..GROUPS-1]
    g = 0
    while g < GROUPS:
        start = g * EXP_PER_GRP
        # First top-1 within this group
        max1 = -float('inf')
        idx1 = 0
        j = 0
        while j < EXP_PER_GRP:
            col = start + j
            val = scores_row[col]
            if val > max1:
                max1 = val
                idx1 = col
            j += 1
        # Exclude idx1 by setting to -inf
        scores_row = scores_row
        scores_row[idx1] = -float('inf')
        max2 = -float('inf')
        idx2 = 0
        j = 0
        while j < EXP_PER_GRP:
            col = start + j
            val = scores_row[col]
            if val > max2:
                max2 = val
                idx2 = col
            j += 1
        group_score = max1 + max2
        tl.store(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_n, group_score)
        g += 1


@triton.jit
def _group_top4_select_kernel(
    GroupScores_ptr,  # [M, 8], float32
    GroupIdx_ptr,     # [M, 4], int32
    M, N,
    stride_gs_m, stride_gs_n,
    stride_gim, stride_gin,
    K: tl.constexpr,  # number of groups to select (4)
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    top = tl.zeros((K,), dtype=tl.int32) - 1
    val = tl.zeros((K,), dtype=tl.float32) - 1.0
    # Iterative top-k: k=4
    k_iter = 0
    while k_iter < K:
        best = -float('inf')
        best_idx = -1
        j = 0
        while j < 8:
            gs = tl.load(GroupScores_ptr + pid_m * stride_gs_m + j * stride_gs_n)
            if gs > best:
                best = gs
                best_idx = j
            j += 1
        top[k_iter] = best_idx
        val[k_iter] = best
        # Mark selected group by setting its score to -inf
        if best_idx >= 0:
            tl.store(GroupScores_ptr + pid_m * stride_gs_m + best_idx * stride_gs_n, -float('inf'))
        k_iter += 1

    # Store selected group indices
    k = 0
    while k < K:
        tl.store(GroupIdx_ptr + pid_m * stride_gim + k * stride_gin, top[k])
        k += 1


@triton.jit
def _final_top8_and_normalize_kernel(
    X_ptr,           # [M, N], float32
    GroupIdx_ptr,    # [M, 4], int32
    TopIdx_ptr,      # [M, 8], int32
    TopWT_ptr,       # [M, 8], float32 (to hold normalized weights)
    M, N,
    stride_xm, stride_xn,
    stride_gim, stride_gin,
    stride_tom, stride_to_n,
    stride_wm, stride_wn,
    routed_scaling_factor: tl.constexpr,
    EXP_PER_GRP: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK_FINAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    scores_row = tl.load(X_ptr + pid_m * stride_xm + tl.arange(0, N) * stride_xn)  # [N]

    # Build score_mask: select groups from GroupIdx
    # For each selected group, set corresponding 32 experts to 1, others 0
    # Then set non-selected group experts to -inf
    j = 0
    while j < N:
        # Check if j belongs to any selected group
        found = 0
        k = 0
        while k < 4:
            idx = tl.load(GroupIdx_ptr + pid_m * stride_gim + k * stride_gin)
            if (idx >= 0) and (idx < 8):
                start = idx * EXP_PER_GRP
                if (start <= j) and (j < start + EXP_PER_GRP):
                    found = 1
                    break
            k += 1
        # If found, do nothing (keep original score); else set to -inf
        if found == 0:
            scores_row[j] = -float('inf')
        j += 1

    # Select top-8 (k=8) indices iteratively and compute sum
    selected = tl.zeros((N,), dtype=tl.int1)
    top_idx = tl.zeros((TOPK_FINAL,), dtype=tl.int32) - 1
    top_val = tl.zeros((TOPK_FINAL,), dtype=tl.float32) - 1.0
    sum_vals = 0.0

    k = 0
    while k < TOPK_FINAL:
        max_val = -float('inf')
        max_pos = -1
        j = 0
        while j < N:
            val_j = scores_row[j]
            if (not selected[j]) and val_j > max_val:
                max_val = val_j
                max_pos = j
            j += 1
        # Store index
        tl.store(TopIdx_ptr + pid_m * stride_tom + k * stride_to_n, max_pos)
        tl.store(TopWT_ptr + pid_m * stride_wm + k * stride_wn, max_val)
        # Mark selected
        selected = selected | (tl.arange(0, N) == max_pos)
        # Accumulate for normalization
        sum_vals += max_val
        k += 1

    # Normalize and scale
    k = 0
    while k < TOPK_FINAL:
        val = tl.load(TopWT_ptr + pid_m * stride_wm + k * stride_wn)
        norm = val / sum_vals
        scaled = norm * routed_scaling_factor
        tl.store(TopWT_ptr + pid_m * stride_wm + k * stride_wn, scaled)
        k += 1


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants for routing
        self.num_experts = 256
        self.experts_per_group = 32
        self.n_group = 8
        self.top_k = 8
        self.topk_group = 4

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,   # [num_experts, hidden_dim]
        expert_bias: torch.Tensor,  # [num_experts]
        routed_scaling_factor: float,
    ):
        # Ensure on GPU and dtype float32
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = self.num_experts

        # 1) Compute logits = hidden_states @ weight.T using Triton GEMM
        # Prepare B = weight.T as [K, N] using weight strides
        # weight is [N, K], so B[k, n] = weight[n, k]
        A = hidden_states.contiguous().to(torch.float32)
        # We'll pass weight directly; Triton will interpret as B[k,n] via strides
        # But to use _matmul_AxB_kernel, we need a [K, N] tensor for B. Create by indexing weight:
        # Build B tensor [K, N] with correct strides: B_ptr[k, n] = weight[n, k]
        # Since Triton expects a contiguous B_ptr, we construct B explicitly:
        B_T = torch.empty((K, N), dtype=torch.float32, device=device)
        # Fill B_T with weight transposed data
        # weight is [N, K]; B_T[k, n] = weight[n, k]
        for k in range(K):
            B_T[k] = weight[:, k]
        C_logits = torch.empty((M, N), dtype=torch.float32, device=device)

        grid = (_ceil_div(M, 64), _ceil_div(N, 64))
        _matmul_AxB_kernel[grid](
            A, B_T, C_logits,
            M, N, K,
            A.stride(0), A.stride(1),
            B_T.stride(0), B_T.stride(1),
            C_logits.stride(0), C_logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # 2) Compute scores = sigmoid(logits) + expert_bias using Triton elementwise kernel
        X = C_logits  # logits
        bias = expert_bias.to(torch.float32)
        scores = torch.empty_like(X)
        grid_scores = (M,)
        _sigmoid_add_bias_kernel[grid_scores](
            X, bias, scores,
            M, N,
            X.stride(0), X.stride(1),
            scores.stride(0), scores.stride(1),
            bias.stride(0),
            BLOCK=256,
            num_warps=4,
        )

        # 3) Compute group_scores per token using Triton
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=device)
        grid_groups = (M,)
        _group_top2_sum_kernel[grid_groups](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXP_PER_GRP=self.experts_per_group,
            GROUPS=self.n_group,
            num_warps=1,
        )

        # 4) Select top-4 group indices per token using Triton
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=device)
        grid_g4 = (M,)
        _group_top4_select_kernel[grid_g4](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            K=self.topk_group,
            num_warps=1,
        )

        # 5) Final top-8 selection and normalization using Triton, write indices and weights
        top_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=device)
        top_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=device)

        grid_final = (M,)
        _final_top8_and_normalize_kernel[grid_final](
            scores, group_idx, top_idx, top_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            top_idx.stride(0), top_idx.stride(1),
            top_weight.stride(0), top_weight.stride(1),
            routed_scaling_factor,
            EXP_PER_GRP=self.experts_per_group,
            GROUPS=self.n_group,
            TOPK_FINAL=self.top_k,
            num_warps=4,
        )

        # Return as original: topk_idx int64, topk_weight float32
        return top_idx.to(torch.int64), top_weight


# Optional: helpers for evaluation harness (not used by ModelNew, but provided)
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    return ModelNew()(hidden_states, weight, expert_bias, routed_scaling_factor)


# Example input helpers (as in original):
def get_inputs():
    # Randomly generate input tensors based on the model architecture
    M = 2048  # or any provided workload
    K = 128   # example hidden_dim
    num_experts = 256
    hidden_states = torch.randn(M, K, device='cuda', dtype=torch.float32)
    weight = torch.randn(num_experts, K, device='cuda', dtype=torch.float32)  # nn.Linear style [out_features, in_features]
    expert_bias = torch.randn(num_experts, device='cuda', dtype=torch.float32)
    routed_scaling_factor = 1.0
    return [hidden_states, weight, expert_bias, routed_scaling_factor]


def get_init_inputs():
    # No special init inputs required
    return []


def run(*args):
    return ModelNew()(*args)
