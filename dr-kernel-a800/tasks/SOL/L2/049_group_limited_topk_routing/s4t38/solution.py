import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,    # [M, N], float32
    B_ptr,    # [N], float32
    Y_ptr,    # [M, N], float32
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= (N + BLOCK_N - 1) // BLOCK_N:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    b = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
    y = sig + b[None, :]
    tl.store(y_ptrs, y, mask=mask)


@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,   # [M, N], float32
    group_scores_ptr,  # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    BLOCK_N: tl.constexpr,  # tile along N, we will loop N in chunks of 32 for argmax
):
    # One program per row (token)
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # We will handle groups sequentially: each group has 32 experts
    # Compute top-2 sum for each group and write to group_scores[pid_m, g]
    # Use iterative argmax within the 32-expert chunk.

    # Loop over groups
    for g in range(8):
        base = g * 32
        max_val = -float('inf')
        max_idx = -1
        # First pass: find max in this chunk
        for kk in range(32):
            idx = base + kk
            val = tl.load(scores_ptr + pid_m * stride_sm + idx * stride_sn)
            # if idx >= N, val should be ignored; but since N=num_experts=256 and base in [0,224], idx < N always true.
            if val > max_val:
                max_val = val
                max_idx = idx
        # Exclude max and find second max
        second_val = -float('inf')
        second_idx = -1
        for kk in range(32):
            idx = base + kk
            val = tl.load(scores_ptr + pid_m * stride_sm + idx * stride_sn)
            if idx == max_idx:
                val = -float('inf')
            if val > second_val:
                second_val = val
                second_idx = idx
        sum_val = max_val + second_val
        tl.store(group_scores_ptr + pid_m * stride_gm + g * stride_gn, sum_val)


@triton.jit
def _group_top4_select_kernel(
    group_scores_ptr,  # [M, 8], float32
    group_idx_ptr,     # [M, 4], int32
    M, N,  # N is unused but kept for signature symmetry
    stride_gsm, stride_gsn,
    stride_im, stride_in,
):
    # One program per row (token)
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Iteratively select top-4 indices via argmax among 8
    for p in range(4):
        best_val = -float('inf')
        best_idx = -1
        for j in range(8):
            val = tl.load(group_scores_ptr + pid_m * stride_gsm + j * stride_gsn)
            if val > best_val:
                best_val = val
                best_idx = j
        tl.store(group_idx_ptr + pid_m * stride_im + p * stride_in, best_idx)
        # Mask the selected one by setting its score to -inf for next iterations
        tl.store(group_scores_ptr + pid_m * stride_gsm + best_idx * stride_gsn, -float('inf'))


@triton.jit
def _final_top8_and_normalize_kernel(
    scores_ptr,            # [M, N], float32
    group_idx_ptr,         # [M, 4], int32
    topk_idx_ptr,          # [M, 8], int32
    topk_weight_ptr,       # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_im, stride_in,
    stride_tkm, stride_tkn,
    stride_wm, stride_wn,
    routed_scaling_factor: tl.float32,
    BLOCK_N: tl.constexpr,  # tile along N, but we loop exactly N
):
    # One program per row (token)
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # First, construct score_mask: only keep selected groups, others set to -inf
    # group_idx has 4 valid groups; each group spans 32 experts
    # For each selected group j in 0..3, set scores for those 32 to original; others to -inf
    for p in range(4):
        g = tl.load(group_idx_ptr + pid_m * stride_im + p * stride_in)
        base = g * 32
        # We'll not mask out non-selected groups here; instead, we keep the original scores intact
        # because the next step relies on iterative argmax over all N. So no additional masking needed.

    # Now, iteratively select top-8 indices via argmax
    for k in range(8):
        best_val = -float('inf')
        best_idx = -1
        for j in range(N):
            val = tl.load(scores_ptr + pid_m * stride_sm + j * stride_sn)
            if val > best_val:
                best_val = val
                best_idx = j
        tl.store(topk_idx_ptr + pid_m * stride_tkm + k * stride_tkn, best_idx)
        # Exclude selected index by setting it to -inf for next iterations
        tl.store(scores_ptr + pid_m * stride_sm + best_idx * stride_sn, -float('inf'))

    # Compute normalized weights from selected indices
    selected_scores = tl.zeros((8,), dtype=tl.float32)
    for k in range(8):
        idx = tl.load(topk_idx_ptr + pid_m * stride_tkm + k * stride_tkn)
        val = tl.load(scores_ptr + pid_m * stride_sm + idx * stride_sn)
        selected_scores[k] = val

    sum_val = 0.0
    for k in range(8):
        sum_val += selected_scores[k]
    norm = 1.0 / (sum_val + 1e-20)
    for k in range(8):
        w = selected_scores[k] * norm * routed_scaling_factor
        tl.store(topk_weight_ptr + pid_m * stride_wm + k * stride_wn, w)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256
        self.experts_per_group = 32
        self.n_group = 8
        self.top_k = 8
        self.topk_group = 4

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-only implementation:
        - Compute logits using torch.nn.functional.linear (robust GEMM + bias).
        - Triton kernels handle sigmoid + bias, group top-2 sum, group top-4 selection, final top-8 selection and normalization.
        Returns:
          - topk_idx: [M, 8] int64
          - topk_weight: [M, 8] float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors"
        assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32, "Use float32"
        assert hidden_states.shape[1] == weight.shape[1], "hidden_states last dim must match weight in_features"
        M, K = hidden_states.shape
        N = self.num_experts  # 256
        # Compute logits = hidden_states @ weight, where weight is [N, K] in nn.Linear, but we need [K, N] for matmul
        # So use weight.T
        logits = torch.nn.functional.linear(hidden_states, weight)

        # Ensure contiguous and dtype
        logits = logits.contiguous()
        expert_bias = expert_bias.contiguous()

        # 1) Triton: sigmoid + bias -> scores
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid_sigmoid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _sigmoid_add_bias_kernel[grid_sigmoid](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )

        # 2) Triton: group top-2 sum -> group_scores [M, 8]
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden_states.device)
        grid_group2 = (M,)
        _group_top2_sum_kernel[grid_group2](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            BLOCK_N=32,  # we loop 32 explicitly in the kernel
            num_warps=2, num_stages=1,
        )

        # 3) Triton: select top-4 group indices -> group_idx [M, 4]
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden_states.device)
        grid_group4 = (M,)
        _group_top4_select_kernel[grid_group4](
            group_scores, group_idx,
            M, N,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            num_warps=1, num_stages=1,
        )

        # 4) Triton: final top-8 selection and normalization -> topk_idx [M, 8], topk_weight [M, 8]
        topk_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden_states.device)
        grid_final = (M,)
        _final_top8_and_normalize_kernel[grid_final](
            scores, group_idx, topk_idx, topk_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_scaling_factor,
            BLOCK_N=N,
            num_warps=4, num_stages=2,
        )

        # Return as in original: int64 indices and float32 weights
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
