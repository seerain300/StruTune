import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_AxB_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [K, N], float32
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # Pointers for A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k[None, :] * stride_ak)
        b_ptrs = B_ptr + (k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary
        a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
        b_mask = (k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Multiply-accumulate
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,  # [M, N], float32
    B_ptr,  # [N], float32
    Y_ptr,  # [M, N], float32
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program per row tile
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Iterate columns in chunks
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        # Pointers
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        # Sigmoid
        y = 1.0 / (1.0 + tl.exp(-x))
        # Add bias (broadcast over rows)
        b = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
        y = y + b[None, :]
        tl.store(y_ptrs, y, mask=mask)


@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,  # [M, N], float32
    group_scores_ptr,  # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    EXPERTS_PER_GROUP: tl.constexpr,  # must be 32
    BLOCK: tl.constexpr,              # chunk size for reduction (>= EXPERTS_PER_GROUP), e.g., 32
):
    # One program per row
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # We need to compute top-2 within each group of size EXPERTS_PER_GROUP = 32
    # We will iterate over groups directly using modulo and slice.
    # Load the full N columns for this row (N=256, EXPERTS_PER_GROUP=32, n_group=8)
    # but since Triton loop over groups is not allowed, we instead treat flattened indices.
    # Instead, we compute top-2 per group by reshaping logic via loops over columns.
    # Approach: for each group g in 0..7:
    #   start = g * EXPERTS_PER_GROUP
    #   max_val = -inf, max_idx
    #   second_val = -inf
    #   scan columns j=start..start+EXPERTS_PER_GROUP-1:
    #     if score > max_val: second_val = max_val; max_val = score; max_idx = j
    #     elif score > second_val: second_val = score
    #   group_scores[m, g] = max_val + second_val
    # Note: Triton does not allow dynamic loops over groups easily; we implement by scanning columns once and detect group per index.
    # However, since N is 256 and EXPERTS_PER_GROUP is 32, we can do this by scanning columns and updating per group.
    # Better approach: since N=256, we load entire row into a vector of length N, then chunk manually.
    # To keep within Triton constraints, we implement two passes per group via chunk iteration.
    # We'll iterate over columns in chunks of BLOCK and maintain per-group best2 per chunk, then merge.

    # For simplicity and correctness, we implement per-group passes using chunking and per-group scans:
    # We know EXPERTS_PER_GROUP=32, N=256, n_group=8, so we can perform 8 passes over scores for this row with group-local scans.
    # Triton allows while loops. We can loop over groups explicitly by using modulo on column index.
    for g in range(8):
        group_sum = 0.0
        start = g * EXPERTS_PER_GROUP
        # Scan columns within this group
        j = start
        max_val = -float('inf')
        max_idx = -1
        while j < start + EXPERTS_PER_GROUP:
            # Load score for this row and column j
            s = tl.load(scores_ptr + pid_m * stride_sm + j * stride_sn)
            # Update top-1
            if s > max_val:
                second_val = max_val
                max_val = s
                max_idx = j
            else:
                # Update second if needed
                if s > second_val:
                    second_val = s
            j += 1
        group_sum += max_val
        group_sum += second_val
        # Store group score
        tl.store(group_scores_ptr + pid_m * stride_gm + g * stride_gn, group_sum)


@triton.jit
def _group_top4_select_kernel(
    group_scores_ptr,  # [M, 8], float32
    group_idx_ptr,     # [M, 4], int32
    M, N,
    stride_gsm, stride_gsn,
    stride_gim, stride_gin,
    BLOCK: tl.constexpr,  # e.g., 8
):
    # One program per row
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Initialize selected flags and selected values
    selected = tl.zeros((4,), dtype=tl.int32)  # not used here
    selected_vals = tl.full((4,), -float('inf'), dtype=tl.float32)
    selected_idx = tl.zeros((4,), dtype=tl.int32)

    # Iteratively select top-4
    for pos in range(4):
        best_val = -float('inf')
        best_idx = -1
        # Scan all 8 groups
        for g in range(8):
            val = tl.load(group_scores_ptr + pid_m * stride_gsm + g * stride_gsn)
            if val > best_val:
                best_val = val
                best_idx = g
        # Store selected index
        tl.store(group_idx_ptr + pid_m * stride_gim + pos * stride_gin, best_idx)
        # Exclude it by setting to -inf
        tl.store(group_scores_ptr + pid_m * stride_gsm + best_idx * stride_gsn, -float('inf'))


@triton.jit
def _final_top8_and_normalize_kernel(
    scores_ptr,         # [M, N], float32
    group_idx_ptr,      # [M, 4], int32
    topk_idx_ptr,       # [M, 8], int32
    topk_weight_ptr,    # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gim, stride_gin,
    stride_tkm, stride_tkn,
    stride_wm, stride_wn,
    routed_scaling_factor: tl.constexpr,
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
):
    # One program per row
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Build score_mask: set non-selected groups to -inf
    # For each selected group g in 0..3: for j in [g*EXPERTS_PER_GROUP : (g+1)*EXPERTS_PER_GROUP], scores[m, j] = -inf
    for pos in range(4):
        g = tl.load(group_idx_ptr + pid_m * stride_gim + pos * stride_gin)
        start = g * EXPERTS_PER_GROUP
        for j in range(EXPERTS_PER_GROUP):
            col = start + j
            if col < N:
                orig = tl.load(scores_ptr + pid_m * stride_sm + col * stride_sn)
                tl.store(scores_ptr + pid_m * stride_sm + col * stride_sn, -float('inf'))

    # Now select top-8 via iterative argmax
    for k in range(8):
        best_val = -float('inf')
        best_idx = -1
        for j in range(0, N, 1):
            val = tl.load(scores_ptr + pid_m * stride_sm + j * stride_sn)
            # Triton doesn't support j in tl.arange in kernel, but N is small (256); we can loop per element
            if val > best_val:
                best_val = val
                best_idx = j
        # Store selected index
        tl.store(topk_idx_ptr + pid_m * stride_tkm + k * stride_tkn, best_idx)
        # Exclude it by setting to -inf
        tl.store(scores_ptr + pid_m * stride_sm + best_idx * stride_sn, -float('inf'))

    # Compute normalized weights for the 8 selected experts
    # Gather selected scores
    selected_scores = tl.zeros((8,), dtype=tl.float32)
    for k in range(8):
        idx = tl.load(topk_idx_ptr + pid_m * stride_tkm + k * stride_tkn)
        val = tl.load(scores_ptr + pid_m * stride_sm + idx * stride_sn)
        selected_scores[k] = val
    # L1 normalize
    sum_val = 0.0
    for k in range(8):
        sum_val += selected_scores[k]
    norm = 1.0 / (sum_val + 1e-20)
    for k in range(8):
        w = selected_scores[k] * norm * routed_scaling_factor
        tl.store(topk_weight_ptr + pid_m * stride_wm + k * stride_wn, w)


# ModelNew entry point
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; constants match original
        self.num_experts = 256
        self.experts_per_group = 32
        self.n_group = 8
        self.top_k = 8
        self.topk_group = 4

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        hidden_states: [M, K] float32 CUDA
        weight: [num_experts, K] float32 CUDA (nn.Linear weight, we will use weight.T for GEMM)
        expert_bias: [num_experts] float32 CUDA
        routed_scaling_factor: float
        Returns:
        - topk_idx: [M, 8] int64
        - topk_weight: [M, 8] float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors"
        assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32, "Use float32"
        M, K = hidden_states.shape
        N = self.num_experts  # 256
        assert N == weight.shape[0], "weight shape mismatch: weight should be [num_experts, K]"
        assert K == weight.shape[1], "weight shape mismatch: weight should be [num_experts, K] with hidden_dim == K"
        assert expert_bias.numel() == N, "expert_bias length must be 256"
        assert self.experts_per_group * self.n_group == N, "Grouping mismatch: 32*8 != 256"

        # Ensure contiguity
        hidden_states = hidden_states.contiguous()
        weight_T = weight.transpose(0, 1).contiguous()  # [K, N]
        expert_bias = expert_bias.contiguous()

        # 1) Compute logits = hidden_states @ weight_T
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        _matmul_AxB_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
            hidden_states, weight_T, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # 2) scores = sigmoid(logits) + expert_bias
        scores = torch.empty_like(logits)
        _sigmoid_add_bias_kernel[(M,)](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4,
        )

        # 3) Group top-2 sum: group_scores [M, 8]
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden_states.device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXPERTS_PER_GROUP=self.experts_per_group,
            BLOCK=32,
            num_warps=4,
        )

        # 4) Select top-4 groups per token: group_idx [M, 4]
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden_states.device)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, N,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            BLOCK=8,
            num_warps=4,
        )

        # 5) Final top-8 selection and normalization: topk_idx [M, 8], topk_weight [M, 8]
        topk_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden_states.device)
        _final_top8_and_normalize_kernel[(M,)](
            scores, group_idx, topk_idx, topk_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_scaling_factor,
            EXPERTS_PER_GROUP=self.experts_per_group,
            num_warps=4,
        )

        # Return as in original: int64 indices and float32 weights
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
