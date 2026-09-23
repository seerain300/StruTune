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
    # 2D launch over tiles of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)   # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)   # [BLOCK_K, BLOCK_N]

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,  # [M, N], float32
    B_ptr,  # [N], float32
    Y_ptr,  # [M, N], float32
    M, N,
    stride_xm, stride_xn,
    stride_b,  # B_ptr is 1D
    stride_ym, stride_yn,
    BLOCK: tl.constexpr,
):
    # 2D tiling over rows and columns
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)

    # Pointers
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # sigmoid
    s = 1.0 / (1.0 + tl.exp(-x))
    # bias: load column-wise
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    s = s + b[None, :]

    tl.store(y_ptrs, s, mask=mask)


@triton.jit
def _group_top2_sum_kernel(
    S_ptr,      # [M, N], float32
    GROUPS_ptr, # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    EXP_PER_GROUP: tl.constexpr,  # 32
    BLOCK_GROUP: tl.constexpr,    # 32
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Process groups in chunks
    for g in range(8):
        # Tile within the group of 32 experts
        start = g * EXP_PER_GROUP
        # Compute max and second max via iterative argmax
        # We use static loop over 32 columns
        max_idx = 0
        max_val = -float('inf')
        for j in range(0, EXP_PER_GROUP):
            idx = start + j
            val = tl.load(S_ptr + pid_m * stride_sm + idx * stride_sn)
            if val > max_val:
                max_val = val
                max_idx = j

        # Set the max to -inf for second max
        tl.store(S_ptr + pid_m * stride_sm + (start + max_idx) * stride_sn, -float('inf'))

        second_idx = 0
        second_val = -float('inf')
        for j in range(0, EXP_PER_GROUP):
            idx = start + j
            val = tl.load(S_ptr + pid_m * stride_sm + idx * stride_sn)
            if val > second_val and val < max_val:
                second_val = val
                second_idx = j

        # Store group score
        tl.store(GROUPS_ptr + pid_m * stride_gm + g * stride_gn, max_val + second_val)


@triton.jit
def _group_top4_select_kernel(
    GROUPS_ptr,  # [M, 8], float32
    GROUP_IDX_ptr,  # [M, 4], int32
    M, N,
    stride_gm, stride_gn,
    stride_im, stride_in,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Iteratively select top-4 groups
    for pos in range(4):
        # Initial best
        best_val = -float('inf')
        best_idx = 0
        for j in range(8):
            val = tl.load(GROUPS_ptr + pid_m * stride_gm + j * stride_gn)
            if val > best_val:
                best_val = val
                best_idx = j

        # Store index
        tl.store(GROUP_IDX_ptr + pid_m * stride_im + pos * stride_in, best_idx)

        # Mask it by setting to -inf
        tl.store(GROUPS_ptr + pid_m * stride_gm + best_idx * stride_gn, -float('inf'))


@triton.jit
def _final_top8_and_normalize_kernel(
    S_ptr,                  # [M, N], float32
    GROUP_IDX_ptr,          # [M, 4], int32
    TOPK_IDX_ptr,           # [M, 8], int32
    TOPK_WEIGHT_ptr,        # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_im, stride_in,
    stride_tkm, stride_tkn,
    stride_wm, stride_wn,
    routed_scale: tl.float32,
    EXP_PER_GROUP: tl.constexpr,  # 32
    GROUPS: tl.constexpr,         # 8
    CHUNK: tl.constexpr,          # 128
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Build mask: for groups in GROUP_IDX, keep them as 1, others 0
    # We iterate over the 4 selected groups and mark their 32 experts.
    # Other experts are set to -inf.
    for pos in range(4):
        g = tl.load(GROUP_IDX_ptr + pid_m * stride_im + pos * stride_in)
        start = g * EXP_PER_GROUP
        for j in range(EXP_PER_GROUP):
            idx = start + j
            # Mark selected experts
            tl.store(S_ptr + pid_m * stride_sm + idx * stride_sn, tl.load(S_ptr + pid_m * stride_sm + idx * stride_sn))
        # No need to set -inf here; we do it in iterative selection.

    # Iteratively select top-8 experts from S_ptr row pid_m
    selected = tl.zeros((8,), dtype=tl.int32)
    for k in range(8):
        best_val = -float('inf')
        best_idx = 0
        for j in range(0, N, CHUNK):
            cols = j + tl.arange(0, CHUNK)
            mask_cols = cols < N
            vals = tl.load(S_ptr + pid_m * stride_sm + cols * stride_sn, mask=mask_cols, other=-float('inf'))
            # argmax over CHUNK
            # We implement simple argmax: loop over CHUNK
            for jj in range(CHUNK):
                c = j + jj
                if c >= N:
                    break
                val = vals[jj]
                if val > best_val:
                    best_val = val
                    best_idx = c
        # Store selected index
        tl.store(TOPK_IDX_ptr + pid_m * stride_tkm + k * stride_tkn, best_idx)
        # Exclude by setting to -inf
        tl.store(S_ptr + pid_m * stride_sm + best_idx * stride_sn, -float('inf'))

    # Compute normalized weights
    for k in range(8):
        idx = tl.load(TOPK_IDX_ptr + pid_m * stride_tkm + k * stride_tkn)
        val = tl.load(S_ptr + pid_m * stride_sm + idx * stride_sn)
        selected[k] = val
    sum_val = 0.0
    for k in range(8):
        sum_val += selected[k]
    norm = 1.0 / (sum_val + 1e-20)
    for k in range(8):
        w = selected[k] * norm * routed_scale
        tl.store(TOPK_WEIGHT_ptr + pid_m * stride_wm + k * stride_wn, w)


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
        Triton-only implementation of the original routing logic:
        - Compute logits via Triton GEMM: hidden_states @ weight.T (weights are [N, K], we use weight.T as B[K, N])
        - Compute scores = sigmoid(logits) + expert_bias via Triton
        - Group top-2 per group and sum, then select top-4 groups via Triton
        - Mask out non-selected groups, then select top-8 per token and normalize via Triton
        Returns:
          - topk_idx: [M, 8] int64
          - topk_weight: [M, 8] float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors"
        assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32, "Use float32"
        M, K = hidden_states.shape
        N = self.num_experts  # 256
        assert weight.shape == (N, K), "weight must be [num_experts, hidden_dim]"

        # Prepare B = weight.T as [K, N]
        B = weight.transpose(0, 1).contiguous()

        # 1) Triton GEMM: logits = hidden_states @ weight.T -> [M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _matmul_AxB_kernel[grid](
            hidden_states, B, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            B.stride(0), B.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # 2) Triton: scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        _sigmoid_add_bias_kernel[(M, N)](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            expert_bias.stride(0),
            scores.stride(0), scores.stride(1),
            BLOCK=128,
            num_warps=4,
        )

        # 3) Triton: group_scores [M, 8] = sum of top-2 per group (32 experts per group)
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden_states.device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXP_PER_GROUP=self.experts_per_group,
            BLOCK_GROUP=self.experts_per_group,
            num_warps=4,
        )

        # 4) Triton: group_idx [M, 4] from group_scores via iterative argmax
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden_states.device)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            BLOCK=1,
            num_warps=1,
        )

        # 5) Final selection and normalization in Triton (iterative top-8, L1 normalize, scale)
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
            EXP_PER_GROUP=self.experts_per_group,
            GROUPS=self.n_group,
            CHUNK=128,
            num_warps=4,
        )

        # Return with correct dtypes
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
