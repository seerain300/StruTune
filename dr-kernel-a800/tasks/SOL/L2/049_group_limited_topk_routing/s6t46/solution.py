import torch
import triton
import triton.language as tl


@triton.jit
def matmul_logits_kernel(
    a_ptr,         # *f32, [M, K]
    b_ptr,         # *f32, [N, K]
    out_ptr,       # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)  # [BM, BK]
        b = tl.load(b_ptr + offs_n[None, :] * K + offs_k[:, None],
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
                    other=0.0)  # [BK, BN]
        acc += tl.dot(a, b)  # [BM, BN]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,    # *f32, [M, N]
    bias_ptr,      # *f32, [N]
    out_ptr,       # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    BLOCK_M: tl.constexpr,  # e.g., 64
    BLOCK_N: tl.constexpr,  # e.g., 64
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(logits_ptr + offs_m[:, None] * N + offs_n[None:], mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)  # [BLOCK_N]
    y = 1.0 / (1.0 + tl.exp(-x)) + b[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None:], y, mask=mask)


@triton.jit
def group_top2_sum_kernel(
    scores_ptr,     # *f32, [M, 8, 32]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
):
    pid = tl.program_id(0)  # one program per token
    if pid >= M:
        return
    for g in range(8):
        total = 0.0
        base = scores_ptr + pid * 256 + g * 32
        max1 = -float('inf')
        max2 = -float('inf')
        for i in range(EXPERTS_PER_GROUP):
            val = tl.load(base + i)
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
        total = max1 + max2
        tl.store(group_scores_ptr + pid * 8 + g, total)


@triton.jit
def topk_group_kernel(
    group_scores_ptr,   # *f32, [M, 8]
    group_idx_ptr,      # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,    # 4
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for g in range(8):
        val = tl.load(group_scores_ptr + pid * 8 + g)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = g
                break
    for t in range(K):
        tl.store(group_idx_ptr + pid * K + t, best_idxs[t])


@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,      # *int32, [M, 4]
    group_mask_ptr,     # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 4
    GROUP_COUNT: tl.constexpr,  # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(GROUP_COUNT):
        tl.store(group_mask_ptr + pid * GROUP_COUNT + g, 0.0)
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * GROUP_COUNT + g_idx, 1.0)


@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,     # *f32, [M, 8]
    masked_scores_ptr,  # *f32, [M, 256]
    M: tl.int32,
    N: tl.int32,        # 256
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    neg_inf = -float('inf')
    for n in range(N):
        g = n // 32
        if tl.load(group_mask_ptr + pid * 8 + g) != 1.0:
            tl.store(masked_scores_ptr + pid * N + n, neg_inf)


@triton.jit
def final_topk_kernel(
    scores_ptr,         # *f32, [M, 256]
    final_idx_ptr,      # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,        # 256
    K: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for n in range(N):
        val = tl.load(scores_ptr + pid * N + n)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break
    for t in range(K):
        tl.store(final_idx_ptr + pid * K + t, best_idxs[t])


@triton.jit
def gather_selected_scores_kernel(
    logits_ptr,         # *f32, [M, 256]
    final_idx_ptr,      # *int32, [M, 8]
    gathered_ptr,       # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(final_idx_ptr + pid * K + t)
        val = tl.load(logits_ptr + pid * 256 + idx)
        tl.store(gathered_ptr + pid * K + t, val)


@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,       # *f32, [M, 8]
    out_weight_ptr,     # *f32, [M, 8]
    M: tl.int32,
    scaling_factor: tl.float32,
    K: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    denom = 0.0
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t)
        denom += val
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t)
        normalized = val / denom
        tl.store(out_weight_ptr + pid * K + t, normalized * scaling_factor)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure inputs are contiguous
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()
        expert_bias = expert_bias.contiguous()

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]  # 128
        N = weight.shape[0]          # 256

        # 1) Compute logits = hidden_states @ weight.T using Triton
        logits = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        matmul_logits_kernel[grid](
            hidden_states, weight, logits,
            M, N, K,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # 2) Compute scores = sigmoid(logits) + expert_bias using Triton
        scores = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid2 = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        sigmoid_add_bias_kernel[grid2](
            logits, expert_bias, scores,
            M, N,
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )

        # 3) Compute group_scores [M, 8] via Triton (sum of top-2 per group)
        group_scores = torch.empty((M, 8), device=hidden_states.device, dtype=torch.float32)
        scores_reshaped = scores.view(M, 8, 32)
        grid3 = (M,)
        group_top2_sum_kernel[grid3](
            scores_reshaped, group_scores,
            M, EXPERTS_PER_GROUP=32,
        )

        # 4) Select top-4 groups per token using Triton
        group_idx = torch.empty((M, 4), device=hidden_states.device, dtype=torch.int32)
        grid4 = (M,)
        topk_group_kernel[grid4](
            group_scores, group_idx,
            M, K=4,
        )

        # 5) Build group_mask [M, 8] using Triton
        group_mask = torch.empty((M, 8), device=hidden_states.device, dtype=torch.float32)
        grid5 = (M,)
        build_group_mask_kernel[grid5](
            group_idx, group_mask,
            M, K=4, GROUP_COUNT=8,
        )

        # 6) Expand group_mask to [M, 256] and set non-selected groups to -inf in masked_scores using Triton
        masked_scores = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid6 = (M,)
        expand_and_set_ninf_kernel[grid6](
            group_mask, masked_scores,
            M, N,
        )

        # 7) Select final top-8 experts from masked_scores using Triton
        final_idx = torch.empty((M, 8), device=hidden_states.device, dtype=torch.int32)
        grid7 = (M,)
        final_topk_kernel[grid7](
            masked_scores, final_idx,
            M, N, K=8,
        )

        # 8) Gather original logits (pre-bias) for those 8 selected experts using Triton
        # We gather from logits (which is pre-bias


def run(*args):
    return ModelNew()(*args)
