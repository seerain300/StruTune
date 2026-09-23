import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Constants
NUM_EXPERTS = 256
N_GROUP = 8
EXP_PER_GROUP = NUM_EXPERTS // N_GROUP  # 32
TOP_K = 8
TOPK_GROUP = 4


# Triton kernel: matmul logits = hidden @ weight^T
# hidden: [M, K], weight: [N, K] (num_experts, hidden_dim), out: [M, N]
@triton.jit
def _matmul_rowwise_kernel(
    hidden_ptr,        # *f32, [M, K]
    weight_ptr,        # *f32, [N, K]
    out_ptr,           # *f32, [M, N]
    M, K, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Load A tile: hidden[offs_m, offs_k]
        a_ptrs = hidden_ptr + (offs_m[:, None] * K) + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # Load B tile: weight[offs_n, offs_k]
        b_ptrs = weight_ptr + (offs_n[:, None] * K) + offs_k[None, :]
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    # Store acc to out[offs_m, offs_n]
    out_ptrs = out_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# Triton kernel: elementwise sigmoid on input X -> Y
@triton.jit
def _sigmoid_kernel(X_ptr, Y_ptr, M, N):
    row = tl.program_id(0)
    for col in range(0, N):
        val = tl.load(X_ptr + row * N + col)
        y = 1.0 / (1.0 + tl.exp(-val))
        tl.store(Y_ptr + row * N + col, y)


# Triton kernel: add bias (size N) to X (M, N) -> Y (M, N)
@triton.jit
def _add_bias_kernel(X_ptr, Bias_ptr, Y_ptr, M, N):
    row = tl.program_id(0)
    for col in range(0, N):
        x = tl.load(X_ptr + row * N + col)
        b = tl.load(Bias_ptr + col)
        y = x + b
        tl.store(Y_ptr + row * N + col, y)


# Triton kernel: group top-2 sum for each token
# Input: scores_for_routing (M, NUM_EXPERTS), Output: group_scores (M, N_GROUP)
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,         # [M, NUM_EXPERTS] f32
    groupS_ptr,         # [M, N_GROUP] f32
    M, NUM_EXPERTS, N_GROUP, EXP_PER_GROUP,
):
    row = tl.program_id(0)
    for g in range(0, N_GROUP):
        base = g * EXP_PER_GROUP
        max1 = tl.full((), -1.0e20, dtype=tl.float32)
        max2 = tl.full((), -1.0e20, dtype=tl.float32)
        for ee in range(0, EXP_PER_GROUP):
            e = base + ee
            val = tl.load(scores_ptr + row * NUM_EXPERTS + e)
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
        sum2 = max1 + max2
        tl.store(groupS_ptr + row * N_GROUP + g, sum2)


# Triton kernel: select top TOPK_GROUP groups per token (argmax over group_scores)
@triton.jit
def _select_top4_groups_kernel(
    groupS_ptr,         # [M, N_GROUP] f32
    groupIdx_ptr,       # [M, TOPK_GROUP] int32
    M, N_GROUP, TOPK_GROUP,
):
    row = tl.program_id(0)
    for t in range(0, TOPK_GROUP):
        best_val = tl.full((), -1.0e20, dtype=tl.float32)
        best_idx = tl.zeros((), dtype=tl.int32)
        for g in range(0, N_GROUP):
            val = tl.load(groupS_ptr + row * N_GROUP + g)
            take = val > best_val
            best_val = tl.where(take, val, best_val)
            best_idx = tl.where(take, g, best_idx)
        tl.store(groupIdx_ptr + row * TOPK_GROUP + t, best_idx)


# Triton kernel: build expert-level mask (1 for selected group's 32 experts, 0 otherwise)
@triton.jit
def _build_group_mask_kernel(
    groupIdx_ptr,       # [M, TOPK_GROUP] int32
    scoreMask_ptr,      # [M, NUM_EXPERTS] int32
    M, TOPK_GROUP, NUM_EXPERTS, EXP_PER_GROUP,
):
    row = tl.program_id(0)
    for t in range(0, TOPK_GROUP):
        g = tl.load(groupIdx_ptr + row * TOPK_GROUP + t)
        base = g * EXP_PER_GROUP
        for ee in range(0, EXP_PER_GROUP):
            e = base + ee
            tl.store(scoreMask_ptr + row * NUM_EXPERTS + e, 1)


# Triton kernel: masked fill: set masked_scores[row, e] = -inf if scoreMask[row, e] == 0, else keep scores
@triton.jit
def _masked_fill_kernel(
    scores_ptr,         # [M, NUM_EXPERTS] f32
    scoreMask_ptr,      # [M, NUM_EXPERTS] int32
    masked_ptr,         # [M, NUM_EXPERTS] f32
    M, NUM_EXPERTS,
):
    row = tl.program_id(0)
    for col in range(0, NUM_EXPERTS):
        s = tl.load(scores_ptr + row * NUM_EXPERTS + col)
        msk = tl.load(scoreMask_ptr + row * NUM_EXPERTS + col)
        neg_inf = -1.0e20
        new_val = tl.where(msk == 1, s, neg_inf)
        tl.store(masked_ptr + row * NUM_EXPERTS + col, new_val)


# Triton kernel: final top-8 selection on masked_scores -> topIdx [M, TOP_K], topVals [M, TOP_K]
@triton.jit
def _final_top8_selection_kernel(
    masked_ptr,         # [M, NUM_EXPERTS] f32
    topIdx_ptr,         # [M, TOP_K] int32
    topVals_ptr,        # [M, TOP_K] f32
    M, NUM_EXPERTS, TOP_K,
):
    row = tl.program_id(0)
    for t in range(0, TOP_K):
        best_val = tl.full((), -1.0e20, dtype=tl.float32)
        best_idx = tl.zeros((), dtype=tl.int32)
        for e in range(0, NUM_EXPERTS):
            val = tl.load(masked_ptr + row * NUM_EXPERTS + e)
            take = val > best_val
            best_val = tl.where(take, val, best_val)
            best_idx = tl.where(take, e, best_idx)
        tl.store(topIdx_ptr + row * TOP_K + t, best_idx)
        tl.store(topVals_ptr + row * TOP_K + t, best_val)


# Triton kernel: normalize and scale selected values -> topk_weight
@triton.jit
def _normalize_scale_kernel(
    topVals_ptr,        # [M, TOP_K] f32
    topkWeight_ptr,     # [M, TOP_K] f32
    M, TOP_K, routed_factor,
):
    row = tl.program_id(0)
    sum_val = tl.zeros((), dtype=tl.float32)
    for t in range(0, TOP_K):
        val = tl.load(topVals_ptr + row * TOP_K + t)
        sum_val += val
    eps = 1.0e-20
    for t in range(0, TOP_K):
        val = tl.load(topVals_ptr + row * TOP_K + t)
        norm = val / (sum_val + eps)
        scaled = norm * routed_factor
        tl.store(topkWeight_ptr + row * TOP_K + t, scaled)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure Triton and CUDA
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        # Contiguous and FP32
        hidden = hidden_states.contiguous().to(torch.float32)       # [M, K]
        weight = weight.contiguous().to(torch.float32)              # [NUM_EXPERTS, K]
        bias = expert_bias.contiguous().to(torch.float32)           # [NUM_EXPERTS]
        M = hidden.shape[0]
        K = hidden.shape[1]
        NUM_EXPERTS = weight.shape[0]
        assert NUM_EXPERTS == 256, "num_experts must be 256."
        assert K == 768, "hidden_dim K must be 768 to match the original model."

        # 1) Matmul for logits [M, NUM_EXPERTS]
        logits = torch.empty((M, NUM_EXPERTS), dtype=torch.float32, device=hidden.device)
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(NUM_EXPERTS, BLOCK_N))
        _matmul_rowwise_kernel[grid](
            hidden, weight, logits,
            M, K, NUM_EXPERTS,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid (Triton)
        scores = torch.empty((M, NUM_EXPERTS), dtype=torch.float32, device=hidden.device)
        _sigmoid_kernel[(M,)](logits, scores, M, NUM_EXPERTS)

        # 3) Add bias (Triton)
        scores_for_routing = torch.empty((M, NUM_EXPERTS), dtype=torch.float32, device=hidden.device)
        _add_bias_kernel[(M,)](scores, bias, scores_for_routing, M, NUM_EXPERTS)

        # 4) Group top-2 sum per group (Triton)
        group_scores = torch.empty((M, N_GROUP), dtype=torch.float32, device=hidden.device)
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, NUM_EXPERTS, N_GROUP, EXP_PER_GROUP,
        )

        # 5) Select top-4 groups per token (Triton)
        group_idx = torch.empty((M, TOPK_GROUP), dtype=torch.int32, device=hidden.device)
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, N_GROUP, TOPK_GROUP,
        )

        # 6) Build expert-level mask (Triton)
        score_mask = torch.empty((M, NUM_EXPERTS), dtype=torch.int32, device=hidden.device)
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, TOPK_GROUP, NUM_EXPERTS, EXP_PER_GROUP,
        )

        # 7) Masked fill: set non-selected to -inf (Triton)
        masked_scores = torch.empty((M, NUM_EXPERTS), dtype=torch.float32, device=hidden.device)
        _masked_fill_kernel[(M,)](
            scores_for_routing, score_mask, masked_scores,
            M, NUM_EXPERTS,
        )

        # 8) Final top-8 selection (Triton)
        top8_idx = torch.empty((M, TOP_K), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, TOP_K), dtype=torch.float32, device=hidden.device)
        _final_top8_selection_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, NUM_EXPERTS, TOP_K,
        )

        # 9) Normalize and scale (Triton)
        topk_weight = torch.empty((M, TOP_K), dtype=torch.float32, device=hidden.device)
        _normalize_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, TOP_K, routed_scaling_factor,
        )

        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
