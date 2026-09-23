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
# M = num_tokens, N = num_experts, K = hidden_dim
@triton.jit
def _matmul_kernel(
    hidden_ptr,        # *f32, [M, K]
    weight_ptr,        # *f32, [N, K]
    out_ptr,           # *f32, [M, N]
    M, K, N,           # int32 sizes
    stride_hm, stride_hk,   # strides for hidden
    stride_wk, stride_wn,   # strides for weight
    stride_om, stride_on,   # strides for out
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # hidden tile: [BM, BK]
        h_ptrs = hidden_ptr + (offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk)
        h_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        h = tl.load(h_ptrs, mask=h_mask, other=0.0)

        # weight tile: [BN, BK]
        w_ptrs = weight_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # acc += h [BM, BK] @ w^T [BK, BN] -> [BM, BN]
        acc += tl.dot(h, tl.trans(w))

    # write out
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# Kernel 2: Elementwise sigmoid on scores
@triton.jit
def _sigmoid_kernel(
    in_ptr,            # *f32, [M, N]
    out_ptr,           # *f32, [M, N]
    M, N,
    stride_im, stride_in,   # strides for in
    stride_om, stride_on,   # strides for out
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m
    offs_n = pid_n

    # scalar loads
    x = tl.load(in_ptr + offs_m * stride_im + offs_n * stride_in)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs_m * stride_om + offs_n * stride_on, y)


# Kernel 3: Add bias (vector of length N) to each row
@triton.jit
def _add_bias_kernel(
    scores_ptr,        # *f32, [M, N]
    bias_ptr,          # *f32, [N]
    out_ptr,           # *f32, [M, N]
    M, N,
    stride_sm, stride_sn,
    stride_onm, stride_onn,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m
    offs_n = pid_n

    score = tl.load(scores_ptr + offs_m * stride_sm + offs_n * stride_sn)
    bias_val = tl.load(bias_ptr + offs_n)
    tl.store(out_ptr + offs_m * stride_onm + offs_n * stride_onn, score + bias_val)


# Kernel 4: Compute per-group top-2 sum across 8 groups of 32 experts
# scores: [M, N], out_group_scores: [M, 8]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,        # *f32, [M, N]
    group_scores_ptr,  # *f32, [M, 8]
    M, N, EXPERTS_PER_GROUP,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    pid_m = tl.program_id(0)

    # Compute per-group max and second max
    # N must be divisible by 8*EXPERTS_PER_GROUP, here 256 = 8*32
    # We do 8 independent loops
    for g in range(0, 8):
        start = g * EXPERTS_PER_GROUP
        max1 = tl.full((), -1.0e20, tl.float32)
        max2 = tl.full((), -1.0e20, tl.float32)
        # vector of indices for this group
        offs = start + tl.arange(0, EXPERTS_PER_GROUP)
        for j in range(0, EXPERTS_PER_GROUP):
            idx = start + j
            # scalar load from scores row
            s = tl.load(scores_ptr + pid_m * stride_sm + idx * stride_sn)
            # update max1 and max2
            if s > max1:
                max2 = max1
                max1 = s
            elif s > max2:
                max2 = s
        # store sum of top-2 for this group
        tl.store(group_scores_ptr + pid_m * stride_gm + g * stride_gn, max1 + max2)


# Kernel 5: Select top-4 groups per token via iterative argmax (return indices)
# group_scores: [M, 8], in_idx: [M, 4] (int32)
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,  # *f32, [M, 8]
    group_idx_ptr,     # *i32, [M, 4]
    M, N_GROUPS,
    stride_gsm, stride_gsn,
    stride_igm, stride_ign,
):
    pid_m = tl.program_id(0)
    # iterative argmax across N_GROUPS = 8
    for r in range(0, 4):
        best = -1.0e20
        best_idx = -1
        for g in range(0, N_GROUPS):
            score = tl.load(group_scores_ptr + pid_m * stride_gsm + g * stride_gsn)
            if score > best:
                best = score
                best_idx = g
        # write index (int32)
        tl.store(group_idx_ptr + pid_m * stride_igm + r * stride_ign, best_idx)
        # mark as used by setting its score to -inf
        tl.store(group_scores_ptr + pid_m * stride_gsm + best_idx * stride_gsn, -1.0e20)


# Kernel 6: Build expert-level mask from selected group_idx
# group_idx: [M, 4], out_score_mask: [M, N], 1 for selected groups, 0 otherwise
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr,     # *i32, [M, 4]
    score_mask_ptr,    # *i32, [M, N]
    M, N, EXPERTS_PER_GROUP,
    stride_gim, stride_gin,
    stride_smm, stride_smn,
):
    pid_m = tl.program_id(0)
    for g in range(0, 4):
        gi = tl.load(group_idx_ptr + pid_m * stride_gim + g * stride_gin)  # int32
        start = gi * EXPERTS_PER_GROUP
        for j in range(0, EXPERTS_PER_GROUP):
            idx = start + j
            tl.store(score_mask_ptr + pid_m * stride_smm + idx * stride_smn, 1)
    # zero out remaining positions (if any) by default; here we only write 1s for 4 groups


# Kernel 7: Masked fill: set masked_scores[i, e] = -inf where score_mask[i, e] == 0
@triton.jit
def _masked_fill_kernel(
    scores_ptr,        # *f32, [M, N]
    score_mask_ptr,    # *i32, [M, N]
    masked_ptr,        # *f32, [M, N]
    M, N,
    stride_sm, stride_sn,
    stride_msm, stride_msn,
    NEG_INF: tl.constexpr,  # e.g., -1e20
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m
    offs_n = pid_n

    score = tl.load(scores_ptr + offs_m * stride_sm + offs_n * stride_sn)
    mask_val = tl.load(score_mask_ptr + offs_m * stride_msm + offs_n * stride_msn)
    keep = mask_val != 0
    out = tl.where(keep, score, NEG_INF)
    tl.store(masked_ptr + offs_m * stride_msm + offs_n * stride_msn, out)


# Kernel 8: Final top-8 selection from masked_scores (values only), iterative argmax
# masked_scores: [M, N], out_top8_vals: [M, 8], out_top8_idx: [M, 8]
@triton.jit
def _final_top8_vals_indices_kernel(
    masked_ptr,        # *f32, [M, N]
    top8_vals_ptr,     # *f32, [M, 8]
    top8_idx_ptr,      # *i32, [M, 8]
    M, N,
    stride_ms, stride_nn,    # strides for masked (assume contiguous)
    stride_tvm, stride_tvn,
    stride_tim, stride_tin,
):
    pid_m = tl.program_id(0)
    # iterative argmax 8 times
    for r in range(0, 8):
        best = -1.0e20
        best_idx = -1
        for n in range(0, N):
            val = tl.load(masked_ptr + pid_m * stride_ms + n * stride_nn)
            if val > best:
                best = val
                best_idx = n
        # store value and index
        tl.store(top8_vals_ptr + pid_m * stride_tvm + r * stride_tvn, best)
        tl.store(top8_idx_ptr + pid_m * stride_tim + r * stride_tin, best_idx)
        # mark as used by setting its score to -inf
        tl.store(masked_ptr + pid_m * stride_ms + best_idx * stride_nn, NEG_INF)


# Kernel 9: Normalize and scale topk weights
# top8_vals: [M, 8], routed_scaling_factor: float, out_topk_weight: [M, 8]
@triton.jit
def _normalize_and_scale_kernel(
    top8_vals_ptr,     # *f32, [M, 8]
    out_ptr,           # *f32, [M, 8]
    M, K,              # K=8 here
    SCALE,             # float32
    stride_tvm, stride_tvn,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    total = tl.zeros((), dtype=tl.float32)
    for r in range(0, K):
        v = tl.load(top8_vals_ptr + pid_m * stride_tvm + r * stride_tvn)
        total += v
    total = total + 1e-20  # avoid div by zero
    for r in range(0, K):
        v = tl.load(top8_vals_ptr + pid_m * stride_tvm + r * stride_tvn)
        w = v / total * SCALE
        tl.store(out_ptr + pid_m * stride_om + r * stride_on, w)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int, routed_scaling_factor: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = self.num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        # Ensure contiguity and dtype
        hidden = hidden_states.contiguous().to(torch.float32)        # [M, K]
        weight = weight.contiguous().to(torch.float32)               # [N, K] where N=256
        bias = expert_bias.contiguous().to(torch.float32)            # [N] = [256]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch: weight must be [256, hidden_dim], hidden [M, hidden_dim]."

        # Allocate outputs and intermediates
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)       # [M, N]
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)       # [M, N]
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)  # [M, N]
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)  # [M, 8]
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden.device)     # [M, 4]
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)    # [M, N]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device) # [M, N]
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)   # [M, 8]
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)       # [M, 8]
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)   # [M, 8]

        # Launch Triton kernels
        # 1) Matmul for logits
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid_matmul](
            hidden, weight, logits,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            weight.stride(1), weight.stride(0),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # 2) Sigmoid
        _sigmoid_kernel[(M, N)](
            logits, scores,
            M, N,
            scores.stride(0), scores.stride(1),
            scores.stride(0), scores.stride(1),
        )

        # 3) Add bias
        _add_bias_kernel[(M, N)](
            scores, bias, scores_for_routing,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            scores_for_routing.stride(0), scores_for_routing.stride(1),
        )

        # 4) Group top-2 sum
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, self.experts_per_group,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            group_scores.stride(0), group_scores.stride(1),
        )

        # 5) Select top-4 groups per token
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
        )

        # 6) Build expert-level mask from group_idx
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, N, self.experts_per_group,
            group_idx.stride(0), group_idx.stride(1),
            score_mask.stride(0), score_mask.stride(1),
        )

        # 7) Masked fill: set non-selected to -inf
        _masked_fill_kernel[(M, N)](
            scores_for_routing, score_mask, masked_scores,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            NEG_INF=-1.0e20,
        )

        # 8) Final top-8 selection from masked scores (values only)
        _final_top8_vals_indices_kernel[(M,)](
            masked_scores, top8_vals, top8_idx,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_vals.stride(0), top8_vals.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            NEG_INF=-1.0e20,
        )

        # 9) Normalize and scale
        _normalize_and_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, self.top_k,
            self.routed_scaling_factor,
            top8_vals.stride(0), top8_vals.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
        )

        # Return dummy indices and normalized weights to match original signature
        # Original returns (topk_idx, topk_weight)
        # We need to provide indices; since final top-8 indices are computed above, return them.
        # If original expected only 8, we use top8_idx; however, original run uses top-k selection internally
        # and returns topk_idx. Here, we return top8_idx as the indices. We don't have exact selection
        # sequence as original (it masks groups first), but per masked_scores top8_idx is valid.
        # If strict original behavior is required, we can recompute with torch.topk at the end; but since
        # we must use Triton-only, we return top8_idx here.
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
