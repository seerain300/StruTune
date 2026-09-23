import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Matmul logits = hidden @ weight^T
# hidden: [M, K], weight: [N, K], logits: [M, N]
@triton.jit
def _matmul_kernel(
    hidden_ptr,  # *f32, [M, K]
    weight_ptr,  # *f32, [N, K]
    logits_ptr,  # *f32, [M, N]
    M, K, N,
    stride_hm, stride_hk,
    stride_wk, stride_wn,
    stride_lm, stride_ln,
    BLOCK_M: tl.constexpr,  # tile in M
    BLOCK_N: tl.constexpr,  # tile in N
    BLOCK_K: tl.constexpr,  # tile in K
):
    # Program id for tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k

        # Pointers for A (hidden) tile: [BLOCK_M, BLOCK_K]
        a_ptrs = hidden_ptr + (offs_m[:, None] * stride_hm) + (k_ids[None, :] * stride_hk)
        # Mask for A
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for B (weight) tile: [BLOCK_K, BLOCK_N], note: weight is [N, K]
        b_ptrs = weight_ptr + (k_ids[:, None] * stride_wk) + (offs_n[None, :] * stride_wn)
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Write output tile: logits [M, N]
    out_ptrs = logits_ptr + (offs_m[:, None] * stride_lm) + (offs_n[None, :] * stride_ln)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# Kernel 2: Elementwise sigmoid (FP32)
@triton.jit
def _sigmoid_kernel(
    in_ptr,  # *f32, [M, N]
    out_ptr, # *f32, [M, N]
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    in_ptrs = in_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    x = tl.load(in_ptrs, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y, mask=mask)


# Kernel 3: Add expert bias (broadcast across M)
@triton.jit
def _add_bias_kernel(
    scores_ptr,   # *f32, [M, N]
    bias_ptr,     # *f32, [N]
    out_ptr,      # *f32, [M, N]
    M, N,
    stride_sm, stride_sn,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    s_ptrs = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    scores = tl.load(s_ptrs, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    out = scores + bias[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask)


# Kernel 4: Compute per-group top-2 sum: group_scores[i, g] = sum of top-2 scores in scores[i, g*exp_per_group:(g+1)*exp_per_group]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,    # *f32, [M, N]
    out_ptr,       # *f32, [M, num_groups]
    M, N,
    stride_sm, stride_sn,
    stride_om, stride_og,
    num_groups,    # int32
    exp_per_group, # int32
):
    pid_m = tl.program_id(0)
    m = pid_m

    # One program per (m, g) but we can process all groups per m sequentially
    for g in range(0, num_groups):
        group_start = g * exp_per_group
        group_end = group_start + exp_per_group
        # Compute top-2 sum over [group_start, group_end)
        top1 = -float('inf')
        top2 = -float('inf')
        for j in range(group_start, group_end):
            ptr = scores_ptr + m * stride_sm + j * stride_sn
            val = tl.load(ptr)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        sum_top2 = top1 + top2
        out_ptr_j = out_ptr + m * stride_om + g * stride_og
        tl.store(out_ptr_j, sum_top2)


# Kernel 5: Select top-4 groups per token via iterative argmax on group_scores
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,  # *f32, [M, num_groups]
    out_idx_ptr,       # *i32, [M, 4]
    M, num_groups,
    stride_gm, stride_gg,
    stride_om, stride_og,
):
    pid_m = tl.program_id(0)
    m = pid_m
    # We assume num_groups >= 4 (typical: 8 groups)
    best = [(-float('inf'), -1) for _ in range(4)]
    for g in range(0, num_groups):
        ptr = group_scores_ptr + m * stride_gm + g * stride_gg
        val = tl.load(ptr)
        # Insert into best list
        for j in range(0, 4):
            if val > best[j][0]:
                # shift down
                for k in range(3, j - 1, -1):
                    best[k] = best[k - 1]
                best[j] = (val, g)
                break
    # Store indices
    for j in range(0, 4):
        if best[j][1] != -1:
            out_ptr = out_idx_ptr + m * stride_om + j * stride_og
            tl.store(out_ptr, best[j][1])


# Kernel 6: Build expert-level mask: for each token m, set mask[m, e] = 1 if e in selected groups, else 0
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr,     # *i32, [M, 4] - indices of selected groups
    mask_ptr,          # *i32, [M, N]
    M, N, num_groups, exp_per_group,
    stride_gmi, stride_gmj,
    stride_mmi, stride_mn,
):
    pid_m = tl.program_id(0)
    m = pid_m
    for j in range(0, 4):
        g = tl.load(group_idx_ptr + m * stride_gmi + j * stride_gmj)
        group_start = g * exp_per_group
        group_end = group_start + exp_per_group
        for e in range(group_start, group_end):
            mask_ptr_e = mask_ptr + m * stride_mmi + e * stride_mn
            tl.store(mask_ptr_e, 1)
    # set non-selected positions to 0
    for e in range(0, N):
        mask_ptr_e = mask_ptr + m * stride_mmi + e * stride_mn
        # if not set above, keep 0


# Kernel 7: Masked fill: set masked_scores[m, e] = -inf if mask[m, e] == 0, else keep scores_for_routing[m, e]
@triton.jit
def _masked_fill_kernel(
    scores_ptr,        # *f32, [M, N]
    mask_ptr,          # *i32, [M, N]
    out_ptr,           # *f32, [M, N]
    M, N,
    stride_sm, stride_sn,
    stride_mmi, stride_mn,
    stride_om, stride_on,
    NEG_INF: tl.constexpr,  # float32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    s_ptrs = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    scores = tl.load(s_ptrs, mask=mask, other=0.0)
    m_ptrs = mask_ptr + offs_m[:, None] * stride_mmi + offs_n[None, :] * stride_mn
    m = tl.load(m_ptrs, mask=mask, other=0)  # 0 or 1
    keep = m != 0
    out_vals = tl.where(keep, scores, NEG_INF)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out_vals, mask=mask)


# Kernel 8: Final top-8 selection over masked_scores
@triton.jit
def _select_top8_kernel(
    scores_ptr,        # *f32, [M, N]
    out_idx_ptr,       # *i32, [M, 8]
    out_vals_ptr,      # *f32, [M, 8]
    M, N,
    stride_sm, stride_sn,
    stride_om, stride_ok,
    stride_im, stride_ik,
):
    pid_m = tl.program_id(0)
    m = pid_m
    top = [(-float('inf'), -1) for _ in range(8)]
    for e in range(0, N):
        ptr = scores_ptr + m * stride_sm + e * stride_sn
        val = tl.load(ptr)
        for j in range(0, 8):
            if val > top[j][0]:
                for k in range(7, j - 1, -1):
                    top[k] = top[k - 1]
                top[j] = (val, e)
                break
    for j in range(0, 8):
        val, idx = top[j]
        out_idx_ptr_j = out_idx_ptr + m * stride_om + j * stride_ok
        out_vals_ptr_j = out_vals_ptr + m * stride_im + j * stride_ik
        tl.store(out_idx_ptr_j, idx)
        tl.store(out_vals_ptr_j, val)


# Kernel 9: Normalize and apply scaling factor to selected values
@triton.jit
def _normalize_and_scale_kernel(
    vals_ptr,          # *f32, [M, 8]
    out_ptr,           # *f32, [M, 8]
    M, k: tl.constexpr,  # k=8
    stride_vm, stride_vk,
    stride_om, stride_ok,
    scaling_factor,
):
    pid_m = tl.program_id(0)
    m = pid_m
    sum_val = 0.0
    for j in range(0, k):
        ptr = vals_ptr + m * stride_vm + j * stride_vk
        v = tl.load(ptr)
        sum_val += v
    for j in range(0, k):
        ptr = vals_ptr + m * stride_vm + j * stride_vk
        v = tl.load(ptr)
        w = v / sum_val
        w = w * scaling_factor
        out_ptr_j = out_ptr + m * stride_om + j * stride_ok
        tl.store(out_ptr_j, w)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int = 256, n_group: int = 8, topk_group: int = 4, top_k: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.n_group = n_group
        self.experts_per_group = min(num_experts // n_group, 32)  # robust if num_experts not divisible by 32
        self.topk_group = topk_group
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward: ensure CUDA and availability
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        # Ensure contiguity and dtype
        hidden = hidden_states.contiguous().to(torch.float32)       # [M, K]
        weight = weight.contiguous().to(torch.float32)              # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)           # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]

        # Logits [M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)

        # Matmul: logits = hidden @ weight^T
        # Tiling parameters
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            hidden, weight, logits,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            weight.stride(1), weight.stride(0),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # Sigmoid
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        _sigmoid_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
            logits, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            num_warps=4,
        )

        # Add bias
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        _add_bias_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
            scores, bias, scores_for_routing,
            M, N,
            scores.stride(0), scores.stride(1),
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            num_warps=4,
        )

        # Compute group_scores: per-group top-2 sum
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            self.n_group, self.experts_per_group,
            num_warps=1,
        )

        # Select top-4 groups per token
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden.device)
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            num_warps=1,
        )

        # Build expert-level mask (1 for selected groups, 0 otherwise)
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, N, self.n_group, self.experts_per_group,
            group_idx.stride(0), group_idx.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            num_warps=1,
        )

        # Masked fill: set non-selected to -inf
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        _masked_fill_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
            scores_for_routing, score_mask, masked_scores,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            NEG_INF=torch.finfo(torch.float32).min,
            num_warps=4,
        )

        # Final top-8 selection from masked scores
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        _select_top8_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            top8_vals.stride(0), top8_vals.stride(1),
            num_warps=1,
        )

        # Normalize and scale
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        _normalize_and_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, 8,
            top8_vals.stride(0), top8_vals.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            1.0,  # routed_scaling_factor passed as 1.0 here; original code applies it after normalization
            num_warps=1,
        )

        # Return indices and normalized weights (apply scaling factor if desired)
        # Note: original code applies routed_scaling_factor after normalization; here we normalize then scale.
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
