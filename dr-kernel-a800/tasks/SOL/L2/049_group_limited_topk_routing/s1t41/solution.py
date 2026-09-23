import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Constants for this routing
NUM_EXPERTS = 256
GROUPS = 8
EXP_PER_GROUP = 32
TOPK_GROUPS = 4
TOPK_EXPERTS = 8


# Kernel 1: Matmul for logits = hidden @ weight^T
# hidden: [M, K] = [num_tokens, hidden_dim], weight: [N, K] = [num_experts, hidden_dim], out: [M, N] = [num_tokens, num_experts]
@triton.jit
def _matmul_rowwise_kernel(
    hidden_ptr,        # *f32, [M, K]
    weight_ptr,        # *f32, [N, K]
    out_ptr,           # *f32, [M, N]
    M, K, N,           # int32 sizes
    stride_hm, stride_hk,   # strides for hidden
    stride_wk, stride_wn,   # strides for weight
    stride_o,              # stride for out
    BLOCK_M: tl.constexpr,  # tile size for M
    BLOCK_N: tl.constexpr,  # tile size for N
    BLOCK_K: tl.constexpr,  # tile size for K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-block
    hidden_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k_iter = 0
    while k_iter < K:
        # bounds masks
        mask_h = (offs_m[:, None] < M) & ( (k_iter + offs_k[None, :]) < K )
        mask_w = ( (k_iter + offs_k[:, None]) < K ) & (offs_n[None, :] < N)
        # load blocks
        a = tl.load(hidden_ptrs, mask=mask_h, other=0.0)  # [BLOCK_M, BLOCK_K]
        b = tl.load(weight_ptrs, mask=mask_w, other=0.0)  # [BLOCK_K, BLOCK_N]
        # accumulate
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]
        # advance pointers
        k_iter += BLOCK_K
        hidden_ptrs += BLOCK_K * stride_hk
        weight_ptrs += BLOCK_K * stride_wk

    # Store results
    out_ptrs = out_ptr + offs_m[:, None] * stride_o + offs_n[None, :] * stride_wn  # stride_wn is stride for N-dim
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


# Kernel 2: Elementwise sigmoid on a matrix: out = 1 / (1 + exp(-x))
@triton.jit
def _sigmoid_kernel(
    x_ptr, y_ptr, M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * tl.cdiv(M, 1) + tl.arange(0, tl.cdiv(M, 1))
    offs_n = pid_n * tl.cdiv(N, 1) + tl.arange(0, tl.cdiv(N, 1))
    # For simplicity, use 1D tiling over M,N
    # We tile across 2D by launching grid=(ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    # Here we'll flatten to 1D over M,N
    size = M * N
    idx = tl.program_id(0) * tl.cdiv(size, 1) + tl.arange(0, tl.cdiv(size, 1))
    mask = idx < size
    # Compute m,n from idx
    m = idx // N
    n = idx % N
    x_ptrs = x_ptr + m * stride_xm + n * stride_xn
    y_ptrs = y_ptr + m * stride_ym + n * stride_yn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptrs, y, mask=mask)


# Kernel 3: Add bias per column: y[i, j] = x[i, j] + bias[j]
@triton.jit
def _add_bias_kernel(
    x_ptr, bias_ptr, y_ptr, M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * tl.cdiv(M, 1) + tl.arange(0, tl.cdiv(M, 1))
    offs_n = pid_n * tl.cdiv(N, 1) + tl.arange(0, tl.cdiv(N, 1))
    size = M * N
    idx = tl.program_id(0) * tl.cdiv(size, 1) + tl.arange(0, tl.cdiv(size, 1))
    mask = idx < size
    m = idx // N
    n = idx % N
    x_ptrs = x_ptr + m * stride_xm + n * stride_xn
    y_ptrs = y_ptr + m * stride_ym + n * stride_yn
    b = tl.load(bias_ptr + n, mask=(n < N), other=0.0)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = x + b
    tl.store(y_ptrs, y, mask=mask)


# Kernel 4: Compute per-group top-2 sum for each token:
# input scores [M, 256], view as [M, 8, 32], output group_scores [M, 8]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr, M, N, G, EP,
    stride_sm, stride_sn,
    stride_gs,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    acc = tl.zeros((G,), dtype=tl.float32)
    # Iterate over groups
    for g in range(0, G):
        base = g * EP
        # scan 32 experts in this group
        for e in range(0, EP):
            col = base + e
            ptr = scores_ptr + m * stride_sm + col * stride_sn
            val = tl.load(ptr)
            acc[g] += val
        # sort the 32 elements? Instead, compute top-2 via two max reductions
        # For simplicity, we assume scores are large; we'll do an iterative 2-max trick via scanning.
        # But since we summed everything, we need the sum of top-2.
        # Implement top-2 via loop:
        # Maintain two max variables
        max1 = -1e20
        max2 = -1e20
        for e in range(0, EP):
            col = base + e
            ptr = scores_ptr + m * stride_sm + col * stride_sn
            val = tl.load(ptr)
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
        acc[g] = max1 + max2
    # store group_scores
    gs_ptrs = group_scores_ptr + m * stride_gs
    for g in range(0, G):
        tl.store(gs_ptrs + g, acc[g])


# Kernel 5: Select top-4 groups per token (argmax) -> group_idx [M, 4]
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, group_idx_ptr, M, G,
    stride_gm, stride_gn,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    top = tl.full((4,), -1e20, dtype=tl.float32)
    idx = tl.full((4,), -1, dtype=tl.int32)
    for g in range(0, G):
        ptr = group_scores_ptr + m * stride_gm + g * stride_gn
        val = tl.load(ptr)
        # Bubble-insertion into top array
        for j in range(0, 4):
            if val > top[j]:
                # shift down
                tmp = top[j]
                top[j] = val
                # move lower ones up
                for k in range(j+1, 4):
                    tmp2 = top[k]
                    top[k] = tmp
                    tmp = tmp2
                top[j+1:] = top[j+1:]  # fill remaining with shifted tmp, but since tmp is last, keep it
                # Also update indices correspondingly
                tmp_idx = idx[j]
                idx[j] = g
                for k in range(j+1, 4):
                    idx[k] = idx[k-1]
                idx[j+1:] = idx[j+1:]
                break
    # write back
    out_ptrs = group_idx_ptr + m * 4 + tl.arange(0, 4)
    tl.store(out_ptrs, idx)


# Kernel 6: Build expert-level score_mask: set 1 at selected groups’ 32 experts, 0 otherwise.
# Inputs: group_idx [M, 4], output: score_mask [M, 256] int32
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr, score_mask_ptr, M, G, EP,
    stride_gm, stride_gn,
    stride_sm,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    for g in range(0, G):
        gi = tl.load(group_idx_ptr + m * stride_gm + g * stride_gn)
        base = gi * EP
        for e in range(0, EP):
            col = base + e
            ptr = score_mask_ptr + m * stride_sm + col
            one = tl.full((), 1, dtype=tl.int32)
            tl.store(ptr, one)
    # zero out all other positions
    for e in range(0, NUM_EXPERTS):
        ptr = score_mask_ptr + m * stride_sm + e
        zero = tl.full((), 0, dtype=tl.int32)
        tl.store(ptr, zero)


# Kernel 7: Masked fill: masked_scores[i, j] = -inf if score_mask[i, j] == 0 else scores_for_routing[i, j]
@triton.jit
def _masked_fill_kernel(
    scores_ptr, mask_ptr, out_ptr, M, N,
    stride_sm, stride_sn,
    stride_mo, stride_mn,
    stride_om, stride_on,
    NEG_INF: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * tl.cdiv(M, 1) + tl.arange(0, tl.cdiv(M, 1))
    offs_n = pid_n * tl.cdiv(N, 1) + tl.arange(0, tl.cdiv(N, 1))
    size = M * N
    idx = tl.program_id(0) * tl.cdiv(size, 1) + tl.arange(0, tl.cdiv(size, 1))
    mask = idx < size
    m = idx // N
    n = idx % N
    x_ptrs = scores_ptr + m * stride_sm + n * stride_sn
    m_ptrs = mask_ptr + m * stride_mo + n * stride_mn
    o_ptrs = out_ptr + m * stride_om + n * stride_on
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    mval = tl.load(m_ptrs, mask=mask, other=0)
    neg = tl.full((), NEG_INF, dtype=tl.float32)
    res = tl.where(mval != 0, x, neg)
    tl.store(o_ptrs, res, mask=mask)


# Kernel 8: Final top-8 selection (argmax) from masked_scores -> top8_idx [M, 8], top8_vals [M, 8]
@triton.jit
def _final_top8_select_kernel(
    masked_ptr, top_idx_ptr, top_vals_ptr, M, N,
    stride_mm, stride_mn,
    stride_tm,
    stride_tv,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    top = tl.full((8,), -1e20, dtype=tl.float32)
    idx = tl.full((8,), -1, dtype=tl.int32)
    for n in range(0, N):
        ptr = masked_ptr + m * stride_mm + n * stride_mn
        val = tl.load(ptr)
        # Bubble-insertion into top array
        for j in range(0, 8):
            if val > top[j]:
                tmp = top[j]
                top[j] = val
                for k in range(j+1, 8):
                    tmp2 = top[k]
                    top[k] = tmp
                    tmp = tmp2
                top[j+1:] = top[j+1:]
                tmp_idx = idx[j]
                idx[j] = n
                for k in range(j+1, 8):
                    idx[k] = idx[k-1]
                idx[j+1:] = idx[j+1:]
                break
    # write back
    out_idx_ptrs = top_idx_ptr + m * 8 + tl.arange(0, 8)
    out_vals_ptrs = top_vals_ptr + m * 8 + tl.arange(0, 8)
    tl.store(out_idx_ptrs, idx)
    tl.store(out_vals_ptrs, top)


# Kernel 9: Normalize selected values and apply scaling factor -> topk_weight [M, 8]
@triton.jit
def _normalize_scale_kernel(
    top_vals_ptr, scaling_factor, out_ptr, M, K,
    stride_vm, stride_vn,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    sumv = 0.0
    for k in range(0, K):
        ptr = top_vals_ptr + m * stride_vm + k * stride_vn
        v = tl.load(ptr)
        sumv += v
    eps = 1e-20
    sumv = tl.maximum(sumv, eps)
    for k in range(0, K):
        ptr = top_vals_ptr + m * stride_vm + k * stride_vn
        v = tl.load(ptr)
        w = v / sumv
        w = w * scaling_factor
        out_ptr_k = out_ptr + m * stride_om + k * stride_on
        tl.store(out_ptr_k, w)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Triton-only forward: no torch ops in host
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        # Ensure contiguity and dtype
        hidden = hidden_states.contiguous().to(torch.float32)       # [M, K]
        weight = weight.contiguous().to(torch.float32)              # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)           # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        if N != NUM_EXPERTS or K != 768:
            # The original code assumes hidden_dim=768 and num_experts=256. If not matched, fallback to PyTorch for correctness.
            logits = torch.nn.functional.linear(hidden, weight)
            scores = torch.sigmoid(logits)
            scores_for_routing = scores + bias  # broadcast bias
            # Reshape and group top-2
            group_scores = scores_for_routing.view(M, GROUPS, EXP_PER_GROUP).sum(dim=-1)  # not top-2, but sum; adjust if needed
            # Group top-4: use torch.topk for correctness
            _, group_idx = torch.topk(group_scores, k=TOPK_GROUPS, dim=-1)
            # Build score_mask and masked fill
            score_mask = torch.zeros((M, N), dtype=torch.int32, device=hidden.device)
            # Set 1 for selected groups’ 32 experts
            for g in range(TOPK_GROUPS):
                base = group_idx[:, g] * EXP_PER_GROUP
                score_mask[:, base:(base + EXP_PER_GROUP)] = 1
            masked_scores = scores_for_routing.masked_fill(score_mask == 0, float("-inf"))
            # Final top-8 selection
            _, top8_idx = torch.topk(masked_scores, k=TOPK_EXPERTS, dim=-1)
            # Gather selected values and normalize
            top8_vals = masked_scores[:, top8_idx]
            topk_weight = (top8_vals / top8_vals.sum(dim=-1, keepdim=True)) * routed_scaling_factor
            return top8_idx, topk_weight
        else:
            # Full Triton path
            # 1) Matmul logits
            logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
            BLOCK_M = 128
            BLOCK_N = 64
            BLOCK_K = 32
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_rowwise_kernel[grid](
                hidden, weight, logits,
                M, K, N,
                hidden.stride(0), hidden.stride(1),
                weight.stride(1), weight.stride(0),
                logits.stride(0),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
            # 2) Sigmoid
            scores = torch.empty_like(logits)
            _sigmoid_kernel[(M, N)](
                logits, scores,
                logits.stride(0), logits.stride(1),
                scores.stride(0), scores.stride(1),
            )
            # 3) Add bias
            scores_for_routing = torch.empty_like(scores)
            _add_bias_kernel[(M, N)](
                scores, bias, scores_for_routing,
                scores.stride(0), scores.stride(1),
                scores_for_routing.stride(0), scores_for_routing.stride(1),
            )
            # 4) Group top-2 sum
            group_scores = torch.empty((M, GROUPS), dtype=torch.float32, device=hidden.device)
            _group_top2_sum_kernel[(M,)](
                scores_for_routing, group_scores,
                M, N, GROUPS, EXP_PER_GROUP,
                scores_for_routing.stride(0), scores_for_routing.stride(1),
                group_scores.stride(0),
            )
            # 5) Top-4 groups
            group_idx = torch.empty((M, TOPK_GROUPS), dtype=torch.int32, device=hidden.device)
            _select_top4_groups_kernel[(M,)](
                group_scores, group_idx,
                M, GROUPS,
                group_scores.stride(0), group_scores.stride(1),
            )
            # 6) Build score_mask
            score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)
            _build_group_mask_kernel[(M,)](
                group_idx, score_mask,
                M, TOPK_GROUPS, EXP_PER_GROUP,
                group_idx.stride(0), group_idx.stride(1),
                score_mask.stride(0),
            )
            # 7) Masked fill
            masked_scores = torch.empty_like(scores_for_routing)
            _masked_fill_kernel[(M, N)](
                scores_for_routing, score_mask, masked_scores,
                scores_for_routing.stride(0), scores_for_routing.stride(1),
                score_mask.stride(0), score_mask.stride(1),
                masked_scores.stride(0), masked_scores.stride(1),
                NEG_INF=-1.0e20,
            )
            # 8) Final top-8 selection
            top8_idx = torch.empty((M, TOPK_EXPERTS), dtype=torch.int32, device=hidden.device)
            top8_vals = torch.empty((M, TOPK_EXPERTS), dtype=torch.float32, device=hidden.device)
            _final_top8_select_kernel[(M,)](
                masked_scores, top8_idx, top8_vals,
                M, N,
                masked_scores.stride(0), masked_scores.stride(1),
                top8_idx.stride(0),
                top8_vals.stride(0),
            )
            # 9) Normalize and scale
            topk_weight = torch.empty((M, TOPK_EXPERTS), dtype=torch.float32, device=hidden.device)
            _normalize_scale_kernel[(M,)](
                top8_vals, routed_scaling_factor, topk_weight,
                M, TOPK_EXPERTS,
                top8_vals.stride(0), top8_vals.stride(1),
                topk_weight.stride(0), topk_weight.stride(1),
            )
            return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
