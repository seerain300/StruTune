import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [N, K] but we read as B[k, n] = weight[n, k] -> effectively [K, N]
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,   # strides for A: row-major [M, K]
    stride_bk, stride_bn,   # strides for B (we pass weight [N, K]): weight[n, k] with these strides
    stride_cm, stride_cn,   # strides for C: row-major [M, N]
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # 2D launch: pid_m along M tiles, pid_n along N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Loop over K in chunks of BK
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        # Load A tile: [BM, BK]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # Load B tile as [BK, BN]; B is [N, K] but we access B[k, n] = weight[n, k] using strides
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr,       # [M, N], float32
    bias_ptr,         # [N], float32
    out_ptr,          # [M, N], float32
    M, N,
    stride_lm, stride_ln,
    stride_bn,
    stride_om, stride_on,
):
    # 2D grid: one program per (row, col)
    t = tl.program_id(0)
    col = tl.program_id(1)
    if t >= M or col >= N:
        return
    x = tl.load(logits_ptr + t * stride_lm + col * stride_ln)
    b = tl.load(bias_ptr + col * stride_bn)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + t * stride_om + col * stride_on, y + b)


@triton.jit
def _top2_group_sum_kernel(
    scores_ptr,       # [M, N], float32
    group_scores_ptr, # [M, 8], float32
    M, N,
):
    # One program per token (row)
    t = tl.program_id(0)
    if t >= M:
        return
    # Compute top-2 sum for each group of 32 experts
    for g in range(8):
        start = g * 32
        local = tl.zeros((), dtype=tl.float32)
        # Scan 32 experts in the group
        for i in range(32):
            idx = start + i
            # Load scalar
            val = tl.load(scores_ptr + t * N + idx)
            # Track top-2 within this group
            maxv = -float('inf')
            second = -float('inf')
            for j in range(32):
                vj = tl.load(scores_ptr + t * N + start + j)
                if vj > maxv:
                    second = maxv
                    maxv = vj
                elif vj > second:
                    second = vj
            local += (maxv + second)
        tl.store(group_scores_ptr + t * 8 + g, local)


@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,  # [M, 8], float32
    top4_groups_ptr,   # [M, 4], int32
    M, N,
):
    # One program per token; select 4 maxima indices from 8 groups
    t = tl.program_id(0)
    if t >= M:
        return
    for r in range(4):
        maxv = -float('inf')
        max_idx = -1
        for g in range(8):
            v = tl.load(group_scores_ptr + t * 8 + g)
            cond = v > maxv
            max_idx = tl.where(cond, g, max_idx)
            maxv = tl.where(cond, v, maxv)
        tl.store(top4_groups_ptr + t * 4 + r, max_idx)


@triton.jit
def _mask_nonselected_groups_kernel(
    logits_ptr,         # [M, N], float32 (original logits before sigmoid)
    top4_groups_ptr,    # [M, 4], int32
    masked_ptr,         # [M, N], float32
    M, N,
):
    # For each token, set columns of non-selected groups to -inf; selected groups remain original
    t = tl.program_id(0)
    if t >= M:
        return
    neg_inf = -float('inf')
    for col in range(N):
        # Initialize to -inf
        tl.store(masked_ptr + t * N + col, neg_inf)
    # Overwrite selected groups with original logits
    for r in range(4):
        g = tl.load(top4_groups_ptr + t * 4 + r)
        start = g * 32
        end = start + 32
        for j in range(32):
            idx = start + j
            val = tl.load(logits_ptr + t * N + idx)
            tl.store(masked_ptr + t * N + idx, val)


@triton.jit
def _select_top8_per_token_kernel(
    masked_logits_ptr,  # [M, N], float32
    top8_idx_ptr,       # [M, 8], int32
    M, N,
):
    # One program per token; perform 8 argmax scans to select top-8
    t = tl.program_id(0)
    if t >= M:
        return
    for r in range(8):
        maxv = -float('inf')
        max_idx = -1
        for col in range(N):
            v = tl.load(masked_logits_ptr + t * N + col)
            cond = v > maxv
            max_idx = tl.where(cond, col, max_idx)
            maxv = tl.where(cond, v, maxv)
        tl.store(top8_idx_ptr + t * 8 + r, max_idx)
        # Mark selected column by setting to -inf
        tl.store(masked_logits_ptr + t * N + max_idx, -float('inf'))


@triton.jit
def _normalize_and_scale_kernel(
    masked_logits_ptr,   # [M, N], float32
    top8_idx_ptr,        # [M, 8], int32
    out_ptr,             # [M, 8], float32 (final topk_weight)
    M, N,
    routed_scale,        # float32
    eps,                 # float32 (1e-20)
):
    # For each token, load top8 indices, gather masked_logits values, normalize, and scale
    t = tl.program_id(0)
    if t >= M:
        return
    sum_selected = tl.zeros((), dtype=tl.float32)
    for r in range(8):
        idx = tl.load(top8_idx_ptr + t * 8 + r)  # int32
        val = tl.load(masked_logits_ptr + t * N + idx)
        sum_selected += val
    for r in range(8):
        idx = tl.load(top8_idx_ptr + t * 8 + r)
        val = tl.load(masked_logits_ptr + t * N + idx)
        norm = val / (sum_selected + eps)
        scaled = norm * routed_scale
        tl.store(out_ptr + t * 8 + r, scaled)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype float32 for computation
        hidden = hidden_states.to(torch.float32).contiguous()  # [M, K]
        weight_t = weight.to(torch.float32).contiguous()       # [N, K]
        M, K = hidden.shape
        N, K_w = weight_t.shape
        assert K == K_w, "hidden_dim mismatch"
        assert N == 256, "num_experts must be 256"
        device = hidden.device

        # 1) Triton matmul: logits = hidden @ weight.T → [M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        grid_m = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        _matmul_kernel[grid_m](
            hidden, weight_t, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            logits.stride(0), logits.stride(1),
            BM=128, BN=128, BK=64,
            num_warps=4, num_stages=3,
        )

        # 2) Triton: sigmoid + expert bias → scores [M, N]
        scores = torch.empty_like(logits)
        grid_sigmoid = (M, N)
        _sigmoid_add_bias_kernel[grid_sigmoid](
            logits, expert_bias.to(torch.float32), scores,
            M, N,
            logits.stride(0), logits.stride(1),
            expert_bias.stride(0),
            scores.stride(0), scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 3) Triton: compute group top-2 sums per token → [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        _top2_group_sum_kernel[(M,)](scores, group_scores, M, N)

        # 4) Triton: select top-4 groups per token → [M, 4] int32
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        _select_top4_groups_kernel[(M,)](group_scores, top4_groups, M, N)

        # 5) Triton: mask non-selected groups in original logits → [M, N]
        masked_logits = torch.empty((M, N), dtype=torch.float32, device=device)
        _mask_nonselected_groups_kernel[(M,)](logits, top4_groups, masked_logits, M, N)

        # 6) Triton: select top-8 indices per token from masked_logits → [M, 8] int32
        top8_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        _select_top8_per_token_kernel[(M,)](masked_logits, top8_idx, M, N)

        # 7) Triton: normalize and scale to produce topk_weight [M, 8] float32
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        _normalize_and_scale_kernel[(M,)](masked_logits, top8_idx, topk_weight, M, N, routed_scaling_factor, 1e-20)

        # 8) Return topk_idx (int64) and topk_weight (float32)
        topk_idx = top8_idx.to(torch.int64)

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
