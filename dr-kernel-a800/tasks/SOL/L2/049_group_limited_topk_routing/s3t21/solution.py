import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K] = hidden, float32, contiguous
    B_ptr,  # [K, N] = weight.T, float32, contiguous
    C_ptr,  # [M, N] = logits, float32, contiguous
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: one program per tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b)
        k += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _sigmoid_bias_kernel(
    X_ptr,   # [M, N] logits, float32, contiguous
    Bias_ptr, # [N] expert bias, float32, contiguous
    Y_ptr,   # [M, N] output (sigmoid + bias), float32, contiguous
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_b,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    x = tl.load(
        X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    )
    b = tl.load(Bias_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x)) + b
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _group_top2_sum_kernel(
    S_ptr,      # [M, 256] scores, float32, contiguous
    GS_ptr,     # [M, 8] group_scores, float32, contiguous
    GIdx_ptr,   # [M, 8] group indices, int32, contiguous
    M, E,       # E=256
    stride_sm, stride_sn,
    stride_gsm, stride_gsn,
    stride_gim, stride_gin,
    GROUPS: tl.constexpr,           # 8
    EXP_PER_GROUP: tl.constexpr,    # 32
):
    # Each program handles one token (row)
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    for group_id in range(GROUPS):
        base = group_id * EXP_PER_GROUP
        exp_vec = base + tl.arange(0, EXP_PER_GROUP)
        s = tl.load(
            S_ptr + pid_m * stride_sm + exp_vec * stride_sn,
            mask=exp_vec < E,
            other=-1e30,
        )

        neg_inf = -1e30
        best_val = neg_inf
        best_idx = -1
        second_val = neg_inf
        second_idx = -1

        for j in range(EXP_PER_GROUP):
            val = s[j]
            if val > best_val:
                second_val = best_val
                second_idx = best_idx
                best_val = val
                best_idx = j
            elif val > second_val:
                second_val = val
                second_idx = j

        group_score = best_val + second_val
        tl.store(GS_ptr + pid_m * stride_gsm + group_id * stride_gsn, group_score)
        tl.store(GIdx_ptr + pid_m * stride_gim + group_id * stride_gin, base + best_idx)


@triton.jit
def _select_top4_groups_kernel(
    GS_ptr,      # [M, 8] group_scores, float32, contiguous
    TopIdx_ptr,  # [M, 4] selected group indices, int32, contiguous
    M, N,        # N=8
    stride_gsm, stride_gsn,
    stride_tom, stride_ton,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Load group scores
    scores = tl.load(
        GS_ptr + pid_m * stride_gsm + tl.arange(0, N) * stride_gsn,
        mask=tl.arange(0, N) < N,
        other=-1e30,
    )
    # Initialize top-4 buffers
    top_vals = tl.full((4,), -1e30, tl.float32)
    top_idxs = tl.full((4,), -1, tl.int32)

    for i in range(N):
        val = scores[i]
        best_idx = i
        # Bubble up insertion into top_vals/top_idxs
        for j in range(3, -1, -1):
            cond = val > top_vals[j]
            # Swap down
            tmp_val = top_vals[j]
            tmp_idx = top_idxs[j]
            top_vals[j] = tl.where(cond, val, tmp_val)
            top_idxs[j] = tl.where(cond, best_idx, tmp_idx)
            val = tl.where(cond, tmp_val, val)
            best_idx = tl.where(cond, tmp_idx, best_idx)

    # Store selected indices
    for j in range(4):
        tl.store(TopIdx_ptr + pid_m * stride_tom + j * stride_ton, top_idxs[j])


@triton.jit
def _build_group_mask_kernel(
    TopIdx_ptr,  # [M, 4], int32
    Mask_ptr,    # [M, 8], float32
    M, N,        # N=8
    stride_tom, stride_ton,
    stride_mm, stride_mn,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Load selected group indices
    # Note: original logic keeps only top-4 groups; non-selected groups are 0
    for j in range(4):
        group_id = tl.load(TopIdx_ptr + pid_m * stride_tom + j * stride_ton)
        # We do not need to write them explicitly; we'll initialize mask to zeros and scatter
    # Initialize mask row to zeros and then scatter 1s for the 4 selected groups
    # We write 1s to groups 0..7 that match the 4 group_ids
    # Scatter 1s
    for j in range(4):
        group_id = tl.load(TopIdx_ptr + pid_m * stride_tom + j * stride_ton)
        tl.store(Mask_ptr + pid_m * stride_mm + group_id * stride_mn, 1.0)


@triton.jit
def _mask_scores_kernel(
    S_ptr,       # [M, 256] scores, float32, contiguous
    Mask_ptr,    # [M, 8] float32 mask (0 or 1)
    Msked_ptr,   # [M, 256] masked scores, float32, contiguous
    M, E,        # E=256
    stride_sm, stride_sn,
    stride_mm, stride_mn,
    stride_mmsep, stride_msen,
):
    # Each program handles one token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Load group mask and compute which groups are selected (mask > 0)
    selected_groups = tl.zeros((8,), dtype=tl.int32)
    for g in range(8):
        is_sel = tl.load(Mask_ptr + pid_m * stride_mm + g * stride_mn) > 0
        selected_groups[g] = tl.where(is_sel, g, -1)

    # For each expert, check if its group is in selected_groups
    for e in range(256):
        group_id = e // 32  # since 256=8*32, group = expert // 32
        # Check if group_id is in selected_groups (non-negative)
        found = 0
        for sg in range(8):
            if selected_groups[sg] == group_id:
                found = 1
                break
        val = tl.load(S_ptr + pid_m * stride_sm + e * stride_sn)
        masked_val = tl.where(found == 1, val, -1e30)
        tl.store(Msked_ptr + pid_m * stride_mmsep + e * stride_msen, masked_val)


@triton.jit
def _select_top8_final_kernel(
    Msked_ptr,   # [M, 256] masked scores, float32, contiguous
    TopKIdx_ptr, # [M, 8], int32, contiguous
    M, E,        # E=256
    stride_msm, stride_msn,
    stride_tkm, stride_tkn,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    top_vals = tl.full((8,), -1e30, tl.float32)
    top_idxs = tl.full((8,), -1, tl.int32)

    for i in range(E):
        val = tl.load(Msked_ptr + pid_m * stride_msm + i * stride_msn)
        best_idx = i
        # Insert into top-8 via bubble-up
        for j in range(7, -1, -1):
            cond = val > top_vals[j]
            tmp_val = top_vals[j]
            tmp_idx = top_idxs[j]
            top_vals[j] = tl.where(cond, val, tmp_val)
            top_idxs[j] = tl.where(cond, best_idx, tmp_idx)
            val = tl.where(cond, tmp_val, val)
            best_idx = tl.where(cond, tmp_idx, best_idx)

    for j in range(8):
        tl.store(TopKIdx_ptr + pid_m * stride_tkm + j * stride_tkn, top_idxs[j])


@triton.jit
def _normalize_scale_kernel(
    S_ptr,        # [M, 256] original scores (post sigmoid + bias), float32, contiguous
    TopKIdx_ptr,  # [M, 8], int32, contiguous
    OutW_ptr,     # [M, 8], float32, contiguous
    M, N,         # N=8
    stride_sm, stride_sn,
    stride_tkm, stride_tkn,
    eps: tl.constexpr,
    scaling: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Gather selected expert scores
    sum_ = 0.0
    for j in range(N):
        idx = tl.load(TopKIdx_ptr + pid_m * stride_tkm + j * stride_tkn)
        val = tl.load(S_ptr + pid_m * stride_sm + idx * stride_sn)
        sum_ += val

    # Compute normalized weights
    # We'll store one weight per selected expert j
    for j in range(N):
        idx = tl.load(TopKIdx_ptr + pid_m * stride_tkm + j * stride_tkn)
        val = tl.load(S_ptr + pid_m * stride_sm + idx * stride_sn)
        w = val / (sum_ + eps)
        w = w * scaling
        tl.store(OutW_ptr + pid_m * stride_tkm + j * stride_tkn, w)


def _run_triton_routing(hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
    # Ensure contiguity and dtype
    hidden = hidden_states.contiguous().to(torch.float32)  # [M, K], K default 768
    weight_t = weight.t().contiguous().to(torch.float32)   # [K, E], E=256
    bias = expert_bias.contiguous().to(torch.float32)      # [E]

    M = hidden.shape[0]
    K = hidden.shape[1]
    E = weight_t.shape[1]

    # 1) Linear projection: logits = hidden @ weight.T
    logits = torch.empty((M, E), device=hidden.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 64), triton.cdiv(E, 64))
    _matmul_kernel[grid](
        hidden, weight_t, logits,
        M, E, K,
        hidden.stride(0), hidden.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        logits.stride(0), logits.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    # 2) Sigmoid + expert bias
    scores = torch.empty((M, E), device=hidden.device, dtype=torch.float32)
    grid2 = (triton.cdiv(M, 128), triton.cdiv(E, 64))
    _sigmoid_bias_kernel[grid2](
        logits, bias, scores,
        M, E,
        logits.stride(0), logits.stride(1),
        scores.stride(0), scores.stride(1),
        bias.stride(0),
    )

    # 3) Compute group_scores [M, 8] and per-group indices [M, 8]
    group_scores = torch.empty((M, 8), device=hidden.device, dtype=torch.float32)
    group_idxs = torch.empty((M, 8), device=hidden.device, dtype=torch.int32)
    grid3 = (M,)
    _group_top2_sum_kernel[grid3](
        scores, group_scores, group_idxs,
        M, E,
        scores.stride(0), scores.stride(1),
        group_scores.stride(0), group_scores.stride(1),
        group_idxs.stride(0), group_idxs.stride(1),
        GROUPS=8, EXP_PER_GROUP=32,
    )

    # 4) Select top-4 groups per token: TopIdx [M, 4]
    top4_idx = torch.empty((M, 4), device=hidden.device, dtype=torch.int32)
    grid4 = (M,)
    _select_top4_groups_kernel[grid4](
        group_scores, top4_idx,
        M, 8,
        group_scores.stride(0), group_scores.stride(1),
        top4_idx.stride(0), top4_idx.stride(1),
    )

    # 5) Build group_mask [M, 8] (float32, 0/1)
    group_mask = torch.empty((M, 8), device=hidden.device, dtype=torch.float32)
    grid5 = (M,)
    _build_group_mask_kernel[grid5](
        top4_idx, group_mask,
        M, 8,
        top4_idx.stride(0), top4_idx.stride(1),
        group_mask.stride(0), group_mask.stride(1),
    )

    # 6) Mask scores: set non-selected group scores to -inf
    masked_scores = torch.empty((M, E), device=hidden.device, dtype=torch.float32)
    grid6 = (M,)
    _mask_scores_kernel[grid6](
        scores, group_mask, masked_scores,
        M, E,
        scores.stride(0), scores.stride(1),
        group_mask.stride(0), group_mask.stride(1),
        masked_scores.stride(0), masked_scores.stride(1),
    )

    # 7) Select final top-8 experts from masked scores
    top8_idx = torch.empty((M, 8), device=hidden.device, dtype=torch.int32)
    grid7 = (M,)
    _select_top8_final_kernel[grid7](
        masked_scores, top8_idx,
        M, E,
        masked_scores.stride(0), masked_scores.stride(1),
        top8_idx.stride(0), top8_idx.stride(1),
    )

    # 8) Gather selected expert scores from original 'scores' (post sigmoid + bias), normalize and scale
    out_weights = torch.empty((M, 8), device=hidden.device, dtype=torch.float32)
    eps = 1e-20
    grid8 = (M,)
    _normalize_scale_kernel[grid8](
        scores, top8_idx, out_weights,
        M, 8,
        scores.stride(0), scores.stride(1),
        top8_idx.stride(0), top8_idx.stride(1),
        eps=eps, scaling=routed_scaling_factor,
    )

    # Return indices and normalized weights
    # top8_idx contains per-token selected expert indices [0..255]
    # out_weights contains per-token normalized routing weights [8] after scaling
    return top8_idx, out_weights

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure on same device
        # If inputs are on CPU, Triton requires CUDA; move to CUDA if available.
        device = hidden_states.device
        if device.type != 'cuda':
            hidden_states = hidden_states.to('cuda')
            weight = weight.to('cuda')
            expert_bias = expert_bias.to('cuda')
        top8_idx, out_weights = _run_triton_routing(hidden_states, weight, expert_bias, routed_scaling_factor)
        # If original input was on CPU, you may return CPU tensors; here we keep CUDA outputs for Triton.
        return top8_idx, out_weights


def run(*args):
    return ModelNew()(*args)
