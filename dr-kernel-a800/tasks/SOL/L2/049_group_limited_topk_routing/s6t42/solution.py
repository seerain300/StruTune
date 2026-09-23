import torch
import triton
import triton.language as tl


@triton.jit
def matmul_logits_kernel(
    a_ptr,             # *f32, [M, K], a = hidden_states
    b_ptr,             # *f32, [N, K], b = weight
    out_ptr,           # *f32, [M, N], logits
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32,
    stride_ak: tl.int32,
    stride_bn: tl.int32,
    stride_bk: tl.int32,
    stride_outm: tl.int32,
    stride_outn: tl.int32,
):
    # 2D tiling over M and N
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_block = tl.load(
            a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b_block = tl.load(
            b_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk),
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a_block, b_block)

    out_ptrs = out_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,        # *f32, [M, N]
    bias_ptr,          # *f32, [N]
    scores_ptr,        # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    stride_lm: tl.int32,
    stride_ln: tl.int32,
    stride_sm: tl.int32,
    stride_sn: tl.int32,
):
    # elementwise: scores = sigmoid(logits) + bias
    BLOCK_M = 128
    BLOCK_N = 128

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    logits = tl.load(logits_ptr + (offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln), mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
    scores = 1.0 / (1.0 + tl.exp(-logits)) + bias[None, :]
    tl.store(scores_ptr + (offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn), scores, mask=mask)


@triton.jit
def group_top2_sum_kernel(
    scores_ptr,        # *f32, [M, 8, 32] flattened pointer
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    group_count: tl.constexpr,   # 8
    experts_per_group: tl.constexpr,  # 32
):
    # Each program handles one token m
    pid = tl.program_id(0)
    if pid >= M:
        return

    base = pid * group_count * experts_per_group
    top1 = -float('inf')
    top2 = -float('inf')

    # Loop over 32 experts in the group
    for j in range(experts_per_group):
        val = tl.load(scores_ptr + base + j)
        if val > top1:
            top2 = top1
            top1 = val
        elif val > top2:
            top2 = val

    total = top1 + top2
    tl.store(group_scores_ptr + pid * group_count, total)


@triton.jit
def topk_group_kernel(
    group_scores_ptr,  # *f32, [M, 8]
    group_idx_ptr,     # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,   # 4
):
    # Iterative selection to find top-K (K=4) group indices per token
    pid = tl.program_id(0)
    if pid >= M:
        return

    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)

    for g in range(8):
        val = tl.load(group_scores_ptr + pid * 8 + g)
        for j in range(K):
            if val > best_vals[j]:
                # shift down
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
    group_idx_ptr,     # *int32, [M, 4]
    group_mask_ptr,    # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,   # 4
    group_count: tl.constexpr,  # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Initialize mask to zeros
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # Set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,      # *f32, [M, 8]
    masked_scores_ptr,   # *f32, [M, 256]
    M: tl.int32,
    N: tl.int32,         # 256
    stride_msm: tl.int32,
    stride_nsm: tl.int32,
):
    # Each program handles one token m
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(8):
        mask_val = tl.load(group_mask_ptr + pid * 8 + g)
        if mask_val == 0.0:
            offs = g * 32
            for i in range(32):
                ptr = masked_scores_ptr + pid * stride_msm + (offs + i) * stride_nsm
                tl.store(ptr, -float('inf'))


@triton.jit
def topk_experts_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    topk_idx_ptr,        # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    stride_msm: tl.int32,
    stride_nsm: tl.int32,
):
    # Iterative top-8 selection per token
    pid = tl.program_id(0)
    if pid >= M:
        return

    best_vals = tl.full((8,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((8,), dtype=tl.int32)

    for n in range(N):
        val = tl.load(masked_scores_ptr + pid * stride_msm + n * stride_nsm)
        for j in range(8):
            if val > best_vals[j]:
                for jj in range(7, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break

    for t in range(8):
        tl.store(topk_idx_ptr + pid * 8 + t, best_idxs[t])


# Helper Triton kernel: gather original scores at selected indices into selected_scores_ptr
# We’ll call this after we have topk_idx, from scores[M, N].
@triton.jit
def gather_selected_scores_kernel(
    scores_ptr,          # *f32, [M, N]
    topk_idx_ptr,        # *int32, [M, 8]
    selected_scores_ptr, # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    stride_sm: tl.int32,
    stride_sn: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(8):
        n_idx = tl.load(topk_idx_ptr + pid * 8 + t)
        val = tl.load(scores_ptr + pid * stride_sm + n_idx * stride_sn)
        tl.store(selected_scores_ptr + pid * 8 + t, val)


# Helper Triton kernel: normalize selected_scores per token and apply routed_scaling_factor
@triton.jit
def normalize_scale_kernel(
    selected_scores_ptr, # *f32, [M, 8]
    topk_weight_ptr,     # *f32, [M, 8]
    M: tl.int32,
    routed_scaling_factor: tl.float32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    denom = tl.zeros((), dtype=tl.float32)
    for t in range(8):
        val = tl.load(selected_scores_ptr + pid * 8 + t)
        denom += val
    denom = denom + 1e-20
    for t in range(8):
        val = tl.load(selected_scores_ptr + pid * 8 + t)
        normalized = val / denom
        scaled = normalized * routed_scaling_factor
        tl.store(topk_weight_ptr + pid * 8 + t, scaled)


def _choose_block_for_dim(x, dim):
    # simple heuristic
    if dim >= 2048:
        return 128
    elif dim >= 1024:
        return 128
    else:
        return 64


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-optimized version of the original routing, fully in Triton:
        - Computes logits = hidden_states @ weight.T
        - Computes scores = sigmoid(logits) + expert_bias
        - Performs group-limited top-k routing:
          * Reshape scores to [M, 8, 32], compute per-group top-2 sum → [M, 8]
          * Select top-4 groups per token → [M, 4]
          * Build group_mask [M, 8] (1 for selected groups)
          * Expand to [M, 256], set non-selected group’s 32 entries to -inf
          * Select top-8 experts → [M, 8]
          * Gather original scores at those 8 positions
          * Normalize by sum and apply routed_scaling_factor
        Returns:
          - topk_idx: [M, 8] int32
          - topk_weight: [M, 8] float32
        """
        # Ensure dtype and device
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be CUDA for Triton kernels."
        device = hidden_states.device

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]  # number of experts

        # 1) Triton: logits = hidden_states @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        BLOCK_M = _choose_block_for_dim(M, M)
        BLOCK_N = _choose_block_for_dim(N, N)
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_logits_kernel[grid](
            hidden_states, weight, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
        )

        # 2) Triton: scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        BLOCK_M = 128
        BLOCK_N = 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        sigmoid_add_bias_kernel[grid](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
        )

        # 3) Compute group_scores [M, 8] using Triton: sum of top-2 per group
        # We pass a flattened view of scores as [M, 8, 32] with 8*32=256 elements per token.
        # Since N=256, we can view scores as (M, 8, 32) directly via reshape.
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        # Ensure scores is contiguous for reshape
        scores = scores.contiguous()
        # Reshape to [M, 8, 32]
        scores_reshaped = scores.view(M, 8, 32)
        # Flatten for kernel: we can pass scores as flat pointer; we need to re-layout per token’s 256 elements.
        # However Triton kernel expects linear pointer. We can pass scores as flat pointer with per-token offset.
        # Each program will read scores_ptr + pid * (8*32) + g*32 + j
        group_scores[...] = -float('inf')  # initialize
        grid = (M,)
        group_top2_sum_kernel[grid](
            scores,                # flat pointer to [M, 8, 32]
            group_scores,
            M, 8, 32
        )

        # 4) Triton: top-4 group indices per token
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=device)
        grid = (M,)
        topk_group_kernel[grid](
            group_scores, group_idx, M, 4
        )

        # 5) Triton: build group_mask [M, 8] (1.0 where selected)
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=device)
        grid = (M,)
        build_group_mask_kernel[grid](
            group_idx, group_mask, M, 4, 8
        )

        # 6) Triton: expand group_mask to [M, 256] and set non-selected groups to -inf in masked_scores
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=device)
        # Initially set masked_scores = scores
        masked_scores.copy_(scores)
        stride_msm = masked_scores.stride(0)
        stride_nsm = masked_scores.stride(1)
        grid = (M,)
        expand_and_set_ninf_kernel[grid](
            group_mask, masked_scores, M, N, stride_msm, stride_nsm
        )

        # 7) Triton: top-8 expert indices per token from masked_scores
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        stride_msm = masked_scores.stride(0)
        stride_nsm = masked_scores.stride(1)
        grid = (M,)
        topk_experts_kernel[grid](
            masked_scores, topk_idx, M, N, stride_msm, stride_nsm
        )

        # 8) Triton: gather original scores at those 8 positions → selected_scores [M, 8]
        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_sm = scores.stride(0)
        stride_sn = scores.stride(1)
        grid = (M,)
        gather_selected_scores_kernel[grid](
            scores, topk_idx, selected_scores, M, N, stride_sm, stride_sn
        )

        # 9) Triton: normalize selected_scores by sum (add 1e-20) and apply routed_scaling_factor → final topk_weight
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        grid = (M,)
        normalize_scale_kernel[grid](
            selected_scores, topk_weight, M, routed_scaling_factor
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
