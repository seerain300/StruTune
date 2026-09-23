import torch
import triton
import triton.language as tl


@triton.jit
def matmul_logits_kernel(
    hidden_ptr,       # *f32, [M, K]
    weight_ptr,       # *f32, [N, K]
    logits_ptr,       # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_hm: tl.int32, stride_hk: tl.int32,
    stride_wk: tl.int32, stride_wn: tl.int32,
    stride_lm: tl.int32, stride_ln: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            hidden_ptr + (offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            weight_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
    out_ptrs = logits_ptr + (offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,       # *f32, [M, N]
    bias_ptr,         # *f32, [N]
    scores_ptr,       # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    stride_lm: tl.int32,
    stride_ln: tl.int32,
    stride_sm: tl.int32,
    stride_sn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    logits = tl.load(
        logits_ptr + (offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln),
        mask=mask,
        other=0.0,
    )
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)[None, :]
    scores = 1.0 / (1.0 + tl.exp(-logits)) + bias
    tl.store(scores_ptr + (offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn), scores, mask=mask)


@triton.jit
def group_top2_sum_kernel(
    scores_ptr,       # *f32, [M, N]
    group_scores_ptr, # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,      # 256
    stride_sm: tl.int32,
    stride_sn: tl.int32,
    stride_gsm: tl.int32,
    stride_gsn: tl.int32,  # stride along 8 groups (we treat as 1D here)
    group_count: tl.constexpr,  # 8
    experts_per_group: tl.constexpr,  # 32
):
    # Each program handles one token (row) m
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # Initialize top-2 arrays for each group
    top1 = tl.full((group_count,), -float('inf'), dtype=tl.float32)
    top2 = tl.full((group_count,), -float('inf'), dtype=tl.float32)

    # Loop over groups and compute top-2 per group
    for g in range(group_count):
        base = g * experts_per_group
        # Iterate over 32 experts in this group
        for i in range(experts_per_group):
            n_idx = base + i
            val = tl.load(scores_ptr + pid_m * stride_sm + n_idx * stride_sn)
            # Update top-2
            if val > top1[g]:
                top2[g] = top1[g]
                top1[g] = val
            elif val > top2[g]:
                top2[g] = val
    # Sum top-2 per group
    total = top1 + top2
    tl.store(group_scores_ptr + pid_m * group_count + tl.arange(0, group_count), total)


@triton.jit
def topk_group_kernel(
    group_scores_ptr, # *f32, [M, 8]
    group_idx_ptr,    # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,  # 4
    group_count: tl.constexpr,  # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for g in range(group_count):
        val = tl.load(group_scores_ptr + pid * group_count + g)
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
    group_idx_ptr,    # *int32, [M, 4]
    group_mask_ptr,   # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,  # 4
    group_count: tl.constexpr,  # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
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
    group_count: tl.constexpr,  # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        mask_val = tl.load(group_mask_ptr + pid * group_count + g)
        if mask_val == 0.0:
            base = g * experts_per_group
            for i in range(experts_per_group):
                ptr = masked_scores_ptr + pid * stride_msm + (base + i) * stride_nsm
                tl.store(ptr, -float('inf'))


@triton.jit
def topk_experts_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    topk_idx_ptr,        # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    stride_msm: tl.int32,
    stride_nsm: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for n in range(N):
        val = tl.load(masked_scores_ptr + pid * stride_msm + n * stride_nsm)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break
    for t in range(K):
        tl.store(topk_idx_ptr + pid * K + t, best_idxs[t])


@triton.jit
def gather_and_normalize_kernel(
    scores_ptr,          # *f32, [M, N]
    topk_idx_ptr,        # *int32, [M, 8]
    topk_weight_ptr,     # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    stride_sm: tl.int32,
    stride_sn: tl.int32,
    stride_wm: tl.int32,
    stride_wn: tl.int32,
    routed_scale: tl.float32,
    eps: tl.float32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    total = tl.zeros((), dtype=tl.float32)
    for t in range(K):
        n_idx = tl.load(topk_idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * stride_sm + n_idx * stride_sn)
        total += val
    inv = 1.0 / (total + eps)
    for t in range(K):
        n_idx = tl.load(topk_idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * stride_sm + n_idx * stride_sn) * inv * routed_scale
        tl.store(topk_weight_ptr + pid * K + t, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype and device
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be on CUDA for Triton."
        M = hidden_states.shape[0]
        N = weight.shape[0]
        K = hidden_states.shape[1]
        assert N == weight.shape[1], "weight shape must be [N, K]"
        assert K == 128, "This implementation expects hidden size K=128 as per original context."
        assert N == 256, "This implementation expects N=256 (256 experts) as per original context."

        # 1) Triton matmul: logits = hidden_states @ weight.T
        logits = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        matmul_logits_kernel[
            (triton.cdiv(M, 128), triton.cdiv(N, 128))
        ](
            hidden_states, weight,
            logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(1), weight.stride(0),
            logits.stride(0), logits.stride(1),
            128, 128, 32,
            num_warps=4, num_stages=3,
        )

        # 2) Triton elementwise: scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        sigmoid_add_bias_kernel[
            (triton.cdiv(M, 128), triton.cdiv(N, 128))
        ](
            logits, expert_bias,
            scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            128, 128,
            num_warps=4, num_stages=2,
        )

        # 3) Triton: group_scores [M, 8] = sum of top-2 per group from scores [M, 256]
        group_scores = torch.empty((M, 8), device=hidden_states.device, dtype=torch.float32)
        group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), 1,  # stride_gsn=1 for contiguous [M, 8]
            8, 32,
            num_warps=2, num_stages=1,
        )

        # 4) Triton: top-4 group indices per token → [M, 4]
        group_idx = torch.empty((M, 4), device=hidden_states.device, dtype=torch.int32)
        topk_group_kernel[(M,)](
            group_scores, group_idx,
            M, 4, 8,
            num_warps=1, num_stages=1,
        )

        # 5) Triton: build group_mask [M, 8] (float32, 1.0 where selected)
        group_mask = torch.empty((M, 8), device=hidden_states.device, dtype=torch.float32)
        build_group_mask_kernel[(M,)](
            group_idx, group_mask,
            M, 4, 8,
            num_warps=1, num_stages=1,
        )

        # 6) Triton: expand group_mask to [M, 256] and set non-selected groups to -inf in masked_scores
        masked_scores = scores.clone()
        # Strides for masked_scores
        stride_msm = masked_scores.stride(0)
        stride_nsm = masked_scores.stride(1)
        expand_and_set_ninf_kernel[(M,)](
            group_mask, masked_scores,
            M, N,
            stride_msm, stride_nsm,
            8, 32,
            num_warps=1, num_stages=1,
        )

        # 7) Triton: top-8 expert indices per token from masked_scores → [M, 8]
        topk_idx = torch.empty((M, 8), device=hidden_states.device, dtype=torch.int32)
        topk_experts_kernel[(M,)](
            masked_scores, topk_idx,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            8,
            num_warps=4, num_stages=2,
        )

        # 8) Triton: gather original scores at topk_idx and normalize, apply scaling
        topk_weight = torch.empty((M, 8), device=hidden_states.device, dtype=torch.float32)
        gather_and_normalize_kernel[(M,)](
            scores, topk_idx, topk_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_scaling_factor, 1e-20,
            8,
            num_warps=4, num_stages=2,
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
