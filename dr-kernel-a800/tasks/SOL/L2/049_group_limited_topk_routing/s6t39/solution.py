import torch
import triton
import triton.language as tl


@triton.jit
def matmul_logits_kernel(
    a_ptr,        # *f32, [M, K]
    b_ptr,        # *f32, [N, K]
    c_ptr,        # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: program_id(0) over rows, program_id(1) over cols
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Pointers to tiles in A and B
        a = tl.load(
            a_ptr + (offs_m[:, None] * K + (k + offs_k[None, :])),
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + ((k + offs_k[:, None]) * N + offs_n[None, :]),
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    # Write back
    tl.store(
        c_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,   # *f32, [M, N]
    bias_ptr,     # *f32, [N]
    scores_ptr,   # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logits = tl.load(
        logits_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    )
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
    scores = 1.0 / (1.0 + tl.exp(-logits)) + bias[None, :]
    tl.store(
        scores_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        scores,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def group_top2_sum_kernel(
    scores_ptr,   # *f32, [M, N]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,
    group_count: tl.constexpr,   # 8
    experts_per_group: tl.constexpr,  # 32
):
    # Each program handles one token
    pid = tl.program_id(0)
    if pid >= M:
        return

    base = pid * N
    for g in range(group_count):
        group_base = base + g * experts_per_group
        top1 = -float('inf')
        top2 = -float('inf')
        # Loop over 32 experts in the group
        for i in range(experts_per_group):
            val = tl.load(scores_ptr + group_base + i)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        tl.store(group_scores_ptr + pid * group_count + g, top1 + top2)


@triton.jit
def topk_group_kernel(
    group_scores_ptr,   # *f32, [M, 8]
    selected_idx_ptr,   # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,    # 4
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    # Iteratively select top-K
    for g in range(8):
        val = tl.load(group_scores_ptr + pid * 8 + g)
        # Insert into best_vals/best_idxs
        for j in range(K):
            if val > best_vals[j]:
                # shift down
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = g
                break
    # Write out
    for t in range(K):
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])


@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,      # *int32, [M, 4]
    group_mask_ptr,     # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 4
    group_count: tl.constexpr,  # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # zero init
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,     # *f32, [M, 8]
    masked_scores_ptr,  # *f32, [M, 256], will be mutated in-place
    M: tl.int32,
    N: tl.int32,        # 256
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # base row pointer
    base_row = masked_scores_ptr + pid * N
    for g in range(8):
        active = tl.load(group_mask_ptr + pid * 8 + g)
        if active != 1.0:
            start = g * 32
            # set all entries of this group to -inf
            for i in range(32):
                tl.store(base_row + start + i, -float('inf'))


@triton.jit
def topk_final_kernel(
    masked_scores_ptr,  # *f32, [M, 256]
    final_idx_ptr,      # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,        # 256
    K: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    base_row = masked_scores_ptr + pid * N
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for e in range(N):
        val = tl.load(base_row + e)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = e
                break
    for t in range(K):
        tl.store(final_idx_ptr + pid * K + t, best_idxs[t])


@triton.jit
def gather_selected_scores_kernel(
    scores_ptr,         # *f32, [M, N]
    final_idx_ptr,      # *int32, [M, 8]
    gathered_ptr,       # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,        # 256
    K: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    base = scores_ptr + pid * N
    for t in range(K):
        e = tl.load(final_idx_ptr + pid * K + t)
        val = tl.load(base + e)
        tl.store(gathered_ptr + pid * K + t, val)


@triton.jit
def normalize_and_scale_kernel(
    gathered_scores_ptr,  # *f32, [M, 8]
    topk_weight_ptr,      # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,      # 8
    scale: tl.float32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    sum_val = 0.0
    for t in range(K):
        val = tl.load(gathered_scores_ptr + pid * K + t)
        sum_val += val
    inv = 1.0 / sum_val
    for t in range(K):
        val = tl.load(gathered_scores_ptr + pid * K + t) * inv
        val = val * scale
        tl.store(topk_weight_ptr + pid * K + t, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure contiguity and dtype
        hidden = hidden_states.contiguous().to(torch.float32)  # [M, K]
        weight = weight.contiguous().to(torch.float32)         # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)      # [N]
        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]  # expected 256

        # 1) Triton matmul for logits: [M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_logits_kernel[grid](hidden, weight, logits, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)

        # 2) Triton: scores = sigmoid(logits) + bias
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        BLOCK_M2, BLOCK_N2 = 64, 64
        grid2 = (triton.cdiv(M, BLOCK_M2), triton.cdiv(N, BLOCK_N2))
        sigmoid_add_bias_kernel[grid2](logits, bias, scores, M, N, BLOCK_M2, BLOCK_N2)

        # 3) Triton: compute group_scores [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid3 = (M,)
        group_top2_sum_kernel[grid3](scores, group_scores, M, N, 8, 32)

        # 4) Triton: select top-4 group indices [M, 4]
        selected_group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden.device)
        topk_group_kernel[(M,)](group_scores, selected_group_idx, M, 4)

        # 5) Triton: build group_mask [M, 8] (float 0.0/1.0)
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        build_group_mask_kernel[(M,)](selected_group_idx, group_mask, M, 4, 8)

        # 6) Triton: expand group_mask to [M, 256] and set non-selected groups to -inf
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        expand_and_set_ninf_kernel[(M,)](group_mask, masked_scores, M, N)

        # 7) Triton: select final top-8 expert indices [M, 8]
        final_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden.device)
        topk_final_kernel[(M,)](masked_scores, final_idx, M, N, 8)

        # 8) Triton: gather selected original scores from 'scores' at final_idx → [M, 8]
        gathered_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        gather_selected_scores_kernel[(M,)](scores, final_idx, gathered_scores, M, N, 8)

        # 9) Triton: normalize and scale by routed_scaling_factor → [M, 8]
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        normalize_and_scale_kernel[(M,)](gathered_scores, topk_weight, M, 8, routed_scaling_factor)

        return final_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
