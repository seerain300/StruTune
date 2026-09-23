import torch
import triton
import triton.language as tl

# Kernel 1: logits = hidden_states @ weight.T
@triton.jit
def matmul_logits_kernel(
    hidden_ptr,    # *f32, [M, K]
    weight_ptr,    # *f32, [N, K]
    logits_ptr,    # *f32, [M, N]
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
    offs_k = tl.arange(0, BLOCK_K)
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K
    for k in range(0, K, BLOCK_K):
        a_ptrs = hidden_ptr + (offs_m[:, None] * K + (k + offs_k)[None, :])
        b_ptrs = weight_ptr + (offs_n[None, :] * K + (k + offs_k)[:, None])
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k + offs_k[:, None] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    # Write back to logits
    logits_ptrs = logits_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(logits_ptrs, acc, mask=c_mask)

# Kernel 2: scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,    # *f32, [M, N]
    bias_ptr,      # *f32, [N]
    scores_ptr,    # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ptrs = logits_ptr + (offs_m[:, None] * N + offs_n[None, :])
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    logits = tl.load(ptrs, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs_n[None, :], mask=(offs_n[None, :] < N), other=0.0)
    scores = 1.0 / (1.0 + tl.exp(-logits)) + bias
    tl.store(scores_ptr + (offs_m[:, None] * N + offs_n[None, :]), scores, mask=mask)

# Kernel 3: group_top2_sum → output [M, 8]
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,     # *f32, [M, 256], already scores = sigmoid(logits) + bias
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    group_count: tl.constexpr,    # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)  # one program per token
    if pid >= M:
        return
    # We need to compute per-group top-2 and sum for this token
    # Loop over groups 0..7
    for g in range(group_count):
        base = g * experts_per_group
        best1 = -float('inf')
        best2 = -float('inf')
        # loop over 32 experts in this group
        for i in range(experts_per_group):
            idx = base + i
            val = tl.load(scores_ptr + pid * 256 + idx)
            # Update top-2
            if val > best1:
                best2 = best1
                best1 = val
            elif val > best2:
                best2 = val
        total = best1 + best2
        tl.store(group_scores_ptr + pid * group_count + g, total)

# Kernel 4: topk_group → selected group indices [M, 4]
@triton.jit
def topk_group_kernel(
    group_scores_ptr,    # *f32, [M, 8]
    selected_idx_ptr,    # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,     # 4
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for g in range(8):
        val = tl.load(group_scores_ptr + pid * 8 + g)
        # Insertion into top-K (descending)
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
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])

# Kernel 5: build_group_mask → [M, 8], 1 at selected groups
@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,       # *int32, [M, 4]
    group_mask_ptr,      # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 4
    group_count: tl.constexpr,  # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # initialize zeros
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)

# Kernel 6: expand_and_set_ninf → masked_scores [M, 256], set non-selected groups' 32 entries to -inf
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,      # *f32, [M, 8]
    masked_scores_ptr,   # *f32, [M, 256]
    M: tl.int32,
    N: tl.int32,         # 256
    group_count: tl.constexpr,  # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        if tl.load(group_mask_ptr + pid * group_count + g) == 0.0:
            base = g * experts_per_group
            for i in range(experts_per_group):
                idx = base + i
                ptr = masked_scores_ptr + pid * N + idx
                # Set to -inf
                tl.store(ptr, -float('inf'))

# Kernel 7: topk_final → final top-8 expert indices [M, 8] from masked_scores
@triton.jit
def topk_final_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    final_idx_ptr,       # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for i in range(N):
        val = tl.load(masked_scores_ptr + pid * N + i)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = i
                break
    for t in range(K):
        tl.store(final_idx_ptr + pid * K + t, best_idxs[t])

# Kernel 8: gather_selected_scores at final_idx from scores → gathered_scores [M, 8]
@triton.jit
def gather_selected_scores_kernel(
    scores_ptr,          # *f32, [M, 256] (this is actually scores [M, N] post-bias)
    final_idx_ptr,       # *int32, [M, 8]
    gathered_scores_ptr, # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # N=256
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(final_idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * N + idx)
        tl.store(gathered_scores_ptr + pid * K + t, val)

# Kernel 9: normalize_and_scale → topk_weight [M, 8]
@triton.jit
def normalize_and_scale_kernel(
    gathered_scores_ptr, # *f32, [M, 8]
    topk_weight_ptr,     # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
    scale: tl.float32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    sum_val = 0.0
    for t in range(K):
        val = tl.load(gathered_scores_ptr + pid * K + t)
        sum_val += val
    # sum_val > 0.0 is ensured since masked top-k selects from non -inf and normalization uses only those
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
        N = weight.shape[0]  # 256
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

        # 3) Triton: group_top2_sum → [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid3 = (M,)
        group_top2_sum_kernel[grid3](scores, group_scores, M, 8, 32)

        # 4) Triton: topk_group → [M, 4] int32
        selected_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden.device)
        grid4 = (M,)
        topk_group_kernel[grid4](group_scores, selected_idx, M, 4)

        # 5) Triton: build_group_mask → [M, 8] float
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid5 = (M,)
        build_group_mask_kernel[grid5](selected_idx, group_mask, M, 4, 8)

        # 6) Triton: expand_and_set_ninf → masked_scores [M, 256], set non-selected to -inf
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        # Initialize with scores
        masked_scores.copy_(scores)
        grid6 = (M,)
        expand_and_set_ninf_kernel[grid6](group_mask, masked_scores, M, N, 8, 32)

        # 7) Triton: topk_final → final indices [M, 8] int32
        final_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden.device)
        grid7 = (M,)
        topk_final_kernel[grid7](masked_scores, final_idx, M, N, 8)

        # 8) Triton: gather selected original scores from 'scores' at final_idx → [M, 8]
        # Note: 'scores' is the post-bias scores tensor we created in Triton step 2.
        gathered_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid8 = (M,)
        gather_selected_scores_kernel[grid8](scores, final_idx, gathered_scores, M, N, 8)

        # 9) Triton: normalize and scale
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid9 = (M,)
        normalize_and_scale_kernel[grid9](gathered_scores, topk_weight, M, 8, routed_scaling_factor)

        # Return indices and normalized weights (as in original: idx, weight)
        # Note: We return final_idx and topk_weight which match the original outputs.
        return final_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
