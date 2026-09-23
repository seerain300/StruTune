import torch
import triton
import triton.language as tl


# 1) Triton matmul: C[M, N] = A[M, K] @ B[N, K]^T, where B is weight [N, K], we pass as [N, K] and compute dot over K.
@triton.jit
def matmul_kernel(
    A_ptr,  # *f32, [M, K]
    B_ptr,  # *f32, [N, K]
    C_ptr,  # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    BLOCK_M: tl.constexpr,  # e.g., 128
    BLOCK_N: tl.constexpr,  # e.g., 32
    BLOCK_K: tl.constexpr,  # e.g., 32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k

        # A block: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * K) + k_ids[None, :]
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # B block: [BLOCK_K, BLOCK_N], B is [N, K]
        B_ptrs = B_ptr + (offs_n[None, :] * K) + k_ids[:, None]
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    # write C
    C_ptrs = C_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


# 2) Triton elementwise: C = sigmoid(A) + B
@triton.jit
def sigmoid_add_bias_kernel(
    A_ptr,    # *f32, [M, N]
    B_ptr,    # *f32, [N] (bias)
    C_ptr,    # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    BLOCK_M: tl.constexpr,  # e.g., 128
    BLOCK_N: tl.constexpr,  # e.g., 64
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_ptrs = A_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    C_ptrs = C_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(A_ptrs, mask=mask, other=0.0)
    # sigmoid
    s = 1.0 / (1.0 + tl.exp(-a))
    # add bias: broadcast bias across rows
    bias_ptrs = B_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
    c = s + bias[None, :]  # broadcast
    tl.store(C_ptrs, c, mask=mask)


# 3) Triton: group_top2_sum — compute per-token group_scores [M, 8]
# Input: scores [M, N], treat as [M, 8, 32] to find top-2 per group and sum.
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,         # *f32, [M, N]
    group_scores_ptr,   # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,        # 256
    group_count: tl.constexpr,  # 8
    per_group: tl.constexpr,    # 32
):
    pid = tl.program_id(0)  # one program per token
    if pid >= M:
        return
    # Loop over groups
    for g in range(group_count):
        base = g * per_group
        # top-2 values as scalars
        top1 = -float('inf')
        top2 = -float('inf')
        # loop over 32
        for j in range(per_group):
            val = tl.load(scores_ptr + pid * N + base + j)
            # update top-2
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = top1 + top2
        tl.store(group_scores_ptr + pid * group_count + g, total)


# 4) Triton: topk_group — select top-4 groups per token (K=4) and write indices [M, 4]
@triton.jit
def topk_group_kernel(
    group_scores_ptr,   # *f32, [M, 8]
    group_idx_ptr,      # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,    # 4
):
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


# 5) Triton: build_group_mask — given selected group_idx [M, 4], set group_mask [M, 8] to 1 at selected positions
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
    # initialize zeros
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


# 6) Triton: expand_and_negate_nonselected — expand group_mask [M, 8] to [M, 256], set non-selected groups' 32 entries to -inf
@triton.jit
def expand_and_negate_nonselected_kernel(
    group_mask_ptr,      # *f32, [M, 8]
    masked_scores_ptr,   # *f32, [M, 256], we will write negated scores for non-selected groups
    M: tl.int32,
    N: tl.int32,         # 256
    group_count: tl.constexpr,  # 8
    per_group: tl.constexpr,    # 32
):
    pid = tl.program_id(0)  # one program per token
    if pid >= M:
        return
    # For each group
    for g in range(group_count):
        sel = tl.load(group_mask_ptr + pid * group_count + g)  # 1.0 or 0.0
        base = g * per_group
        # For selected: keep original scores
        # For non-selected: set to -inf
        neg_large = -1e20  # large negative sentinel
        for j in range(per_group):
            val = tl.load(masked_scores_ptr + pid * N + base + j)
            new_val = tl.where(sel > 0.5, val, neg_large)
            tl.store(masked_scores_ptr + pid * N + base + j, new_val)


# 7) Triton: topk_experts — select top-8 expert indices from masked_scores [M, 256] → [M, 8]
@triton.jit
def topk_experts_kernel(
    scores_ptr,          # *f32, [M, 256]
    topk_idx_ptr,        # *int32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    N = 256
    for j in range(N):
        val = tl.load(scores_ptr + pid * N + j)
        for k in range(K):
            if val > best_vals[k]:
                for kk in range(K - 1, k, -1):
                    best_vals[kk] = best_vals[kk - 1]
                    best_idxs[kk] = best_idxs[kk - 1]
                best_vals[k] = val
                best_idxs[k] = j
                break
    for t in range(K):
        tl.store(topk_idx_ptr + pid * K + t, best_idxs[t])


# 8) Triton: gather_original_scores — given topk_expert indices [M, 8], gather original_logits (pre-bias) [M, 256] at those positions → [M, 8]
@triton.jit
def gather_original_scores_kernel(
    original_logits_ptr,  # *f32, [M, 256]
    indices_ptr,          # *int32, [M, 8]
    gathered_ptr,         # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,      # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(indices_ptr + pid * K + t)  # 0..255
        val = tl.load(original_logits_ptr + pid * 256 + idx)
        tl.store(gathered_ptr + pid * K + t, val)


# 9) Triton: normalize_and_scale — normalize each row by sum of gathered scores and apply routed_scaling_factor
@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,         # *f32, [M, 8]
    routed_scale,         # f32 scalar
    output_ptr,           # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,      # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    sum_val = 0.0
    for t in range(K):
        v = tl.load(gathered_ptr + pid * K + t)
        sum_val += v
    inv = 1.0 / (sum_val + 1e-20)
    for t in range(K):
        v = tl.load(gathered_ptr + pid * K + t) * routed_scale
        tl.store(output_ptr + pid * K + t, v * inv)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [M, 128], float32
        weight: torch.Tensor,           # [N, 128] where N=256, float32
        expert_bias: torch.Tensor,      # [N], float32
        routed_scaling_factor: float = 1.0,
    ):
        # Ensure contiguity and dtype
        hidden_states = hidden_states.contiguous().to(torch.float32)
        weight = weight.contiguous().to(torch.float32)  # [N, 128]
        expert_bias = expert_bias.contiguous().to(torch.float32)  # [N]
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]  # 128
        N = weight.shape[0]         # 256
        group_count = 8
        per_group = 32

        # 1) Matmul: logits [M, N]
        logits = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        # launch grid
        grid = (triton.cdiv(M, 128), triton.cdiv(N, 32))
        matmul_kernel[grid](
            hidden_states, weight, logits,
            M, N, K,
            BLOCK_M=128, BLOCK_N=32, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

        # 2) scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid2 = (triton.cdiv(M, 128), triton.cdiv(N, 64))
        sigmoid_add_bias_kernel[grid2](
            logits, expert_bias, scores,
            M, N,
            BLOCK_M=128, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        # 3) per-group top-2 sum: [M, 8]
        group_scores = torch.empty((M, group_count), device=hidden_states.device, dtype=torch.float32)
        group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N, group_count, per_group,
            num_warps=1, num_stages=1
        )

        # 4) top-4 groups per token: [M, 4]
        group_idx = torch.empty((M, 4), device=hidden_states.device, dtype=torch.int32)
        topk_group_kernel[(M,)](
            group_scores, group_idx,
            M, K=4,
            num_warps=1, num_stages=1
        )

        # 5) group_mask [M, 8]
        group_mask = torch.empty((M, group_count), device=hidden_states.device, dtype=torch.float32)
        build_group_mask_kernel[(M,)](
            group_idx, group_mask,
            M, K=4, group_count=8,
            num_warps=1, num_stages=1
        )

        # 6) expand and set non-selected to -inf in masked_scores [M, 256]
        masked_scores = scores.clone()  # keep original post-bias scores to mark non-selected as -inf
        expand_and_negate_nonselected_kernel[(M,)](
            group_mask, masked_scores,
            M, N, group_count=8, per_group=32,
            num_warps=1, num_stages=1
        )

        # 7) top-8 experts from masked_scores: [M, 8]
        topk_idx = torch.empty((M, 8), device=hidden_states.device, dtype=torch.int32)
        topk_experts_kernel[(M,)](
            masked_scores, topk_idx,
            M, K=8,
            num_warps=1, num_stages=1
        )

        # 8) gather original logits (pre-bias) at top-8 indices: [M, 8]
        original_logits = logits  # we computed logits in Triton matmul
        gathered_scores = torch.empty((M, 8), device=hidden_states.device, dtype=torch.float32)
        gather_original_scores_kernel[(M,)](
            original_logits, topk_idx, gathered_scores,
            M, K=8,
            num_warps=1, num_stages=1
        )

        # 9) normalize and apply scaling
        output_weights = torch.empty((M, 8), device=hidden_states.device, dtype=torch.float32)
        normalize_and_scale_kernel[(M,)](
            gathered_scores, self.routed_scaling_factor, output_weights,
            M, K=8,
            num_warps=1, num_stages=1
        )

        # Return selected expert indices and normalized weights
        return topk_idx, output_weights


def run(*args):
    return ModelNew()(*args)
