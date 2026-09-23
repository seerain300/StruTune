import torch
import triton
import triton.language as tl

# 1) Triton matmul: logits = hidden_states @ weight.T
@triton.jit
def logits_kernel(
    A_ptr,           # *f32, [M, K]
    B_ptr,           # *f32, [N, K]
    C_ptr,           # *f32, [M, N]
    M: tl.int32,     # num_tokens
    N: tl.int32,     # num_experts (256)
    K: tl.int32,     # hidden_size (128)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # A block: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K) + offs_k[None, :]
        # B block: [BLOCK_K, BLOCK_N], note B is [N, K], we want [K, N]
        b_ptrs = B_ptr + (offs_n[None, :] * K) + offs_k[:, None]
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    # write C
    c_ptrs = C_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 2) Triton elementwise: scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,      # *f32, [M, N]
    bias_ptr,        # *f32, [N]
    scores_ptr,      # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m
    offs_n = pid_n
    # safe loads
    logits = tl.load(logits_ptr + offs_m * N + offs_n)
    bias = tl.load(bias_ptr + offs_n)
    # sigmoid
    s = 1.0 / (1.0 + tl.exp(-logits))
    out = s + bias
    tl.store(scores_ptr + offs_m * N + offs_n, out)

# 3) Triton: compute per-token group_scores [M, 8] (sum of top-2 per group from scores reshaped [M, 8, 32])
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,       # *f32, [M, 256]
    group_scores_ptr, # *f32, [M, 8]
    M: tl.int32,
    group_count: tl.constexpr,     # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)  # one program per token
    if pid >= M:
        return
    total = 0.0
    for g in range(group_count):
        group_sum = 0.0
        base = g * experts_per_group
        # find top-2 within this group of 32
        for j in range(experts_per_group):
            idx = base + j
            val = tl.load(scores_ptr + pid * 256 + idx)  # scores for this token
            # simple find-top-2 loop
            max1 = -float('inf')
            max2 = -float('inf')
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
            group_sum += max1 + max2
        total += group_sum
    tl.store(group_scores_ptr + pid * 8, total)

# 4) Triton: top-4 groups per token from group_scores [M, 8] -> indices [M, 4]
@triton.jit
def topk_group_kernel(
    group_scores_ptr, # *f32, [M, 8]
    selected_idx_ptr, # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,      # 4
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
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])

# 5) Triton: build_group_mask [M, 8] from selected group indices [M, 4], set 1 at selected positions
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
    # initialize zeros
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)

# 6) Triton: expand group_mask to [M, 256] and set non-selected groups' 32 entries to -inf
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,     # *f32, [M, 8]
    masked_scores_ptr,  # *f32, [M, 256], will be mutated in-place
    M: tl.int32,
    N: tl.int32,        # 256
    group_count: tl.constexpr,  # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        is_one = tl.load(group_mask_ptr + pid * group_count + g)  # 0.0 or 1.0
        if is_one == 0.0:
            base = g * experts_per_group
            for j in range(experts_per_group):
                idx = base + j
                # set -inf for this group's entries
                val = tl.load(masked_scores_ptr + pid * N + idx)
                val = -float('inf')
                tl.store(masked_scores_ptr + pid * N + idx, val)

# 7) Triton: masked top-8 experts selection on masked_scores [M, 256] -> indices [M, 8]
@triton.jit
def masked_top8_experts_kernel(
    masked_scores_ptr,  # *f32, [M, 256]
    selected_idx_ptr,   # *int32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 8
    N: tl.int32,        # 256
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for n in range(0, N):
        val = tl.load(masked_scores_ptr + pid * N + n)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break
    for t in range(K):
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])

# 8) Triton: gather original scores for selected indices from scores [M, 256] -> [M, 8]
@triton.jit
def gather_selected_scores_kernel(
    scores_ptr,         # *f32, [M, 256] (sigmoid(logits) + bias)
    idx_ptr,            # *int32, [M, 8]
    gathered_ptr,       # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 8
    N: tl.int32,        # 256
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        n_idx = tl.load(idx_ptr + pid * K + t)  # int32 expert index
        val = tl.load(scores_ptr + pid * N + n_idx)
        tl.store(gathered_ptr + pid * K + t, val)

# 9) Triton: normalize and scale selected scores: weight = gathered / sum + eps * routed_scaling_factor
@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,       # *f32, [M, 8]
    weight_ptr,         # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 8
    eps: tl.float32,    # 1e-20
    scaling: tl.float32,  # routed_scaling_factor
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    total = 0.0
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t)
        total += val
    total = total + eps
    inv = 1.0 / total
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t) * scaling
        tl.store(weight_ptr + pid * K + t, val * inv)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants
        self.num_experts = 256
        self.group_count = 8
        self.experts_per_group = self.num_experts // self.group_count  # 32

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtypes and devices
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors."
        assert hidden_states.ndim == 2, "hidden_states must be [num_tokens, hidden_size]"
        assert weight.ndim == 2 and weight.shape[1] == hidden_states.shape[1], "weight must be [num_experts, hidden_size]"
        assert expert_bias.ndim == 1 and expert_bias.shape[0] == self.num_experts, "expert_bias must be [num_experts]"
        M = hidden_states.shape[0]
        N = self.num_experts

        # 1) Triton matmul logits: [M, N]
        # hidden_states: [M, 128], weight: [N, 128] -> logits [M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32  # good defaults for N=256, K=128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        logits_kernel[grid](
            hidden_states, weight, logits,
            M, N, 128,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # 2) Triton elementwise scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=logits.device)
        grid2 = (M, N)
        sigmoid_add_bias_kernel[grid2](
            logits, expert_bias, scores,
            M, N,
        )

        # 3) Triton: per-token group_scores [M, 8] = sum of top-2 per group
        group_scores = torch.empty((M, self.group_count), dtype=torch.float32, device=scores.device)
        grid3 = (M,)
        group_top2_sum_kernel[grid3](
            scores, group_scores,
            M,
            self.group_count, self.experts_per_group
        )

        # 4) Triton: select top-4 groups per token -> [M, 4]
        selected_idx = torch.empty((M, 4), dtype=torch.int32, device=scores.device)
        grid4 = (M,)
        topk_group_kernel[grid4](
            group_scores, selected_idx,
            M, 4
        )

        # 5) Triton: build group_mask [M, 8]
        group_mask = torch.empty((M, self.group_count), dtype=torch.float32, device=scores.device)
        grid5 = (M,)
        build_group_mask_kernel[grid5](
            selected_idx, group_mask,
            M, 4, self.group_count
        )

        # 6) Triton: expand group_mask to [M, 256] and set non-selected groups to -inf in masked_scores
        masked_scores = scores.clone()  # [M, 256]
        grid6 = (M,)
        expand_and_set_ninf_kernel[grid6](
            group_mask, masked_scores,
            M, N, self.group_count, self.experts_per_group
        )

        # 7) Triton: masked top-8 experts per token -> [M, 8]
        final_idx = torch.empty((M, 8), dtype=torch.int32, device=scores.device)
        grid7 = (M,)
        masked_top8_experts_kernel[grid7](
            masked_scores, final_idx,
            M, 8, N
        )

        # 8) Triton: gather original scores for those 8 from 'scores' (which equals sigmoid(logits) + bias)
        gathered = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        grid8 = (M,)
        gather_selected_scores_kernel[grid8](
            scores, final_idx, gathered,
            M, 8, N
        )

        # 9) Triton: normalize and scale -> [M, 8]
        weight = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        grid9 = (M,)
        normalize_and_scale_kernel[grid9](
            gathered, weight,
            M, 8, 1e-20, routed_scaling_factor
        )

        return final_idx, weight

# For testing: ensure ModelNew runs; original Model uses torch.sigmoid and topk, this uses Triton kernels.
# Note: You can compare outputs against the original Model with random inputs on CUDA.


def run(*args):
    return ModelNew()(*args)
