import torch
import triton
import triton.language as tl

# 1) Triton: matmul_logits_kernel computes logits = hidden_states @ weight.T
@triton.jit
def matmul_logits_kernel(
    a_ptr,        # *f32, [M, K]
    b_ptr,        # *f32, [N, K] (we will pass weight as [N, K], treat as B)
    out_ptr,      # *f32, [M, N]
    M: tl.int32,  # num_tokens
    K: tl.int32,  # hidden size
    N: tl.int32,  # num_experts
    stride_am: tl.int32,  # a.stride(0)
    stride_ak: tl.int32,  # a.stride(1) = K
    stride_bk: tl.int32,  # b.stride(1) = K
    stride_bn: tl.int32,  # b.stride(0) = N
    stride_om: tl.int32,  # out.stride(0)
    stride_on: tl.int32,  # out.stride(1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        # a: [M, K] -> ptrs a_ptr + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak
        a_block = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )
        # b: [N, K], but we need B^T [K, N] -> we load as (offs_k, offs_n) from b
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_block = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=b_mask,
            other=0.0,
        )
        acc += tl.dot(a_block, b_block)
    # Write back
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=out_mask,
    )


# 2) Triton: sigmoid_add_bias_kernel computes scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,   # *f32, [M, N]
    bias_ptr,     # *f32, [N]
    out_ptr,      # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    stride_lm: tl.int32,
    stride_ln: tl.int32,
    stride_om: tl.int32,
    stride_on: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    M_N = M * N
    if pid_m * N + pid_n >= M_N:
        return
    m = pid_m
    n = pid_n
    val = tl.load(logits_ptr + m * stride_lm + n * stride_ln)
    bias_val = tl.load(bias_ptr + n)
    # sigmoid
    sig = 1.0 / (1.0 + tl.exp(-val))
    out = sig + bias_val
    tl.store(out_ptr + m * stride_om + n * stride_on, out)


# 3) Triton: group_top2_sum_kernel computes per-token group_scores [M, 8]
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,     # *f32, [M, 256]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    group_count: tl.constexpr,  # 8
    ep_count: tl.constexpr,     # 32
    stride_sm: tl.int32,        # scores.stride(0)
    stride_sn: tl.int32,        # scores.stride(1)
    stride_gsm: tl.int32,       # group_scores.stride(0)
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        # initialize top2
        top1 = tl.full((), -float('inf'), dtype=tl.float32)
        top2 = tl.full((), -float('inf'), dtype=tl.float32)
        for ep in range(ep_count):
            idx = g * ep_count + ep
            val = tl.load(scores_ptr + pid * stride_sm + idx * stride_sn)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = top1 + top2
        tl.store(group_scores_ptr + pid * group_count + g, total)


# 4) Triton: topk_group_kernel (K=4) — return indices (int32) of selected groups per token
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
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = g
                break
    for t in range(K):
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])


# 5) Triton: build_group_mask_kernel — given selected group_idx [M, 4], set group_mask [M, 8] to 1 at selected positions
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
    # set zeros
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


# 6) Triton: expand_and_set_ninf_kernel — expand group_mask [M, 8] to [M, 256], set non-selected groups' 32 entries to -inf
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,      # *f32, [M, 8]
    masked_scores_ptr,   # *f32, [M, 256], will be mutated in-place
    M: tl.int32,
    N: tl.int32,         # 256
    stride_mm: tl.int32,
    stride_mn: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(8):
        if tl.load(group_mask_ptr + pid * 8 + g) != 1.0:
            base = g * 32
            for ep in range(32):
                idx = base + ep
                tl.store(masked_scores_ptr + pid * stride_mm + idx * stride_mn, -float('inf'))


# 7) Triton: topk_expert_kernel (K=8) — select top-8 experts from masked_scores [M, 256]
@triton.jit
def topk_expert_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    selected_expert_ptr, # *int32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for n in range(256):
        val = tl.load(masked_scores_ptr + pid * 256 + n)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break
    for t in range(K):
        tl.store(selected_expert_ptr + pid * K + t, best_idxs[t])


# 8) Triton: gather_original_scores_kernel — gather original scores [M, 8] from scores [M, 256] using final_expert_idx
@triton.jit
def gather_original_scores_kernel(
    scores_ptr,          # *f32, [M, 256]
    expert_idx_ptr,      # *int32, [M, 8]
    out_scores_ptr,      # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(expert_idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * 256 + idx)
        tl.store(out_scores_ptr + pid * K + t, val)


# 9) Triton: normalize_and_scale_kernel — compute normalized weights and apply routed_scaling_factor
@triton.jit
def normalize_and_scale_kernel(
    original_scores_ptr, # *f32, [M, 8]
    out_weights_ptr,     # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
    scale_factor: tl.float32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    total = tl.full((), 0.0, dtype=tl.float32)
    for t in range(K):
        val = tl.load(original_scores_ptr + pid * K + t)
        total += val
    eps = 1e-20
    total = tl.maximum(total, eps)
    for t in range(K):
        val = tl.load(original_scores_ptr + pid * K + t)
        norm = val / total
        scaled = norm * scale_factor
        tl.store(out_weights_ptr + pid * K + t, scaled)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype and contiguity; compute in fp32 for numerical stability
        hidden_states = hidden_states.contiguous().to(torch.float32)  # [M, 128]
        weight = weight.contiguous().to(torch.float32)               # [256, 128]
        expert_bias = expert_bias.contiguous().to(torch.float32)     # [256]

        M = hidden_states.shape[0]  # num_tokens
        K = hidden_states.shape[1]  # 128
        N = weight.shape[0]         # 256

        # 1) Compute logits = hidden_states @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        matmul_logits_kernel[grid](
            hidden_states, weight, logits,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(1), weight.stride(0),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )

        # 2) scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid2 = (M, N)
        sigmoid_add_bias_kernel[grid2](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
        )

        # 3) group_top2_sum: [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        grid3 = (M,)
        group_top2_sum_kernel[grid3](
            scores, group_scores,
            M,
            group_count=8, ep_count=32,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_gsm=group_scores.stride(0),
        )

        # 4) topk_group: [M, 4]
        selected_group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden_states.device)
        grid4 = (M,)
        topk_group_kernel[grid4](
            group_scores, selected_group_idx,
            M, K=4,
        )

        # 5) build group_mask [M, 8]
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        grid5 = (M,)
        build_group_mask_kernel[grid5](
            selected_group_idx, group_mask,
            M, K=4, group_count=8,
        )

        # 6) expand and set -inf for non-selected groups
        masked_scores = scores.clone()  # in-place to -inf via kernel
        grid6 = (M,)
        expand_and_set_ninf_kernel[grid6](
            group_mask, masked_scores,
            M, N=256,
            stride_mm=masked_scores.stride(0), stride_mn=masked_scores.stride(1),
        )

        # 7) select final top-8 experts from masked_scores
        final_expert_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden_states.device)
        grid7 = (M,)
        topk_expert_kernel[grid7](
            masked_scores, final_expert_idx,
            M, K=8,
        )

        # 8) gather original scores for those 8 experts
        original_selected_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        grid8 = (M,)
        gather_original_scores_kernel[grid8](
            scores, final_expert_idx, original_selected_scores,
            M, K=8,
        )

        # 9) normalize and apply scaling
        final_weights = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        grid9 = (M,)
        normalize_and_scale_kernel[grid9](
            original_selected_scores, final_weights,
            M, K=8, scale_factor=self.routed_scaling_factor,
        )

        # Return indices and normalized weights
        # Note: topk_expert_idx is final_expert_idx; the original returns indices and weights.
        # Adjust if API expects different naming; here we match the required output signature.
        topk_idx = final_expert_idx  # [M, 8], but logic selects top-8, so this matches
        topk_weight = final_weights   # [M, 8]
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
