import torch
import triton
import triton.language as tl


@triton.jit
def matmul_logits_kernel(
    a_ptr,  # *f32, [M, K]
    b_ptr,  # *f32, [N, K]
    out_ptr,  # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,   # strides for a
    stride_bn: tl.int32, stride_bk: tl.int32,   # strides for b
    stride_om: tl.int32, stride_on: tl.int32,   # strides for out
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
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
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,   # *f32, [M, N]
    bias_ptr,     # *f32, [N]
    scores_ptr,   # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    stride_lm: tl.int32, stride_ln: tl.int32,
    stride_sm: tl.int32, stride_sn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
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
    scores_ptr,   # *f32, [M, N] with N=256
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,   # 256
    stride_sm: tl.int32, stride_sn: tl.int32,
    group_count: tl.constexpr,   # 8
    experts_per_group: tl.constexpr,   # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        acc = tl.zeros((), dtype=tl.float32)
        start = g * experts_per_group
        for j in range(experts_per_group):
            score = tl.load(scores_ptr + pid * N + start + j)
            # compute top-2
            max1 = -float('inf')
            max2 = -float('inf')
            for jj in range(experts_per_group):
                s = tl.load(scores_ptr + pid * N + start + jj)
                if s > max1:
                    max2 = max1
                    max1 = s
                elif s > max2:
                    max2 = s
            acc += (max1 + max2)
        tl.store(group_scores_ptr + pid * group_count + g, acc)


@triton.jit
def topk_group_kernel(
    group_scores_ptr,  # *f32, [M, 8]
    group_idx_ptr,     # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,   # 4
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
):
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
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for j in range(N):
        val = tl.load(masked_scores_ptr + pid * stride_msm + j * stride_nsm)
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


@triton.jit
def gather_selected_scores_kernel(
    scores_ptr,          # *f32, [M, N]
    topk_idx_ptr,        # *int32, [M, 8]
    selected_ptr,        # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,
    stride_sm: tl.int32,
    stride_sn: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(topk_idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * N + idx)
        tl.store(selected_ptr + pid * K + t, val)


@triton.jit
def normalize_and_scale_kernel(
    selected_ptr,        # *f32, [M, 8]
    out_weight_ptr,      # *f32, [M, 8]
    M: tl.int32,
    scaling: tl.float32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    s = 0.0
    for t in range(8):
        s += tl.load(selected_ptr + pid * 8 + t)
    eps = 1e-20
    for t in range(8):
        val = tl.load(selected_ptr + pid * 8 + t)
        norm = val / (s + eps)
        tl.store(out_weight_ptr + pid * 8 + t, norm * scaling)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure tensors are on CUDA device and dtype float32
        device = hidden_states.device
        assert device.type == 'cuda', "This Triton implementation requires CUDA tensors."
        M, K = hidden_states.shape
        N, K_w = weight.shape
        assert K == K_w, f"Incompatible shapes: hidden_states [M, {K}] and weight [{N}, {K_w}]"
        assert N == 256, "This implementation expects N=num_experts=256"
        assert K == 128, "This implementation expects K=hidden_size=128"

        # 1) Compute logits = hidden_states @ weight.T
        logits = torch.empty((M, N), device=device, dtype=torch.float32)
        matmul_logits_kernel[(triton.cdiv(M, 128), triton.cdiv(N, 128))](
            hidden_states, weight, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        )

        # 2) scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), device=device, dtype=torch.float32)
        sigmoid_add_bias_kernel[(triton.cdiv(M, 128), triton.cdiv(N, 128))](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=128, BLOCK_N=128,
        )

        # 3) group_scores [M, 8] = sum of top-2 per group (from scores[M, 8, 32])
        group_scores = torch.empty((M, 8), device=device, dtype=torch.float32)
        group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_count=8, experts_per_group=32,
            num_warps=1,
        )

        # 4) top-4 group indices per token
        group_idx = torch.empty((M, 4), device=device, dtype=torch.int32)
        topk_group_kernel[(M,)](
            group_scores, group_idx,
            M, K=4,
            num_warps=1,
        )

        # 5) group_mask [M, 8]
        group_mask = torch.empty((M, 8), device=device, dtype=torch.float32)
        build_group_mask_kernel[(M,)](
            group_idx, group_mask,
            M, K=4, group_count=8,
            num_warps=1,
        )

        # 6) masked_scores [M, 256]: expand group_mask and set non-selected groups' 32 entries to -inf
        masked_scores = torch.empty((M, N), device=device, dtype=torch.float32)
        expand_and_set_ninf_kernel[(M,)](
            group_mask, masked_scores,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=1,
        )

        # 7) final top-8 expert indices per token from masked_scores
        topk_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        topk_experts_kernel[(M,)](
            masked_scores, topk_idx,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            K=8,
            num_warps=1,
        )

        # 8) gather original scores at those 8 positions
        selected_scores = torch.empty((M, 8), device=device, dtype=torch.float32)
        gather_selected_scores_kernel[(M,)](
            scores, topk_idx, selected_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            K=8,
            num_warps=1,
        )

        # 9) normalize and apply scaling factor
        topk_weight = torch.empty((M, 8), device=device, dtype=torch.float32)
        normalize_and_scale_kernel[(M,)](
            selected_scores, topk_weight,
            M, routed_scaling_factor,
            num_warps=1,
        )

        return topk_idx, topk_weight


# If you want to quickly test correctness locally:
# hidden_states = torch.randn(2048, 128, device='cuda', dtype=torch.float32)
# weight = torch.randn(256, 128, device='cuda', dtype=torch.float32)
# expert_bias = torch.randn(256, device='cuda', dtype=torch.float32)
# routed_scaling_factor = 1.0
# model_new = ModelNew()
# idx, weight = model_new(hidden_states, weight, expert_bias, routed_scaling_factor)
# print(idx.shape, weight.shape)  # should be [2048, 8] for both


def run(*args):
    return ModelNew()(*args)
