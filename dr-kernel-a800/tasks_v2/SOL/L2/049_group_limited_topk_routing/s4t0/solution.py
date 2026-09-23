import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_no_bias_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [K, N], float32
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Tile pointers
    A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_tile_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_tile_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(B_tile_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A_tile_ptrs += BLOCK_K * stride_ak
        B_tile_ptrs += BLOCK_K * stride_bk

    # Write back to C
    C_tile_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_tile_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-optimized version:
        - Compute logits = hidden_states @ weight using Triton kernel (FP32, no bias).
        - Apply sigmoid and add expert_bias in PyTorch (same as original).
        - Perform the group-limited top-k expert routing exactly as the original.
        Returns (topk_idx [num_tokens, 8], topk_weight [num_tokens, 8]).
        """
        # Ensure CUDA and dtype for Triton
        device = torch.device("cuda")
        if not hidden_states.is_cuda:
            hidden_states = hidden_states.to(device)
        if not weight.is_cuda:
            weight = weight.to(device)
        if not expert_bias.is_cuda:
            expert_bias = expert_bias.to(device)

        # Cast to float32 and make contiguous
        A = hidden_states.contiguous().to(torch.float32)   # [num_tokens, hidden_dim]
        B = weight.contiguous().to(torch.float32)          # [hidden_dim, 256] -> we pass as [K, N]
        M = A.shape[0]
        K = A.shape[1]
        N = B.shape[1]  # should be 256
        C = torch.empty((M, N), dtype=torch.float32, device=device)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch Triton matmul (no bias)
        # Choose tile sizes (generic defaults)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_no_bias_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        logits = C  # [num_tokens, 256], FP32

        # Apply sigmoid and expert bias (same as original)
        scores = torch.sigmoid(logits)                        # [num_tokens, 256]
        scores = scores + expert_bias.to(torch.float32).to(device)  # [num_tokens, 256]

        # Constants for routing
        num_experts = 256
        n_group = 8
        topk_group = 4
        top_k = 8

        assert num_experts == 256, "This implementation assumes 256 experts."
        assert n_group == 8, "This implementation assumes 8 groups."
        assert topk_group == 4, "This implementation assumes selecting top-4 groups."
        assert top_k == 8, "This implementation assumes final top-8 experts."

        groups_per_expert = num_experts // n_group  # 32
        num_tokens = logits.shape[0]

        # Reshape scores to [num_tokens, 8, 32]
        group_scores_reshaped = scores.view(num_tokens, n_group, groups_per_expert)
        # Top-2 per group, sum -> [num_tokens, 8]
        top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)
        group_scores = top2_vals.sum(dim=-1)

        # Select top-4 groups per token -> [num_tokens, 4]
        _, group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)

        # Create mask of selected groups: 1.0 for selected groups, 0 elsewhere -> [num_tokens, 8]
        group_mask = torch.zeros((num_tokens, n_group), dtype=torch.float32, device=device)
        group_mask.scatter_(1, group_idx, 1.0)

        # Expand mask to the expert level: [num_tokens, 8, 32] -> [num_tokens, 256]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, n_group, groups_per_expert)
            .reshape(num_tokens, num_experts)
        )

        # Mask out non-selected groups by setting their scores to -inf
        neg_inf = torch.finfo(torch.float32).min
        masked_scores = scores.masked_fill(score_mask == 0, neg_inf)

        # Select top-8 experts (indices) per token -> [num_tokens, 8], Long
        _, topk_idx = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)

        # Gather selected scores (from original scores, after sigmoid + expert_bias)
        selected_scores = torch.gather(scores, dim=1, index=topk_idx)  # [num_tokens, 8]

        # Normalize weights: L1 normalize and apply epsilon
        topk_weight = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)

        # Apply routing scaling factor
        topk_weight = topk_weight * routed_scaling_factor

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
