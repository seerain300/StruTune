import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: Compute logits = hidden_states @ weight^T
# hidden_states: [M, K] where M = num_tokens, K = hidden_dim
# weight: [N, K] where N = num_experts (256 here), we will load weight[专家, hidden] in B block
# Output: logits: [M, N] (M and N are runtime, but we use 256 for N; kernel handles any N via grid)
@triton.jit
def linear_kernel(
    A_ptr,      # *fp32, hidden_states: [M, K]
    W_ptr,      # *fp32, weight: [N, K] where N=num_experts, K=hidden_dim
    OUT_ptr,    # *fp32, output logits: [M, N]
    M: tl.constexpr,  # num_tokens
    N: tl.constexpr,  # num_experts (256)
    K: tl.constexpr,  # hidden_dim
    stride_am, stride_ak,   # strides for A
    stride_wn, stride_wk,   # strides for W (weight)
    stride_om, stride_on,   # strides for OUT
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: pid_m over M tiles, pid_n over N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator for tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k

        # A tile: [BLOCK_M, BLOCK_K], load from A[m, k]
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N], load from W[n, k] where W is [N, K], so we access weight[offs_n, k_ids]
        W_tile_ptr = W_ptr + (offs_n[None, :] * stride_wn + k_ids[:, None] * stride_wk)
        W_mask = (offs_n[None, :] < N) & (k_ids[:, None] < K)
        W_tile = tl.load(W_tile_ptr, mask=W_mask, other=0.0)

        # Accumulate: acc += A_tile @ W_tile
        acc += tl.dot(A_tile, W_tile)

    # Store results to OUT[m, n]
    OUT_tile_ptr = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    OUT_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(OUT_tile_ptr, acc, mask=OUT_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        hidden_states: [num_tokens, hidden_dim], CUDA, float16/float32
        weight: [num_experts, hidden_dim], CUDA, float16/float32, num_experts=256
        expert_bias: [num_experts], CUDA, float32
        routed_scaling_factor: float
        Returns:
        - topk_idx: [num_tokens, 8], long
        - topk_weight: [num_tokens, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors"
        assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_dim]"
        assert weight.dim() == 2, "weight must be [num_experts, hidden_dim]"
        assert expert_bias.dim() == 1 and expert_bias.shape[0] == weight.shape[0], "expert_bias must be [num_experts]"
        num_tokens = hidden_states.shape[0]
        num_experts = weight.shape[0]
        hidden_dim = weight.shape[1]
        assert num_experts == 256, "This implementation assumes num_experts=256"
        # groups
        n_group = 8
        experts_per_group = num_experts // n_group  # 32

        # Ensure contiguous
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()
        expert_bias = expert_bias.contiguous()

        # We will compute logits in float32
        # hidden_states: [M, K], weight: [N, K], output: [M, N]
        # Triton kernel needs A[M,K] and W[N,K] (which we can read directly from weight).
        M = num_tokens
        N = num_experts
        K = hidden_dim

        # Allocate output logits [M, N], float32
        logits = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)

        # Strides
        stride_am = hidden_states.stride(0)
        stride_ak = hidden_states.stride(1)
        stride_wn = weight.stride(0)  # row-major: stride for expert dimension
        stride_wk = weight.stride(1)  # stride for hidden_dim
        stride_om = logits.stride(0)
        stride_on = logits.stride(1)

        # Tiling configuration
        BLOCK_M = 128
        BLOCK_N = 64  # 64 * 4 = 256, grid second dim will be 4
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # e.g., (16, 4) for 6144 tokens

        # Launch Triton kernel
        linear_kernel[grid](
            hidden_states, weight, logits,
            M, N, K,
            stride_am, stride_ak,
            stride_wn, stride_wk,
            stride_om, stride_on,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Compute scores: sigmoid(logits) + expert_bias (broadcast bias over tokens)
        scores = torch.sigmoid(logits)  # [num_tokens, 256], float32
        scores_for_routing = scores + expert_bias.to(torch.float32)  # [num_tokens, 256]

        # Group 256 experts into 8 groups of 32: reshape to [num_tokens, 8, 32]
        group_scores_reshaped = scores_for_routing.view(M, n_group, experts_per_group)  # [num_tokens, 8, 32]
        # Top-2 within each group, sum to get [num_tokens, 8]
        top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)
        group_scores = top2_vals.sum(dim=-1)  # [num_tokens, 8]
        _, group_idx = torch.topk(group_scores, k=4, dim=-1, sorted=False)  # select 4 groups per token

        # Build group mask [num_tokens, 8]
        group_mask = torch.zeros((M, n_group), dtype=torch.float32, device=scores_for_routing.device)
        group_mask.scatter_(1, group_idx, 1.0)

        # Expand mask to expert level [num_tokens, 256]
        score_mask = group_mask.unsqueeze(-1).expand(M, n_group, experts_per_group).reshape(M, N)

        # Masked scores: set non-selected to very negative
        neg_inf = torch.finfo(torch.float32).min
        masked_scores = scores_for_routing.masked_fill(score_mask == 0, neg_inf)

        # Select top-8 experts from masked scores
        _, topk_idx = torch.topk(masked_scores, k=8, dim=-1, sorted=False)  # [num_tokens, 8], long

        # Gather selected logits (no bias)
        selected_logits = torch.gather(logits, dim=1, index=topk_idx)  # [num_tokens, 8], float32
        # Normalize by sum
        topk_weight = selected_logits / (selected_logits.sum(dim=-1, keepdim=True) + 1e-20)
        # Apply scaling factor
        topk_weight = topk_weight * routed_scaling_factor

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
