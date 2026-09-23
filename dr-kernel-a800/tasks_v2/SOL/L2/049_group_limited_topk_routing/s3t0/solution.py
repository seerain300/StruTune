import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # pointer to hidden [M, K]
    B_ptr,  # pointer to weight.T [K, N]  (note: weight is [N, K], so B is transposed)
    C_ptr,  # pointer to output [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)  # iterate over rows (tokens)
    pid_n = tl.program_id(1)  # iterate over column blocks of N (experts)

    # derive offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # pointers for A (hidden) and B (weight.T)
    # A: [M, K] -> row-major
    # B: [K, N] -> row-major on K,N
    # C: [M, N] -> row-major
    # We'll loop over K and accumulate
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        k += BLOCK_K

    # write back to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_linear(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute scores = hidden @ weight.T where:
      - hidden: [M, K], float16/float32
      - weight: [N, K], float16/float32 (same as F.linear weight)
      Output: [M, N], float32
    """
    assert hidden.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
    assert hidden.dim() == 2, "hidden must be 2D [num_tokens, K]"
    assert weight.dim() == 2, "weight must be 2D [num_experts, K]"

    M, K = hidden.shape
    N = weight.shape[0]  # num_experts

    # Make sure inputs are contiguous and cast to float32 for accumulation
    hidden_c = hidden.contiguous()
    # weight: [N, K], we need B = weight.T [K, N]
    weight_t = weight.transpose(0, 1).contiguous()  # shape [K, N]
    out = torch.empty((M, N), dtype=torch.float32, device=hidden.device)

    # Choose tiling parameters
    BLOCK_M = 1
    BLOCK_N = 128
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _matmul_kernel[grid](
        hidden_c, weight_t, out,
        M, N, K,
        hidden_c.stride(0), hidden_c.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-optimized version of the original Model.run:
        - Uses Triton for the linear projection (hidden @ weight.T).
        - The rest (sigmoid, expert bias, group masking, final top-k) is done in torch.
        """
        # Ensure inputs are on CUDA for Triton
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors for Triton kernels"

        # Compute scores via Triton matmul: scores = hidden @ weight.T
        # hidden_states: [num_tokens, 768], weight: [256, 768]
        scores = triton_linear(hidden_states.to(torch.float32), weight.to(torch.float32))  # [num_tokens, 256]

        # Apply sigmoid activation
        scores = torch.sigmoid(scores)  # [num_tokens, 256]

        # Add learned expert bias (broadcast over tokens)
        scores = scores + expert_bias.to(torch.float32).unsqueeze(0)  # [num_tokens, 256]

        # Constants
        num_experts = 256
        n_group = 8
        topk_group = 4
        experts_per_group = num_experts // n_group  # 32

        num_tokens = scores.shape[0]

        # Reshape into groups: [num_tokens, 8, 32]
        group_scores = scores.view(num_tokens, n_group, experts_per_group)

        # Top-2 per group and sum (unsorted=False; we only need values, not indices)
        # Note: PyTorch topk returns values sorted by default; here we only use sums, sorted order is irrelevant.
        top2_vals, _ = torch.topk(group_scores, k=2, dim=-1, largest=True, sorted=False)
        group_scores = top2_vals.sum(dim=-1)  # [num_tokens, 8]

        # Top-4 groups per token
        _, group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)  # [num_tokens, 4], int64

        # Group mask for each token
        group_mask = torch.zeros_like(group_scores)  # [num_tokens, 8]
        group_mask.scatter_(1, group_idx, 1.0)      # set selected groups to 1

        # Expand to expert level: [num_tokens, 256]
        score_mask = group_mask.unsqueeze(-1).expand(num_tokens, n_group, experts_per_group).reshape(num_tokens, num_experts)

        # Mask out non-selected groups by setting scores to -inf
        neg_inf = torch.finfo(torch.float32).min
        masked_scores = scores.masked_fill(score_mask == 0, neg_inf)  # [num_tokens, 256]

        # Final top-8 experts
        _, topk_idx = torch.topk(masked_scores, k=8, dim=-1, sorted=False)  # [num_tokens, 8], int64

        # Gather selected expert scores (use original 'scores' which includes sigmoid and bias)
        selected_scores = torch.gather(scores, dim=1, index=topk_idx)  # [num_tokens, 8]

        # Normalize routing weights
        topk_weight = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)  # [num_tokens, 8]

        # Apply routing scaling factor
        topk_weight = topk_weight * routed_scaling_factor

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
