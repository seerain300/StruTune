import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton matmul: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16, fp32 accumulation
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_bf16_bf16(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton matmul: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] fp32 (accumulate in fp32)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_bf16_fp32(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton scatter_add: given grad_topk [B, K] fp32 and indices [B, K] int32, accumulate into grad_scores [B, E] fp32
@triton.jit
def _scatter_add_topk(
    grad_topk_ptr,            # [B, K], fp32
    indices_ptr,              # [B, K], int32
    grad_scores_ptr,          # [B, E], fp32
    B, E, K,
    stride_gm, stride_gk,
    stride_im, stride_ik,
    stride_sm, stride_se,
):
    row = tl.program_id(0)
    if row >= B:
        return
    k = 0
    while k < K:
        idx = tl.load(indices_ptr + row * stride_im + k * stride_ik).to(tl.int32)
        val = tl.load(grad_topk_ptr + row * stride_gm + k * stride_gk)  # fp32
        tl.atomic_add(grad_scores_ptr + row * stride_sm + idx * stride_se, val)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,            # [B, H], bfloat16
        hidden_states: torch.Tensor,          # [B, H], bfloat16
        router_weight: torch.Tensor,          # [E, H], bfloat16
        e_score_correction_bias: torch.Tensor,# [E], float32
        router_logits: torch.Tensor,          # [B, E], float32
        scores: torch.Tensor,                 # [B, E], float32
        topk_indices: torch.Tensor,           # [B, K], long
        topk_weights: torch.Tensor,           # [B, K], float32
        score_mask: torch.Tensor,             # [B, E], float32
        shared_expert_gate_weight: torch.Tensor,  # [H', H], bfloat16 (random)
        shared_expert_up_weight: torch.Tensor,    # [H', H], bfloat16 (random)
        shared_expert_down_weight: torch.Tensor,  # [H, H'], bfloat16 (random)
        shared_gate_output: torch.Tensor,         # [B, H], float32
        shared_up_output: torch.Tensor,           # [B, H], float32
        shared_activated: torch.Tensor,           # [B, H], float32
    ):
        """
        Triton-optimized forward that returns gradients w.r.t:
        - hidden_states
        - router_weight
        - shared_expert_gate_weight
        - shared_expert_up_weight
        - shared_expert_down_weight
        Notes:
        - Triton performs heavy matmuls for down_weight and the routed contribution to hidden.
        - Other gradients use PyTorch for correctness since required intermediates are not provided.
        """
        B, H = hidden_states.shape
        E = router_weight.shape[0]
        H_prime = shared_expert_gate_weight.shape[0]
        K = topk_indices.shape[1]

        # 1) Route-related gradients via Triton
        # Compute grad_topk_weights_norm: ||grad_output||^2 per token, divide by K
        grad_output_f32 = grad_output.to(torch.float32)  # [B, H]
        grad_norm_sq = (grad_output_f32 * grad_output_f32).sum(dim=-1, keepdim=True)  # [B, 1]
        grad_topk_weights_norm = (grad_norm_sq.expand(B, K) / float(K)).contiguous().to(torch.float32)  # [B, K], fp32

        # Allocate grad_scores [B, E], fp32
        grad_scores = torch.zeros((B, E), dtype=torch.float32, device=hidden_states.device)

        # Launch scatter_add kernel: one program per token
        grid_scatter = (B,)
        _scatter_add_topk[grid_scatter](
            grad_topk_weights_norm,                    # [B, K], fp32
            topk_indices.to(torch.int32),             # [B, K], int32
            grad_scores,                              # [B, E], fp32
            B, E, K,
            grad_topk_weights_norm.stride(0), grad_topk_weights_norm.stride(1),
            topk_indices.stride(0), topk_indices.stride(1),
            grad_scores.stride(0), grad_scores.stride(1),
            num_warps=4
        )

        # Apply score_mask
        grad_scores = grad_scores * score_mask  # [B, E], fp32

        # Gradient through sigmoid: d/dx sigmoid(x) = s * (1 - s)
        grad_router_logits = grad_scores * scores * (1.0 - scores)  # [B, E], fp32

        # 2) Route weight gradient (compute with PyTorch for simplicity): grad_router_weight = grad_router_logits.T @ hidden_states
        # hidden_states: [B, H], grad_router_logits: [B, E]
        hidden_for_b = hidden_states.to(torch.float32)  # [B, H]
        grad_router_logits_T


def run(*args):
    return ModelNew()(*args)
