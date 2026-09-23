import torch
import triton
import triton.language as tl


# Triton reduction: per-row squared norm of A[M, N] -> out[i] = sum_j A[i, j]^2
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 256}, num_warps=8),
        triton.Config({"BLOCK_K": 512}, num_warps=8),
        triton.Config({"BLOCK_K": 1024}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def _row_sqnorm(
    A_ptr,  # *fp32 (we pass grad_output.float())
    out_ptr,  # *fp32
    M, N,
    stride_am, stride_an,
    stride_out,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    acc = 0.0
    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + row * stride_am + offs_k * stride_an, mask=offs_k < N, other=0.0).to(tl.float32)
        acc += tl.sum(a * a, axis=0)
    tl.store(out_ptr + row * stride_out, acc)


# Triton scatter-add: add grad_topk_weights[row, k] into grad_scores[row, indices[row, k]]
@triton.jit
def _scatter_add_topk(
    grad_topk_ptr,     # *fp32 [B, K]
    indices_ptr,       # *int32 [B, K]
    grad_scores_ptr,   # *fp32 [B, E]
    B, E, K,
    stride_gt0, stride_gt1,
    stride_idx0, stride_idx1,
    stride_gs0, stride_gs1,
):
    row = tl.program_id(0)
    for k in range(0, K):
        val = tl.load(grad_topk_ptr + row * stride_gt0 + k * stride_gt1)
        idx = tl.load(indices_ptr + row * stride_idx0 + k * stride_idx1)
        tl.atomic_add(grad_scores_ptr + row * stride_gs0 + idx * stride_gs1, val)


# Triton matmul: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16 (fp32 accumulation)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8),
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, b)

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# Triton elementwise: shared_activated = silu(gate) * up (silu(x) = x * sigmoid(x))
@triton.jit
def _silu_mul_elementwise(
    gate_ptr, up_ptr, activated_ptr,
    SIZE,
    stride_g, stride_u, stride_a,
):
    idx = tl.program_id(0)
    g = tl.load(gate_ptr + idx * stride_g).to(tl.float32)
    u = tl.load(up_ptr + idx * stride_u).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    y = (g * sig) * u
    tl.store(activated_ptr + idx * stride_a, y.to(tl.bfloat16))


# Triton elementwise: grad_scores *= score_mask
@triton.jit
def _elem_mul_score_mask(
    grad_scores_ptr, score_mask_ptr, out_ptr,
    B, E,
    stride_gs0, stride_gs1, stride_sm0, stride_sm1, stride_out0, stride_out1,
):
    row = tl.program_id(0)
    for e in range(0, E):
        gs = tl.load(grad_scores_ptr + row * stride_gs0 + e * stride_gs1).to(tl.float32)
        sm = tl.load(score_mask_ptr + row * stride_sm0 + e * stride_sm1).to(tl.float32)
        tl.store(out_ptr + row * stride_out0 + e * stride_out1, (gs * sm).to(tl.bfloat16))


# Triton elementwise: grad_router_logits = grad_scores * scores * (1 - scores)
@triton.jit
def _grad_sigmoid_elementwise(
    grad_scores_ptr, scores_ptr, out_ptr,
    SIZE,
    stride_gs, stride_sc, stride_out,
):
    idx = tl.program_id(0)
    gs = tl.load(grad_scores_ptr + idx * stride_gs).to(tl.float32)
    sc = tl.load(scores_ptr + idx * stride_sc).to(tl.float32)
    deriv = sc * (1.0 - sc)
    tl.store(out_ptr + idx * stride_out, (gs * deriv).to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,   # [B, H], bf16
        hidden_states: torch.Tensor, # [B, H], bf16
        router_weight: torch.Tensor, # [E, H], bf16
        e_score_correction_bias: torch.Tensor, # [E], fp32
        router_logits: torch.Tensor, # [B, E], fp32
        scores: torch.Tensor,        # [B, E], fp32
        topk_indices: torch.Tensor,  # [B, K], long
        topk_weights: torch.Tensor,  # [B, K], fp32
        score_mask: torch.Tensor,    # [B, E], fp32
        shared_expert_gate_weight: torch.Tensor, # [H', H], bf16
        shared_expert_up_weight: torch.Tensor,   # [H', H], bf16
        shared_expert_down_weight: torch.Tensor, # [H, H'], bf16
        shared_gate_output: torch.Tensor,        # [B, H'], bf16
        shared_up_output: torch.Tensor,          # [B, H'], bf16
        shared_activated: torch.Tensor,          # [B, H'], bf16 (not used in grad)
    ):
        B, H = grad_output.shape
        E, Hw = router_weight.shape
        B2, E2 = scores.shape
        assert E == E2
        Hprime = shared_expert_gate_weight.shape[0]
        K = topk_indices.shape[1]

        # 1) Row-wise squared norm of grad_output per token (fp32 out)
        grad_norm_sq = torch.empty((B,), device=hidden_states.device, dtype=torch.float32)
        _row_sqnorm[(B,)](
            grad_output.to(torch.float32),
            grad_norm_sq,
            B, H,
            grad_output.stride(0), grad_output.stride(1),
            1,
        )

        # 2) Scatter-add to build grad_scores[B, E]
        grad_topk_norm = (grad_norm_sq.view(B, 1) / float(K)).expand(B, K).contiguous()  # [B, K], fp32
        grad_scores = torch.empty((B, E), device=hidden_states.device, dtype=torch.float32)
        _scatter_add_topk[(B,)](
            grad_topk_norm, topk_indices.to(torch.int32), grad_scores, B, E, K,
            grad_topk_norm.stride(0), grad_topk_norm.stride(1),
            topk_indices.stride(0), topk_indices.stride(1),
            grad_scores.stride(0), grad_scores.stride(1),
        )

        # 3) Apply score_mask
        grad_scores_masked = torch.empty_like(grad_scores, dtype=torch.float32)
        _elem_mul_score_mask[(B,)](
            grad_scores, score_mask, grad_scores_masked,
            B, E,
            grad_scores.stride(0), grad_scores.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            grad_scores_masked.stride(0), grad_scores_masked.stride(1),
        )

        # 4) Compute grad_router_logits = grad_scores_masked * scores * (1 - scores)
        grad_router_logits = torch.empty((B, E), device=hidden_states.device, dtype=torch.bfloat16)
        _grad_sigmoid_elementwise[(B * E,)](
            grad_scores_masked, scores, grad_router_logits,
            B * E,
            grad_scores_masked.stride(0), scores.stride(0), grad_router_logits.stride(0),
        )

        # 5) Route weight gradient: C = grad_router_logits.T @ hidden_states -> [E, H], bf16


def run(*args):
    return ModelNew()(*args)
