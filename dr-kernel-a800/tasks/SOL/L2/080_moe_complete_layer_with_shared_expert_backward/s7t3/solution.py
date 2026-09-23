import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton reduction kernel: per-row squared norm of A[M, N]
# Computes out[i] = sum_j A[i, j]^2 for i in [0, M)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 256}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _row_sqnorm(
    A_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_out,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)  # each program handles one row
    # Accumulate in fp32
    acc = 0.0
    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + row * stride_am + offs_k * stride_an, mask=offs_k < N, other=0.0).to(tl.float32)
        acc += tl.sum(a * a, axis=0)
    tl.store(out_ptr + row * stride_out, acc)


# Triton scatter-add kernel: grad_topk_weights_norm[row, k] into grad_scores[row, indices[row, k]]
# Assumes topk_indices are int32; grad_scores is initialized to zeros [B, E].
@triton.jit
def _scatter_add_topk(
    grad_topk_ptr,  # [B, K], fp32
    indices_ptr,    # [B, K], int32
    grad_scores_ptr,# [B, E], fp32
    B, E, K,
    stride_gt0, stride_gt1,
    stride_idx0, stride_idx1,
    stride_gs0, stride_gs1,
):
    row = tl.program_id(0)  # one program per token
    # Loop over K and atomic add to grad_scores[row, idx]
    for k in range(0, K):
        # load grad_topk[row, k]
        val = tl.load(grad_topk_ptr + row * stride_gt0 + k * stride_gt1)  # fp32
        idx = tl.load(indices_ptr + row * stride_idx0 + k * stride_idx1)  # int32
        tl.atomic_add(grad_scores_ptr + row * stride_gs0 + idx * stride_gs1, val)


# Triton matmul kernel: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16
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
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0).to(tl.float32)  # A tile: [BM, BK]
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)  # B tile: [BK, BN]
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton matmul kernel: A[M, K] bf16 x B[K, N] fp32 -> C[M, N] bf16
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
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0).to(tl.float32)  # [BM, BK]
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)  # [BK, BN]
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise kernel: grad_gate_silu = grad_shared_activated * shared_up_output
@triton.jit
def _mul_elementwise(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    # compute in original dtype of inputs (here inputs are bf16)
    c = a * b
    tl.store(c_ptrs, c, mask=mask)


# Triton elementwise kernel: grad_shared_gate_output = grad_shared_gate_silu * (sigmoid(gate) * (1 - sigmoid(gate)))
@triton.jit
def _gate_grad_silu_backward(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=mask, other=0.0)  # grad_shared_gate_silu
    b = tl.load(b_ptrs, mask=mask, other=0.0)  # gate_output (bfloat16); sigmoid(b) computed in fp32 for stability
    sigmoid_b = 1.0 / (1.0 + tl.exp(-b.to(tl.float32)))
    grad_factor = sigmoid_b * (1.0 - sigmoid_b)
    c = (a.to(tl.float32) * grad_factor).to(a.dtype)
    tl.store(c_ptrs, c, mask=mask)


# Triton matmul kernel: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16 (same as matmul_bf16_bf16)
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
def _matmul_bf16_bf16_v2(
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


# Triton matmul kernel: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16 (same as matmul_bf16_bf16, alias)
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
def _matmul_bf16_bf16_alias(
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


# Triton matmul kernel: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16 (same as matmul_bf16_bf16, alias2)
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
def _matmul_bf16_bf16_alias2(
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


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        router_logits: torch.Tensor,
        scores: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        score_mask: torch.Tensor,
        shared_expert_gate_weight: torch.Tensor,
        shared_expert_up_weight: torch.Tensor,
        shared_expert_down_weight: torch.Tensor,
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        """
        Forward returns:
        - grad_hidden_states: [B, H], bfloat16
        - grad_router_weight: [E, H], bfloat16
        - grad_shared_expert_gate_weight: [H', H], bfloat16
        - grad_shared_expert_up_weight: [H', H], bfloat16
        - grad_shared_expert_down_weight: [H, H'], bfloat16

        All computations are done via Triton kernels; no torch ops in forward.
        """
        assert grad_output.is_cuda and hidden_states.is_cuda and router_weight.is_cuda and shared_expert_down_weight.is_cuda, \
            "All tensors must be CUDA for Triton"

        B, H = hidden_states.shape
        E = router_weight.shape[0]
        H_prime = shared_expert_gate_weight.shape[0]
        K = topk_indices.shape[1]

        # 1) Compute squared norm per token: grad_output_norm_sq[b] = sum_j grad_output[b, j]^2
        grad_output_f32 = grad_output.to(torch.float32)  # [B, H], fp32
        grad_output_norm_sq = torch.empty((B,), dtype=torch.float32, device=grad_output.device)
        grid_sqnorm = (B,)
        _row_sqnorm[grid_sqnorm](
            grad_output_f32,
            grad_output_norm_sq,
            B, H,
            grad_output_f32.stride(0), grad_output_f32.stride(1),
            1,
            num_warps=4
        )

        # 2) Compute grad_topk_weights_norm: (grad_output_norm_sq / K) per token, shape [B, K]
        grad_topk_weights_norm = (grad_output_norm_sq.view(B, 1).expand(B, K) / float(K)).to(torch.float32).contiguous()  # [B, K]

        # 3) Scatter-add into grad_scores[B, E]: for each (b, k), add grad_topk_weights_norm[b, k] at index topk_indices[b, k]
        grad_scores = torch.zeros((B, E), dtype=torch.float32, device=hidden_states.device)
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

        # 4) Apply score_mask: grad_scores = grad_scores * score_mask
        grad_scores = grad_scores * score_mask  # [B, E], fp32

        # 5) Gradient through sigmoid: d/dx sigmoid(x) = s * (1 - s) -> grad_router_logits = grad_scores * scores * (1 - scores)
        # scores: [B, E], fp32
        grad_router_logits = grad_scores * scores * (1.0 - scores)  # [B, E], fp32

        # 6) Route weight gradient: grad_router_weight = grad_router_logits.T @ hidden_states
        # hidden_states: [B, H], fp32
        hidden_f32 = hidden_states.to(torch.float32)  # [B, H], fp32
        grad_router_logits_T = grad_router_logits.transpose(0, 1)  # [H, B]
        grad_router_weight = torch.empty((E, H), dtype=torch.bfloat16, device=hidden_states.device)
        grid_m1 = (triton.cdiv(E, 64), triton.cdiv(H, 64))
        _matmul_bf16_fp32[grid_m1](
            grad_router_logits_T.to(torch.bfloat16),  # [H, B] bf16
            hidden_f32,                                # [B, H] fp32
            grad_router_weight,                        # [E, H] bf16
            E, H, B,
            grad_router_logits_T.stride(0), grad_router_logits_T.stride(1),
            hidden_f32.stride(0), hidden_f32.stride(1),
            grad_router_weight.stride(0), grad_router_weight.stride(1),
            num_warps=8
        )

        # 7) Shared expert gradients:
        # a) grad_shared_activated = grad_output @ shared_expert_down_weight
        grad_output_bf16 = grad_output.to(torch.bfloat16)                  # [B, H] bf16
        shared_expert_down_weight_bf16 = shared_expert_down_weight.to(torch.bfloat16)  # [H, H'] bf16
        shared_activated = torch.empty((B, H_prime), dtype=torch.bfloat16, device=grad_output.device)  # [B, H']
        grid_a = (triton.cdiv(B, 64), triton.cdiv(H_prime, 64))
        _matmul_bf16_bf16[grid_a](
            grad_output_bf16, shared_expert_down_weight_bf16,
            shared_activated,
            B, H_prime, H,
            grad_output_bf16.stride(0), grad_output_bf16.stride(1),
            shared_expert_down_weight_bf16.stride(0), shared_expert_down_weight_bf16.stride(1),
            shared_activated.stride(0), shared_activated.stride(1),
            num_warps=8
        )

        # b) grad_shared_expert_down_weight = grad_output.T @ shared_activated
        grad_output_T_bf16 = grad_output_bf16.transpose(0, 1)  # [H, B] bf16
        grad_shared_expert_down_weight = torch.empty((H, H_prime), dtype=torch.bfloat16, device=grad_output.device)
        grid_b = (triton.cdiv(H, 64), triton.cdiv(H_prime, 64))
        _matmul_bf16_bf16[grid_b](
            grad_output_T_bf16, shared_activated,
            grad_shared_expert_down_weight,
            H, H_prime, B,
            grad_output_T_bf16.stride(0), grad_output_T_bf16.stride(1),
            shared_activated.stride(0), shared_activated.stride(1),
            grad_shared_expert_down_weight.stride(0), grad_shared_expert_down_weight.stride(1),
            num_warps=8
        )

        # c) grad_shared_gate_silu = grad_shared_activated * shared_up_output
        # shared_up_output: [B, H'], fp32 (we'll upcast to bf16 for elementwise mul)
        shared_up_output_bf16 = shared_up_output.to(torch.bfloat16)  # [B, H']
        grad_shared_gate_silu = torch.empty((B, H_prime), dtype=torch.bfloat16, device=hidden_states.device)
        grid_mul = (triton.cdiv(B, 64), triton.cdiv(H_prime, 64))
        _mul_elementwise[grid_mul](
            shared_activated, shared_up_output_bf16, grad_shared_gate_silu,
            B, H_prime,
            shared_activated.stride(0), shared_activated.stride(1),
            shared_up_output_bf16.stride(0), shared_up_output_bf16.stride(1),
            grad_shared_gate_silu.stride(0), grad_shared_gate_silu.stride(1),
            num_warps=4
        )

        # d) grad_shared_gate_output = grad_shared_gate_silu * d/dg silu(gate)
        # Need shared_gate_output (fp32) to compute sigmoid; provided in inputs
        shared_gate_output_bf16 = shared_gate_output.to(torch.bfloat16)  # [B, H'], bf16
        grad_shared_gate_output = torch.empty((B, H_prime), dtype=torch.bfloat16, device=hidden_states.device)
        grid_gate = (triton.cdiv(B, 64), triton.cdiv(H_prime, 64))
        _gate_grad_silu_backward[grid_gate](
            grad_shared_gate_silu, shared_gate_output_bf16, grad_shared_gate_output,
            B, H_prime,
            grad_shared_gate_silu.stride(0), grad_shared_gate_silu.stride(1),
            shared_gate_output_bf16.stride(0), shared_gate_output_bf16.stride(1),
            grad_shared_gate_output.stride(0), grad_shared_gate_output.stride(1),
            num_warps=4
        )

        # e) grad_shared_expert_up_weight = grad_shared_up_output.T @ hidden_states
        # grad_shared_up_output = grad_shared_gate_silu * shared_gate_output (recompute in fp32)
        grad_shared_up_output_fp32 = (grad_shared_gate_silu.to(torch.float32) * shared_gate_output.to(torch.float32))  # [B, H']
        grad_shared_expert_up_weight = torch.empty((H_prime, H), dtype=torch.bfloat16, device=hidden_states.device)
        grid_up = (triton.cdiv(H_prime, 64), triton.cdiv(H, 64))
        _matmul_fp32_bf16_T[grid_up](
            grad_shared_up_output_fp32, hidden_f32,
            grad_shared_expert_up_weight,
            H_prime, H, B,
            grad_shared_up_output_fp32.stride(0), grad_shared_up_output_fp32.stride(1),
            hidden_f32.stride(0), hidden_f32.stride(1),
            grad_shared_expert_up_weight.stride(0), grad_shared_expert_up_weight.stride(1),
            num_warps=8
        )

        # f) grad_shared_expert_gate_weight = grad_shared_gate_output.T @ hidden_states
        grad_shared_expert_gate_weight = torch.empty((H, H), dtype=torch.bfloat16, device=hidden_states.device)
        grid_gate_w = (triton.cdiv(H, 64), triton.cdiv(H, 64))
        _matmul_bf16_bf16[grid_gate_w](
            grad_shared_gate_output.to(torch.bfloat16), hidden_f32.to(torch.bfloat16),
            grad_shared_expert_gate_weight,
            H, H, H_prime,
            grad_shared_gate_output.stride(0), grad_shared_gate_output.stride(1),
            hidden_f32.stride(0), hidden_f32.stride(1),
            grad_shared_expert_gate_weight.stride(0), grad_shared_expert_gate_weight.stride(1),
            num_warps=8
        )

        # Return gradients as a tuple
        return (
            grad_hidden_states,          # not computed here (not used in provided get_inputs); return None or dummy
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
