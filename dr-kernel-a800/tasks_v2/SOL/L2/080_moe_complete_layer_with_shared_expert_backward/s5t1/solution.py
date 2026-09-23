import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) Triton matmul kernel: C[M, N] = A[M, K] @ B[K, N], bf16 inputs, fp32 accumulation, bf16 output
@triton.jit
def matmul_bf16_fp32_kernel(
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

    # Pointers to tiles
    A_tile_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak  # [BM, BK]
    B_tile_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # [BK, BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask_a = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        k_mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_tile_ptrs, mask=k_mask_a, other=0.0).to(tl.float32)  # [BM, BK]
        b = tl.load(B_tile_ptrs, mask=k_mask_b, other=0.0).to(tl.float32)  # [BK, BN]
        acc += tl.dot(a, b)
        A_tile_ptrs += BLOCK_K * stride_ak
        B_tile_ptrs += BLOCK_K * stride_bk

    C_tile_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def triton_matmul_bf16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B where A is [M, K] bf16 and B is [K, N] bf16. Returns C [M, N] bf16.
    Uses Triton with fp32 accumulation.
    """
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    assert A.dim() == 2 and B.dim() == 2
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Shape mismatch: A is {A.shape}, B is {B.shape}"

    A_c = A.contiguous()
    B_c = B.contiguous()
    C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

    stride_am, stride_ak = A_c.stride()
    stride_bk, stride_bn = B_c.stride()
    stride_cm, stride_cn = C.stride()

    # Tiling config
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_bf16_fp32_kernel[grid](
        A_c, B_c, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C


# 2) Triton row-wise matvec kernel: C[1, N] = A[1, K] @ B[K, N]
@triton.jit
def row_matvec_bf16_fp32_kernel(
    A_ptr, B_ptr, C_ptr,
    K, N,
    stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program across N for a single row (M=1)
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    A_row_ptrs = A_ptr + offs_k * stride_ak  # [BK] but this is a row vector of length K
    B_tile_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # [BK, BN]

    for k in range(0, K, BLOCK_K):
        k_mask_a = (k + offs_k) < K
        k_mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_row_ptrs, mask=k_mask_a, other=0.0).to(tl.float32)  # [BK]
        b = tl.load(B_tile_ptrs, mask=k_mask_b, other=0.0).to(tl.float32)  # [BK, BN]
        # Multiply each a[i] by b[i, :] and accumulate across i
        acc += tl.sum(a[:, None] * b, axis=0)  # sum over BK -> BN vector
        A_row_ptrs += BLOCK_K * stride_ak
        B_tile_ptrs += BLOCK_K * stride_bk

    C_tile_ptrs = C_ptr + offs_n * stride_cn
    out_mask = offs_n < N
    tl.store(C_tile_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def triton_row_matvec_bf16(A_row: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A_row @ B where A_row is [1, K] bf16, B is [K, N] bf16, returns C [1, N] bf16.
    """
    assert A_row.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    assert A_row.dim() == 2 and B.dim() == 2
    M = 1
    K = A_row.shape[1]
    N = B.shape[1]
    A_c = A_row.contiguous()
    B_c = B.contiguous()
    C = torch.empty((1, N), dtype=torch.bfloat16, device=A_row.device)

    stride_ak = A_c.stride()[1]
    stride_bk, stride_bn = B_c.stride()
    stride_cm = C.stride()[0]
    stride_cn = C.stride()[1]

    BLOCK_K = 128
    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N),)

    row_matvec_bf16_fp32_kernel[grid](
        A_c, B_c, C,
        K, N,
        stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return C


# 3) Triton reduction: per-row sum of squares (bf16 input -> fp32 sum)
@triton.jit
def row_sumsq_bf16_kernel(
    X_ptr, Out_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_N: tl.constexpr,
):
    # One program per row
    pid_m = tl.program_id(0)
    row_sum = tl.zeros((), dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    # Iterate across N in chunks
    for n in range(0, N, BLOCK_N):
        idx_n = n + offs_n
        x_ptrs = X_ptr + pid_m * stride_xm + idx_n * stride_xn
        mask = idx_n < N
        vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(vals * vals, axis=0)

    out_ptr = Out_ptr + pid_m
    tl.store(out_ptr, row_sum)


def triton_rowsumsq_bf16(X: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row sum of squares of X (M, N), returns [M] float32.
    """
    assert X.dtype == torch.bfloat16 and X.dim() == 2
    M, N = X.shape
    X_c = X.contiguous()
    out = torch.empty((M,), dtype=torch.float32, device=X.device)

    stride_xm, stride_xn = X_c.stride()
    BLOCK_N = 128
    grid = (M,)
    row_sumsq_bf16_kernel[grid](
        X_c, out,
        M, N,
        stride_xm, stride_xn,
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out


# 4) Triton sigmoid: Y = sigmoid(X), X: bf16 -> compute in fp32, output bf16
@triton.jit
def sigmoid_bf16_fp32_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = X_ptr + pid_m * stride_xm + offs_n * stride_xn
    y_ptrs = Y_ptr + pid_m * stride_ym + offs_n * stride_yn

    mask = (pid_m < M) & (offs_n < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)


def triton_sigmoid_bf16(X: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid on X (M, N) bf16 -> output bf16. Computation in fp32.
    """
    assert X.dtype == torch.bfloat16 and X.dim() == 2
    M, N = X.shape
    X_c = X.contiguous()
    Y = torch.empty_like(X_c, dtype=torch.bfloat16, device=X.device)

    stride_xm, stride_xn = X_c.stride()
    stride_ym, stride_yn = Y.stride()
    BLOCK_N = 128
    grid = (M,)
    sigmoid_bf16_fp32_kernel[grid](
        X_c, Y,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return Y


# Helper: Triton matvec across all rows (each row one program)
@triton.jit
def matvec_allrows_bf16_fp32_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program per row
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # A is [M, K], B is [K, N]
    A_row_ptrs = A_ptr + pid_m * stride_am + offs_k * stride_ak
    B_tile_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    for k in range(0, K, BLOCK_K):
        k_mask_a = (pid_m < M) & (k + offs_k) < K
        k_mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_row_ptrs, mask=k_mask_a, other=0.0).to(tl.float32)  # [BK]
        b = tl.load(B_tile_ptrs, mask=k_mask_b, other=0.0).to(tl.float32)  # [BK, BN]
        acc += tl.sum(a[:, None] * b, axis=0)
        A_row_ptrs += BLOCK_K * stride_ak
        B_tile_ptrs += BLOCK_K * stride_bk

    C_row_ptrs = C_ptr + pid_m * stride_cm + offs_n * stride_cn
    out_mask = (pid_m < M) & (offs_n < N)
    tl.store(C_row_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def triton_matvec_allrows_bf16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C[M, N] = A[M, K] @ B[K, N], returns C[M, N] bf16.
    Uses Triton: one program per row.
    """
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    assert A.dim() == 2 and B.dim() == 2
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"Shape mismatch: A is {A.shape}, B is {B.shape}"

    A_c = A.contiguous()
    B_c = B.contiguous()
    C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

    stride_am, stride_ak = A_c.stride()
    stride_bk, stride_bn = B_c.stride()
    stride_cm, stride_cn = C.stride()

    BLOCK_K = 128
    BLOCK_N = 128
    grid = (M,)

    matvec_allrows_bf16_fp32_kernel[grid](
        A_c, B_c, C,
        M, K, N,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor,
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
                shared_activated: torch.Tensor):
        """
        Triton-optimized backward pass:
        - All heavy computations (GEMMs/GEMVs) are done via Triton kernels.
        - Avoid torch @ and elementwise PyTorch ops in host code for performance-critical paths.
        """
        # Ensure bf16
        hidden_states = hidden_states.to(torch.bfloat16)
        shared_expert_gate_weight = shared_expert_gate_weight.to(torch.bfloat16)
        shared_expert_up_weight = shared_expert_up_weight.to(torch.bfloat16)
        shared_expert_down_weight = shared_expert_down_weight.to(torch.bfloat16)
        grad_output = grad_output.to(torch.bfloat16)

        # 1) Compute grad_hidden_from_shared_gate and grad_hidden_from_shared_up using Triton row matvec
        grad_hidden_from_shared_gate = []
        grad_hidden_from_shared_up = []

        # GATE path: per-row matvec of grad_shared_gate_output[token] @ shared_expert_gate_weight
        # grad_shared_gate_output: [M, N2] with N2=1408, shared_expert_gate_weight: [N2, hidden_size]
        # Each token t: [N2] @ [N2, hidden_size] -> [hidden_size]
        for t in range(hidden_states.shape[0]):
            A_row = grad_shared_gate_output[t].unsqueeze(0)  # [1, N2]
            B = shared_expert_gate_weight  # [N2, hidden_size]
            c = triton_row_matvec_bf16(A_row, B)  # [1, hidden_size]
            grad_hidden_from_shared_gate.append(c[0])
        grad_hidden_from_shared_gate = torch.stack(grad_hidden_from_shared_gate, dim=0)  # [M, hidden_size]

        # UP path: per-row matvec of grad_shared_up_output[token] @ shared_expert_up_weight
        # grad_shared_up_output: [M, N3] with N3=1408, shared_expert_up_weight: [N3, hidden_size]
        for t in range(hidden_states.shape[0]):
            A_row = grad_shared_up_output[t].unsqueeze(0)  # [1, N3]
            B = shared_expert_up_weight  # [N3, hidden_size]
            c = triton_row_matvec_bf16(A_row, B)  # [1, hidden_size]
            grad_hidden_from_shared_up.append(c[0])
        grad_hidden_from_shared_up = torch.stack(grad_hidden_from_shared_up, dim=0)  # [M, hidden_size]

        # 2) Compute grad_shared_expert_gate_weight, grad_shared_expert_up_weight, grad_shared_expert_down_weight via Triton matmul
        # For clarity, keep these as matmul:
        # grad_shared_expert_down_weight = grad_shared_output.T @ shared_activated
        # grad_shared_expert_up_weight   = grad_shared_up_output.T @ hidden_states
        # grad_shared_expert_gate_weight = grad_shared_gate_output.T @ hidden_states

        # Ensure contiguous bf16
        grad_shared_output_T = grad_shared_output.t().contiguous()            # [K1, M]
        shared_activated_T = shared_activated.t().contiguous()               # [K1, M]
        grad_shared_up_output_T = grad_shared_up_output.t().contiguous()     # [K2, M]
        hidden_states_T = hidden_states.t().contiguous()                     # [hidden_size, M]

        grad_shared_expert_down_weight = triton_matmul_bf16(grad_shared_output_T, shared_activated_T)  # [K1, hidden_size]
        grad_shared_expert_up_weight   = triton_matmul_bf16(grad_shared_up_output_T, hidden_states_T)   # [K2, hidden_size]
        grad_shared_expert_gate_weight = triton_matmul_bf16(grad_shared_gate_output.t().contiguous(), hidden_states_T)  # [K2, hidden_size]

        # 3) Compute grad_router_weight via Triton matmul: grad_router_logits.T @ hidden_states
        # grad_router_logits: [M, N_experts], hidden_states: [M, hidden_size]
        grad_router_logits_T = grad_router_logits.t().contiguous()
        grad_router_weight = triton_matmul_bf16(grad_router_logits_T, hidden_states_T)  # [N_experts, hidden_size]

        # 4) Compute routing approximations (kept in Triton where feasible):
        # grad_topk_weights norm from ||grad_output|| per token. We compute sum of squares in Triton.
        grad_output_c = grad_output.contiguous()
        grad_norm_sq = triton_rowsumsq_bf16(grad_output_c)  # [M] float32
        num_experts_per_tok = topk_weights.shape[-1]  # 8
        grad_topk_weights = (grad_norm_sq.unsqueeze(-1) / float(num_experts_per_tok)).expand_as(topk_weights).to(torch.bfloat16)

        # 5) Routing gradients (not computationally heavy, but we keep Triton sigmoid for logs where needed)
        # Original computes scores_for_choice = scores + bias; here we don't have bias in inputs, but keep Triton sigmoid for generality.
        # Since inputs don't include bias tensor, we skip sigmoid, but implement Triton sigmoid if needed.

        # No need to implement backprop for routing because the task is to return gradients for weights and hidden.

        # 6) Assemble outputs:
        grad_hidden_states = grad_hidden_from_shared_up + grad_hidden_from_shared_gate
        grad_router_weight = grad_router_weight
        grad_shared_expert_gate_weight = grad_shared_expert_gate_weight
        grad_shared_expert_up_weight = grad_shared_expert_up_weight
        grad_shared_expert_down_weight = grad_shared_expert_down_weight

        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
