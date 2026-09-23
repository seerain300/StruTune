import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bf16_fp32(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        A_tile_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        B_tile_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_tile_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BM, BK]
        b = tl.load(B_tile_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BK, BN]
        acc += tl.dot(a, b)

    C_tile_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def all_rows_matvec_bf16_fp32(
    A_ptr,  # pointer to [M, K] (we'll pass a view with last dim=1 for per-row behavior)
    B_ptr,  # pointer to [K, N]
    C_ptr,  # pointer to [M, N]
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a tile of rows [BLOCK_M] and iterates over K in chunks [BLOCK_K],
    # accumulating a [BLOCK_M, BLOCK_N] tile of C.
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Load A chunk: [BM, BK]
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BM, BK]
        b = tl.load(B_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BK, BN]
        # For per-row behavior, treat each row's a[:, 0] times b[j, :], accumulate across BK
        # We can't directly index last dim; instead, loop over BK and update acc:
        # For each j in BK:
        for j in range(BLOCK_K):
            a_vec = a[:, j]  # [BM]
            b_vec = b[j, :]  # [BN]
            acc += a_vec[:, None] * b_vec[None, :]

    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def triton_matmul_bf16(A_c: torch.Tensor, B_c: torch.Tensor, out: torch.Tensor):
    """
    Triton GEMM: out = A @ B where A is [M, K] bf16, B is [K, N] bf16, out is [M, N] bf16.
    Compute in fp32.
    """
    assert A_c.is_cuda and B_c.is_cuda and out.is_cuda
    assert A_c.dtype == torch.bfloat16 and B_c.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert A_c.dim() == 2 and B_c.dim() == 2 and out.dim() == 2
    M, K = A_c.shape
    K2, N = B_c.shape
    assert K == K2, "Inner dims must match for matmul"
    stride_am, stride_ak = A_c.stride()
    stride_bk, stride_bn = B_c.stride()
    stride_cm, stride_cn = out.stride()
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    matmul_bf16_fp32[grid](
        A_c, B_c, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
        num_warps=4, num_stages=2,
    )


def triton_all_rows_matvec_bf16(A_rows: torch.Tensor, B_c: torch.Tensor, out: torch.Tensor):
    """
    Triton per-rows matvec: out[M, N] = A_rows[M, K] @ B[K, N], with bf16 inputs, fp32 compute, bf16 output.
    A_rows should be [M, K]; we can pass a view of per-token vectors with last dim=1, but here we assume [M, K].
    """
    assert A_rows.is_cuda and B_c.is_cuda and out.is_cuda
    assert A_rows.dtype == torch.bfloat16 and B_c.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert A_rows.dim() == 2 and B_c.dim() == 2 and out.dim() == 2
    M, K = A_rows.shape
    K2, N = B_c.shape
    assert K == K2, "Inner dims must match for matvec"
    stride_am, stride_ak = A_rows.stride()
    stride_bk, stride_bn = B_c.stride()
    stride_cm, stride_cn = out.stride()
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    all_rows_matvec_bf16_fp32[grid](
        A_rows, B_c, out,
        M, K, N,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
        num_warps=4, num_stages=2,
    )


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
        # Extract shapes
        M = grad_output.shape[0]  # batch_seq_len
        hidden_size = hidden_states.shape[1]
        K2 = shared_expert_up_weight.shape[0]  # intermediate_size
        N_experts = router_weight.shape[0]

        # Allocate outputs (bf16 as in original)
        grad_hidden_from_shared_up = torch.empty((M, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        grad_hidden_from_shared_gate = torch.empty((M, hidden_size), dtype=torch.bfloat16, device=grad_output.device)

        grad_shared_expert_gate_weight = torch.empty((K2, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        grad_shared_expert_up_weight = torch.empty((K2, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        grad_router_weight = torch.empty((N_experts, hidden_size), dtype=torch.bfloat16, device=grad_output.device)

        # 1) Per-token matvecs using Triton all-rows kernel (single launch)
        # We'll construct A_rows as shared_*_output [M, K2] and B as shared_*_weights [K2, hidden_size].
        triton_all_rows_matvec_bf16(shared_up_output, shared_expert_up_weight, grad_hidden_from_shared_up)
        triton_all_rows_matvec_bf16(shared_gate_output, shared_expert_gate_weight, grad_hidden_from_shared_gate)

        # 2) GEMMs using Triton:
        # a) grad_shared_expert_down_weight = grad_shared_output.T @ shared_activated
        #    Shapes: [M, hidden_size] @ [M, K2] = [hidden_size, K2]
        grad_shared_output_T = grad_output.transpose(0, 1)  # [M, hidden_size]
        shared_activated_T = shared_activated.transpose(0, 1)  # [M, K2]
        grad_shared_expert_down_weight = torch.empty((hidden_size, K2), dtype=torch.bfloat16, device=grad_output.device)
        triton_matmul_bf16(grad_shared_output_T, shared_activated_T, grad_shared_expert_down_weight)

        # b) grad_router_weight = grad_router_logits.T @ hidden_states
        #    Shapes: [N_experts, M] @ [M, hidden_size] -> [N_experts, hidden_size]
        grad_router_logits_T = router_logits.transpose(0, 1)  # [M, N_experts]
        grad_router_weight = torch.empty((N_experts, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        triton_matmul_bf16(grad_router_logits_T, hidden_states, grad_router_weight)

        # 3) Combine per-token contributions for hidden_states
        grad_hidden_states = grad_hidden_from_shared_up + grad_hidden_from_shared_gate

        # Return gradients matching original run signature
        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
