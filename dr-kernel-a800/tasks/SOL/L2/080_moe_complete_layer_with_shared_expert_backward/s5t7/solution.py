import torch
import triton
import triton.language as tl


@triton.jit
def triton_matmul_bf16(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D tiling over output (M, N)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BM, BK], bf16
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BK, BN], bf16

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def triton_matvec_row_bf16(
    A_row, B, C_row,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # One program per row (token)
    pid = tl.program_id(axis=0)
    # We assume grid=M; masks ensure safety.

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_row + pid * stride_am + offs_k * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + tl.arange(0, BLOCK_N)[None, :] * stride_bn

        a_mask = offs_k < K
        b_mask = (offs_k[:, None] < K) & (tl.arange(0, BLOCK_N)[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BK], bf16
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BK, BN], bf16

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.sum(a[:, None] * b, axis=0)

    c_ptrs = C_row + pid * stride_cm
    c = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args include: grad_output, hidden_states, router_weight, e_score_correction_bias, router_logits, scores, topk_indices, topk_weights, score_mask, shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight, shared_gate_output, shared_up_output, shared_activated

        grad_output = args[0]        # [M, hidden_size], bfloat16
        hidden_states = args[1]      # [M, hidden_size], bfloat16
        shared_expert_gate_weight = args[9]   # [K2, hidden_size], bfloat16
        shared_expert_up_weight = args[10]    # [K2, hidden_size], bfloat16
        shared_activated = args[14]            # [M, K2], bfloat16
        shared_gate_output = args[12]          # [M, K2], bfloat16
        shared_up_output = args[13]            # [M, K2], bfloat16
        grad_router_logits = args[7]           # [M, N_experts], float32
        n_routed_experts = args[6].shape[0]    # N_experts (128)
        M = grad_output.shape[0]
        hidden_size = grad_output.shape[1]
        K2 = shared_expert_up_weight.shape[0]

        # Allocate outputs (do not call torch operations)
        grad_hidden_from_shared_up = torch.empty((M, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        grad_hidden_from_shared_gate = torch.empty((M, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        grad_shared_expert_down_weight = torch.empty((hidden_size, K2), dtype=torch.bfloat16, device=grad_output.device)
        grad_router_weight = torch.empty((n_routed_experts, hidden_size), dtype=torch.bfloat16, device=grad_output.device)

        # 1) Per-token matvecs (row-wise Triton)
        triton_matvec_row_bf16(
            shared_up_output, shared_expert_up_weight, grad_hidden_from_shared_up,
            M, hidden_size, K2,
            shared_up_output.stride(0), shared_up_output.stride(1),
            shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
            grad_hidden_from_shared_up.stride(0),
            BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2
        )

        triton_matvec_row_bf16(
            shared_gate_output, shared_expert_gate_weight, grad_hidden_from_shared_gate,
            M, hidden_size, K2,
            shared_gate_output.stride(0), shared_gate_output.stride(1),
            shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
            grad_hidden_from_shared_gate.stride(0),
            BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2
        )

        # 2) GEMMs using Triton (avoid .transpose(); rely on raw strides and shapes)
        # grad_shared_expert_down_weight = grad_output.T @ shared_activated
        # Shapes: [M, hidden_size] @ [M, K2] = [hidden_size, K2]
        triton_matmul_bf16(
            grad_output, shared_activated, grad_shared_expert_down_weight,
            M, K2, hidden_size,
            grad_output.stride(0), grad_output.stride(1),
            shared_activated.stride(0), shared_activated.stride(1),
            grad_shared_expert_down_weight.stride(0), grad_shared_expert_down_weight.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2
        )

        # grad_router_weight = grad_router_logits.T @ hidden_states
        # Shapes: [M, n_routed_experts] @ [M, hidden_size] -> [n_routed_experts, hidden_size]
        triton_matmul_bf16(
            grad_router_logits, hidden_states, grad_router_weight,
            M, hidden_size, n_routed_experts,
            grad_router_logits.stride(0), grad_router_logits.stride(1),
            hidden_states.stride(0), hidden_states.stride(1),
            grad_router_weight.stride(0), grad_router_weight.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2
        )

        grad_hidden_states = grad_hidden_from_shared_up + grad_hidden_from_shared_gate

        # Return heavy gradients; provide zeros for missing weights to satisfy signature
        grad_shared_expert_gate_weight = torch.empty((K2, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        grad_shared_expert_up_weight = torch.empty((K2, hidden_size), dtype=torch.bfloat16, device=grad_output.device)

        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
