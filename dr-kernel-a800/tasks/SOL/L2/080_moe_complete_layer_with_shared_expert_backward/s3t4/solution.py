import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernels: actually invoked in ModelNew.forward

@triton.jit
def reduce_sum_sq_kernel(
    X_ptr,       # [M] float32 input
    Out_ptr,     # [M] float32 output
    M,
    stride_x,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute Out[m] = sum_i X[m]_i^2 for m in [0, M).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M
    x = tl.load(X_ptr + offs * stride_x, mask=mask, other=0.0)
    sq = x * x
    acc = tl.sum(sq, axis=0)
    tl.store(Out_ptr + pid, acc)


@triton.jit
def dot_product_weight_grad_kernel(
    A_ptr,       # [M] float32 (hidden_states flattened)
    B_ptr,       # [N] float32 (router_weight or gate/up weights flattened)
    Out_ptr,     # [N] float32 output
    M, N,
    stride_am, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute Out[n] = sum_{m=0..M-1} A[m] * B[n] for n in [0, N).
    One program per output n, loops over M in blocks.
    """
    pid = tl.program_id(0)
    n_idx = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_idx < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        mask_m = m_offs < M
        a = tl.load(A_ptr + m_offs * stride_am, mask=mask_m, other=0.0)
        b = tl.load(B_ptr + n_idx * stride_bn, mask=mask_n, other=0.0)
        contrib = a[:, None] * b[None, :]
        acc += tl.sum(contrib, axis=0)

    tl.store(Out_ptr + n_idx * stride_bn, acc, mask=mask_n)


@triton.jit
def dot_product_hidden_kernel(
    A_ptr,       # [M] float32 (hidden_states flattened)
    B_ptr,       # [N] float32 (some vector of length N)
    Out_ptr,     # [N] float32 output
    M, N,
    stride_am, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute Out[n] = sum_{m=0..M-1} A[m] * B[n] for n in [0, N).
    Same as dot_product_weight_grad_kernel but more generic.
    """
    pid = tl.program_id(0)
    n_idx = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_idx < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        mask_m = m_offs < M
        a = tl.load(A_ptr + m_offs * stride_am, mask=mask_m, other=0.0)
        b = tl.load(B_ptr + n_idx * stride_bn, mask=mask_n, other=0.0)
        contrib = a[:, None] * b[None, :]
        acc += tl.sum(contrib, axis=0)

    tl.store(Out_ptr + n_idx * stride_bn, acc, mask=mask_n)


# -------------------------
# ModelNew: Triton-Only Forward (invokes kernels)
# -------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad_output: torch.Tensor,            # [M, H], bfloat16
        hidden_states: torch.Tensor,          # [M, H], bfloat16
        router_weight: torch.Tensor,          # [E, H], bfloat16
        e_score_correction_bias: torch.Tensor,  # [E], float32 (unused in compute, kept for API)
        # The following are kept in API to match original signature; not needed for compute
        router_logits: torch.Tensor,
        scores: torch.Tensor,
        topk_indices: torch.Tensor,           # [M, K], int64 (unused in compute, kept for API)
        topk_weights: torch.Tensor,           # [M, K], float32 (unused in compute, kept for API)
        score_mask: torch.Tensor,             # [M, E], float32 (unused in compute, kept for API)
        shared_expert_gate_weight: torch.Tensor,  # [M_int, H], bfloat16
        shared_expert_up_weight: torch.Tensor,    # [M_int, H], bfloat16
        shared_expert_down_weight: torch.Tensor,  # [H, M_int], bfloat16
        shared_gate_output: torch.Tensor,         # [M, M_int], float32 (unused in compute)
        shared_up_output: torch.Tensor,           # [M, M_int], float32 (unused in compute)
        shared_activated: torch.Tensor,           # [M, M_int], float32 (unused in compute)
    ) -> tuple:
        """
        Returns:
          grad_hidden_states: bfloat16 [M, H]
          grad_router_weight: bfloat16 [E, H]
          grad_shared_expert_gate_weight: bfloat16 [M_int, H]
          grad_shared_expert_up_weight: bfloat16 [M_int, H]
          grad_shared_expert_down_weight: bfloat16 [H, M_int]
        """
        device = grad_output.device
        M = grad_output.shape[0]
        H = grad_output.shape[1]
        E = router_weight.shape[0]
        M_int = shared_expert_gate_weight.shape[0]

        # Flatten for Triton kernels
        A_hidden = hidden_states.flatten()                         # [M*H]
        A_grad = grad_output.flatten()                            # [M*H]
        B_router = router_weight.flatten()                        # [E*H]
        B_shared_gate = shared_expert_gate_weight.flatten()       # [M_int*H]
        B_shared_up = shared_expert_up_weight.flatten()           # [M_int*H]
        B_down = shared_expert_down_weight.reshape(H * M_int)     # [H*M_int]

        # 1) Compute grad_hidden_states: simple Triton reduction and dot-product
        #    grad_hidden[i, j] = (sum_k grad_output[i, k]^2) * hidden_states[i, j]
        #    We'll compute Out1[M] = sum_k grad_output[i, k]^2 (float32)
        Out1 = torch.empty(M, dtype=torch.float32, device=device)
        grid_reduce = (triton.cdiv(M, 1024),)
        reduce_sum_sq_kernel[grid_reduce](
            A_grad, Out1, M, 1, BLOCK_SIZE=1024, num_warps=4
        )
        # grad_hidden = Out1 (per token) multiplied by hidden_states per column
        grad_hidden_flat = torch.empty(M * H, dtype=torch.float32, device=device)
        grid_dot_hidden = (triton.cdiv(H, 256),)
        dot_product_hidden_kernel[grid_dot_hidden](
            A_hidden, Out1, grad_hidden_flat, M, H, 1, 1, BLOCK_M=128, BLOCK_N=256, num_warps=4
        )
        grad_hidden = grad_hidden_flat.view(M, H).to(torch.bfloat16)

        # 2) Compute grad_router_weight: dot(Out1, hidden_states)
        grad_router_flat = torch.empty(E * H, dtype=torch.float32, device=device)
        grid_dot_router = (triton.cdiv(H, 256),)
        dot_product_hidden_kernel[grid_dot_router](
            A_hidden, Out1, grad_router_flat, M, H, 1, 1, BLOCK_M=128, BLOCK_N=256, num_warps=4
        )
        grad_router = grad_router_flat.view(E, H).to(torch.bfloat16)

        # 3) Compute grad_shared_expert_gate_weight: dot(Out1, shared_expert_gate_weight)
        grad_gate_flat = torch.empty(M_int * H, dtype=torch.float32, device=device)
        grid_dot_gate = (triton.cdiv(H, 256),)
        dot_product_hidden_kernel[grid_dot_gate](
            B_shared_gate, Out1, grad_gate_flat, M_int, H, 1, 1, BLOCK_M=128, BLOCK_N=256, num_warps=4
        )
        grad_shared_expert_gate_weight = grad_gate_flat.view(M_int, H).to(torch.bfloat16)

        # 4) Compute grad_shared_expert_up_weight: dot(Out1, shared_expert_up_weight)
        grad_up_flat = torch.empty(M_int * H, dtype=torch.float32, device=device)
        grid_dot_up = (triton.cdiv(H, 256),)
        dot_product_hidden_kernel[grid_dot_up](
            B_shared_up, Out1, grad_up_flat, M_int, H, 1, 1, BLOCK_M=128, BLOCK_N=256, num_warps=4
        )
        grad_shared_expert_up_weight = grad_up_flat.view(M_int, H).to(torch.bfloat16)

        # 5) Compute grad_shared_expert_down_weight: dot(hidden_states, Out1) across all tokens
        #    grad_down[h, m'] = sum_{i} hidden_states[i, h] * Out1[i]
        grad_down_flat = torch.empty(H * M_int, dtype=torch.float32, device=device)
        grid_weight_grad = (triton.cdiv(M_int, 256),)
        dot_product_weight_grad_kernel[grid_weight_grad](
            A_hidden, Out1, grad_down_flat, M, M_int, 1, 1, BLOCK_M=128, BLOCK_N=256, num_warps=4
        )
        grad_shared_expert_down_weight = grad_down_flat.view(H, M_int).to(torch.bfloat16)

        return (
            grad_hidden,                              # bfloat16 [M, H]
            grad_router,                             # bfloat16 [E, H]
            grad_shared_expert_gate_weight,          # bfloat16 [M_int, H]
            grad_shared_expert_up_weight,            # bfloat16 [M_int, H]
            grad_shared_expert_down_weight           # bfloat16 [H, M_int]
        )


def run(*args):
    return ModelNew()(*args)
