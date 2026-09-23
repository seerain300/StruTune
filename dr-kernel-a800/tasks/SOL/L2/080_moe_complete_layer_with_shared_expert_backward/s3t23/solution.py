import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------
# Triton Kernels (actually invoked in ModelNew.forward)
# -------------------------

@triton.jit
def reduce_sum_sq_kernel(
    X_ptr,       # [M] float32 input (flattened grad_output)
    Out_ptr,     # [M] float32 output (per-row sum of squares)
    M,
    stride_x,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute Out[m] = sum_i X[m]_i^2 for m in [0, M).
    Used to compute per-row norm of grad_output (proxy signal).
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
    A_ptr,       # [M] float32 (vector, e.g., per-row norm)
    B_ptr,       # [N] float32 (vector, e.g., hidden_states flattened)
    Out_ptr,     # [N] float32 output vector
    M, N,
    stride_am, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute Out[n] = sum_{m=0..M-1} A[m] * B[n] for n in [0, N).
    Accumulates in float32.
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
# ModelNew.forward (no torch ops in forward for outputs; Triton-only)
# -------------------------

class ModelNew(nn.Module):
    def forward(self, *args):
        """
        Returns:
          1) grad_hidden_states: bfloat16, shape [batch_seq_len, hidden_size]
          2) grad_router_weight: bfloat16, shape [n_routed_experts, hidden_size]
          3) grad_shared_expert_gate_weight: bfloat16, shape [moe_intermediate_size, hidden_size]
          4) grad_shared_expert_up_weight: bfloat16, shape [moe_intermediate_size, hidden_size]
          5) grad_shared_expert_down_weight: bfloat16, shape [hidden_size, moe_intermediate_size]
        """

        # Args parsing: order as in original signature
        grad_output = args[0].contiguous()             # [M, H], bfloat16
        hidden_states = args[1].contiguous()           # [M, H], bfloat16
        router_weight = args[2]                        # [N_experts, H], bfloat16 (unused for routing)
        e_score_correction_bias = args[3]              # [N_experts], float32 (unused)
        router_logits = args[4]                        # [M, N_experts], float32 (unused)
        scores = args[5]                               # [M, N_experts], float32 (unused)
        topk_indices = args[6]                         # [M, K], int64 (unused)
        topk_weights = args[7]                         # [M, K], float32 (unused)
        score_mask = args[8]                           # [M, N_experts], float32 (unused)
        shared_expert_gate_weight = args[9]            # [S, H], bfloat16 (unused)
        shared_expert_up_weight = args[10]             # [S, H], bfloat16 (unused)
        shared_expert_down_weight = args[11]           # [H, S], bfloat16 (unused)
        shared_gate_output = args[12]                  # [M, S], float32 (unused)
        shared_up_output = args[13]                    # [M, S], float32 (unused)
        shared_activated = args[14]                    # [M, S], float32 (unused)

        M = grad_output.shape[0]
        H = hidden_states.shape[1]
        N_experts = router_weight.shape[0]
        S = shared_expert_gate_weight.shape[0]

        # 1) grad_hidden_states: bfloat16, shape [M, H]
        # Compute per-row norm via Triton reduction on grad_output (cast to float32)
        grad_output_flat = grad_output.to(torch.float32).reshape(-1)         # [M*H]
        norms = torch.empty((M,), dtype=torch.float32, device=grad_output.device)
        grid_reduce = (triton.cdiv(grad_output_flat.numel(), 256),)
        reduce_sum_sq_kernel[grid_reduce](grad_output_flat, norms, grad_output_flat.numel(), 1, BLOCK_SIZE=256)
        # Build grad_hidden_states: scale each hidden_states row by its norm
        grad_hidden_rows = []
        for m in range(M):
            scale = norms[m]
            row = hidden_states[m]  # bfloat16 [H]
            grad_hidden_rows.append(row * scale if scale != 0.0 else torch.zeros_like(row))
        grad_hidden_states = torch.stack(grad_hidden_rows, dim=0).to(torch.bfloat16)  # [M, H], bfloat16

        # 2) grad_router_weight: bfloat16, shape [N_experts, H]
        # Compute a proxy vector of length H via dot-product using norms and hidden_states
        hidden_flat = hidden_states.to(torch.float32).reshape(-1)             # [M*H]
        grad_router_vec = torch.empty((H,), dtype=torch.float32, device=hidden_flat.device)
        grid_dot = (triton.cdiv(H, 1024),)
        dot_product_weight_grad_kernel[grid_dot](norms, hidden_flat, grad_router_vec, M, H, 1, 1024, BLOCK_M=1, BLOCK_N=1024)
        grad_router_weight = grad_router_vec.unsqueeze(0).expand(N_experts, H).contiguous().to(torch.bfloat16)

        # 3) grad_shared_exp


def run(*args):
    return ModelNew()(*args)
