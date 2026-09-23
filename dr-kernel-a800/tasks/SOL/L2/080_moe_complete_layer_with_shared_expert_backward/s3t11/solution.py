import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------
# Triton Kernel: per-token sum of squares reduction
# Input: X_ptr [M] float32 (flattened grad_output in bfloat16 -> cast to float32)
# Output: Out_ptr [M] float32 (sum of squares per token)
# -------------------------

@triton.jit
def reduce_sum_sq_kernel(
    X_ptr,       # [M] float32
    Out_ptr,     # [M] float32
    M,
    stride_x,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute Out[m] = sum_i X[m]_i^2 for m in [0, M).
    M is the number of tokens (batch_seq_len).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M
    x = tl.load(X_ptr + offs * stride_x, mask=mask, other=0.0)
    sq = x * x
    acc = tl.sum(sq, axis=0)
    tl.store(Out_ptr + pid, acc)


# -------------------------
# Triton Kernel: dot product Out[n] = sum_{m=0..M-1} A[m] * B[n]
# A_ptr: [M] float32
# B_ptr: [N] float32
# Out_ptr: [N] float32
# -------------------------

@triton.jit
def dot_product_weight_grad_kernel(
    A_ptr,       # [M] float32
    B_ptr,       # [N] float32
    Out_ptr,     # [N] float32
    M, N,
    stride_am, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute Out[n] = sum_{m=0..M-1} A[m] * B[n] for n in [0, N).
    One program per output n in blocks of BLOCK_N, loops over M in blocks of BLOCK_M.
    """
    pid = tl.program_id(0)
    n_idx = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_idx < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        mask_m = m_offs < M
        a = tl.load(A_ptr + m_offs * stride_am, mask=mask_m, other=0.0)  # [BLOCK_M]
        b = tl.load(B_ptr + n_idx * stride_bn, mask=mask_n, other=0.0)   # [BLOCK_N]
        acc += tl.sum(a[:, None] * b[None, :], axis=0)

    tl.store(Out_ptr + n_idx * stride_bn, acc, mask=mask_n)


# -------------------------
# ModelNew.forward: Triton-only, returns 5 bfloat16 tensors
# -------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad_output: torch.Tensor,           # [B, H], bfloat16
        hidden_states: torch.Tensor,         # [B, H], bfloat16
        router_weight: torch.Tensor,         # not used
        e_score_correction_bias: torch.Tensor,  # not used
        router_logits: torch.Tensor,         # not used
        scores: torch.Tensor,                # not used
        topk_indices: torch.Tensor,          # not used
        topk_weights: torch.Tensor,          # not used
        score_mask: torch.Tensor,            # not used
        shared_expert_gate_weight: torch.Tensor,  # [S, H], bfloat16
        shared_expert_up_weight: torch.Tensor,   # [S, H], bfloat16
        shared_expert_down_weight: torch.Tensor, # [H, S], bfloat16
        shared_gate_output: torch.Tensor,    # [B, S], float32
        shared_up_output: torch.Tensor,      # [B, S], float32
        shared_activated: torch.Tensor       # [B, S], float32
    ) -> tuple:
        """
        Returns:
        grad_hidden_states: [B, H], bfloat16
        grad_router_weight: [E, H], bfloat16 (E=128)
        grad_shared_expert_gate_weight: [S, H], bfloat16
        grad_shared_expert_up_weight: [S, H], bfloat16
        grad_shared_expert_down_weight: [H, S], bfloat16
        """
        B, H = grad_output.shape
        # Ensure CUDA tensors
        assert grad_output.is_cuda and hidden_states.is_cuda, "Triton requires CUDA tensors"

        # Compute per-token sum of squares in float


def run(*args):
    return ModelNew()(*args)
