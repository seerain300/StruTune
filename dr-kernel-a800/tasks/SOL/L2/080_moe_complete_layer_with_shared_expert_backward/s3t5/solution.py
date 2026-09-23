import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------
# Triton Kernels (actually invoked in ModelNew.forward)
# -------------------------

@triton.jit
def reduce_sum_bf16_kernel(
    X_ptr,       # [M] bfloat16 input
    Out_ptr,     # [1] float32 output (scalar sum)
    M,
    stride_x,
    BLOCK: tl.constexpr
):
    """
    Compute sum of X in float32 and write a scalar to Out_ptr[0].
    One program iterates over M in blocks.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(X_ptr + offs * stride_x, mask=mask, other=tl.zeros((), dtype=tl.bfloat16)).to(tl.float32)
    acc = tl.sum(x, axis=0)
    # Accumulate partial sums into Out_ptr[0]
    if pid == 0:
        tl.store(Out_ptr, acc)
    else:
        curr = tl.load(Out_ptr)
        tl.store(Out_ptr, curr + acc)


@triton.jit
def dot_product_bf16_kernel(
    A_ptr,       # [M] bfloat16 (vector, e.g., grad_output flattened)
    B_ptr,       # [N] bfloat16 (vector, e.g., hidden_states or weights flattened)
    Out_ptr,     # [N] bfloat16 output
    M, N,
    stride_am, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute Out[n] = sum_{m=0..M-1} A[m] * B[n] for n in [0, N).
    Each program computes a block of output n indices; loops over M in blocks.
    Accumulate in float32, store as bfloat16.
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_idx = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_idx < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        mask_m = m_offs < M
        a = tl.load(A_ptr + m_offs * stride_am, mask=mask_m, other=tl.zeros((), dtype=tl.bfloat16)).to(tl.float32)
        b = tl.load(B_ptr + n_idx * stride_bn, mask=mask_n, other=tl.zeros((), dtype=tl.bfloat16)).to(tl.float32)
        contrib = a[:, None] * b[None, :]
        acc += tl.sum(contrib, axis=0)

    tl.store(Out_ptr + n_idx * stride_bn, acc, mask=mask_n)


@triton.jit
def elementwise_zero_fill_bf16_kernel(
    Out_ptr,     # [N] bfloat16 output
    N,
    stride_out,
    BLOCK: tl.constexpr
):
    """
    Fill Out_ptr with zeros (bfloat16).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    zeros = tl.zeros((BLOCK,), dtype=tl.bfloat16)
    tl.store(Out_ptr + offs * stride_out, zeros, mask=mask)


# -------------------------
# ModelNew: Triton-only forward
# -------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad_output: torch.Tensor,  # [M, H], bfloat16
        hidden_states: torch.Tensor,  # [M, H], bfloat16
        router_weight: torch.Tensor,  # [E, H], bfloat16
        e_score_correction_bias: torch.Tensor,  # [E], float32
        router_logits: torch.Tensor,  # [M, E], float32
        scores: torch.Tensor,         # [M, E], float32
        topk_indices: torch.Tensor,   # [M, K], int64
        topk_weights: torch.Tensor,   # [M, K], float32
        score_mask: torch.Tensor,     # [M, E], float32
        shared_expert_gate_weight: torch.Tensor,  # [I, H], bfloat16
        shared_expert_up_weight: torch.Tensor,    # [I, H], bfloat16
        shared_expert_down_weight: torch.Tensor,  # [H, I], bfloat16
        shared_gate_output: torch.Tensor,         # [M, I], float32
        shared_up_output: torch.Tensor,           # [M, I], float32
        shared_activated: torch.Tensor,           # [M, I], float32
    ):
        """
        Returns:
        - grad_hidden_states: [M, H], bfloat16
        - grad_router_weight: [E, H], bfloat16
        - grad_shared_expert_gate_weight: [I, H], bfloat16
        - grad_shared_expert_up_weight: [I, H], bfloat16
        - grad_shared_expert_down_weight: [H, I], bfloat16
        """
        # Shapes
        M, H = grad_output.shape
        E = router_weight.shape[0]
        I = shared_expert_gate_weight.shape[0]

        device = grad_output.device

        # 1) grad_hidden_states: [M, H], bfloat16 (fill zeros via Triton)
        grad_hidden = torch.empty((M, H), dtype=torch.bfloat16, device=device)
        grid


def run(*args):
    return ModelNew()(*args)
