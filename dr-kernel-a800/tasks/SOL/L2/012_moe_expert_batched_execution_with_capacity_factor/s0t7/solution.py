import math
import torch
import triton
import triton.language as tl


@triton.jit
def sort_stable_odd_even(exp_ptr, wt_ptr, tok_ptr, N, BLOCK: tl.constexpr):
    """
    Stable sort by selected_experts using odd-even transposition sort.
    Sorts N elements. Each program handles BLOCK consecutive elements and iterates N passes.
    exp_ptr: int64
    wt_ptr: bfloat16
    tok_ptr: int64
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    for t in range(0, N):
        # Even phase: pairs (0,1), (2,3), ...
        is_even_pair = (t % 2 == 0) & ((idx % 2) == 0) & in_bounds
        # Odd phase:  pairs (1,2), (3,4), ...
        is_odd_pair  = (t % 2 == 1) & (((idx + 1) % 2) == 0) & in_bounds

        i = idx
        j = i + 1

        exp_i = tl.load(exp_ptr + i, mask=in_bounds, other=0)
        wt_i  = tl.load(wt_ptr  + i, mask=in_bounds, other=0.0)
        tok_i = tl.load(tok_ptr + i, mask=in_bounds, other=0)

        exp_j = tl.load(exp_ptr + j, mask=j < N, other=0)
        wt_j  = tl.load(wt_ptr  + j, mask=j < N, other=0.0)
        tok_j = tl.load(tok_ptr + j, mask=j < N, other=0)

        # Compare and swap when exp_i > exp_j (stable: preserve original order ties)
        swap = (exp_i > exp_j)

        new_exp_i = tl.where(swap, exp_j, exp_i)
        new_exp_j = tl.where(swap, exp_i, exp_j)

        new_wt_i  = tl.where(swap, wt_j,  wt_i)
        new_wt_j  = tl.where(swap, wt_i,  wt_j)

        new_tok_i = tl.where(swap, tok_j, tok_i)
        new_tok_j = tl.where(swap, tok_i, tok_j)

        # Store back
        tl.store(exp_ptr + i, new_exp_i, mask=in_bounds)
        tl.store(exp_ptr + j, new_exp_j, mask=j < N)

        tl.store(wt_ptr  + i, new_wt_i,  mask=in_bounds)
        tl.store(wt_ptr  + j, new_wt_j,  mask=j < N)

        tl.store(tok_ptr + i, new_tok_i, mask=in_bounds)
        tl.store(tok_ptr + j, new_tok_j, mask=j < N)


@triton.jit
def bincount_kernel(inp_ptr, out_ptr, N, E: tl.constexpr, BLOCK: tl.constexpr):
    """
    Triton bincount over inp_ptr (int64). Each program counts a BLOCK subset,
    then atomic_add into out_ptr (int32). We use int64 load for safety and cast to int32.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    vals = tl.load(inp_ptr + idx, mask=in_bounds, other=0).to(tl.int64)
    counts = tl.zeros((BLOCK,), dtype=tl.int32)

    # For each expert id in [0, E), check occurrences in this block
    for e in range(0, E):
        mask_e = (vals == e) & in_bounds
        counts += mask_e.to(tl.int32)

    # Atomic add per element
    tl.atomic_add(out_ptr + vals, counts, mask=in_bounds)


@triton.jit
def cumsum_kernel(inp_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    Cumulative sum of int32 vector inp_ptr -> out_ptr. Each program handles BLOCK elements,
    computes local cumsum, and writes. For simplicity, we assume N fits in grid * BLOCK.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    x = tl.load(inp_ptr + idx, mask=in_bounds, other=0).to(tl.int32)
    # Prefix sum within the block
    prefix = tl.zeros((BLOCK,), dtype=tl.int32)
    running = 0
    for i in range(0, BLOCK):
        running += x[i]
        prefix[i] = running

    tl.store(out_ptr + idx, prefix, mask=in_bounds)


@triton.jit
def silu_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    """
    SiLU elementwise: y = x * sigmoid(x), computed in fp32 for stability, cast back.
    Assumes x_ptr is fp16/bf16 and y_ptr matches dtype of x_ptr.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    x = tl.load(x_ptr + idx, mask=in_bounds, other=0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y.to(tl.dtype_from_ptr(y_ptr)), mask=in_bounds)


@triton.jit
def bmm_forward_kernel_left_right(A_ptr, B_ptr, C_ptr,
                                   M, Ndim, Kdim,
                                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Triton matrix multiply: A [M, Kdim] @ B [Kdim, Ndim] -> C [M, Ndim].
    All inputs are 2D contiguous. We tile over M and N, and reduce over K.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * Kdim + offs_k[None, :])
    b_ptrs = B_ptr + (offs_k[:, None] * Ndim + offs_n[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, Kdim, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < Kdim), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < Kdim) & (offs_n[None, :] < Ndim), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * Ndim + offs_n[None, :])
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < Ndim))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        hidden_states: (num_tokens, hidden_size), bfloat16
        selected_experts: (num_tokens, K), int64
        routing_weights: (num_tokens, K), bfloat16
        expert_gate_weights: (num_experts, hidden_size, intermediate_size), bfloat16
        expert_up_weights:    (num_experts, hidden_size, intermediate_size), bfloat16
        expert_down_weights:  (num_experts, intermediate_size, hidden_size), bfloat16
        """
        # We deliberately avoid any PyTorch tensor compute in forward, except for
        # lightweight shape handling and tensor creation. The heavy work and all
        # preprocessing are performed via Triton kernels launched from here.

        # Extract shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_hs, intermediate_size = expert_gate_weights.shape
        K = selected_experts.shape[1]
        N = num_tokens * K

        # Devices
        device = hidden_states.device

        # Flatten assignments
        flat_exp = selected_experts.reshape(-1).to(torch.int64)     # [N]
        flat_wt  = routing_weights.reshape(-1).to(torch.bfloat16)   # [N]

        # For repeat token ids, we can compute flat_tok as 0..N-1 mapped back to token


def run(*args):
    return ModelNew()(*args)
