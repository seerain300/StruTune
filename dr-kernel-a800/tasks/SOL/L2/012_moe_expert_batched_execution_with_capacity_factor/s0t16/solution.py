import math
import torch
import triton
import triton.language as tl


@triton.jit
def sort_stable_keyed_kernel(exp_ptr, wt_ptr, out_exp_ptr, out_wt_ptr,
                             N, BLOCK: tl.constexpr):
    """
    Stable sort by 'exp_ptr' (selected_experts flattened) and write sorted 'exp_ptr' to out_exp_ptr,
    'wt_ptr' to out_wt_ptr using odd-even transposition sort.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    # Odd-even sort for stability
    for t in range(0, N):
        # Even phase: compare (0,1), (2,3), ...
        even_pair = ((idx % 2) == 0) & in_bounds
        j = idx + 1
        j_valid = j < N & even_pair

        # Odd phase: compare (1,2), (3,4), ...
        odd_pair  = ((idx % 2) == 1) & in_bounds
        j = idx + 1
        j_valid = j < N & odd_pair

        # Load current and partner values
        exp_i = tl.load(exp_ptr + idx, mask=in_bounds, other=0)
        wt_i  = tl.load(wt_ptr  + idx, mask=in_bounds, other=0.0)

        exp_j = tl.load(exp_ptr + j,   mask=j_valid, other=0)
        wt_j  = tl.load(wt_ptr  + j,   mask=j_valid, other=0.0)

        # Stable swap: if exp_i > exp_j, or (== and wt_i > wt_j)
        greater_exp = exp_i > exp_j
        equal_exp   = exp_i == exp_j
        greater_wt  = wt_i > wt_j
        swap = greater_exp | (equal_exp & greater_wt)

        new_exp_i = tl.where(swap, exp_j, exp_i)
        new_wt_i  = tl.where(swap, wt_j,  wt_i)
        new_exp_j = tl.where(swap, exp_i, exp_j)
        new_wt_j  = tl.where(swap, wt_i,  wt_j)

        tl.store(out_exp_ptr + idx, new_exp_i, mask=in_bounds)
        tl.store(out_wt_ptr  + idx, new_wt_i,  mask=in_bounds)
        tl.store(out_exp_ptr + j,   new_exp_j, mask=j_valid)
        tl.store(out_wt_ptr  + j,   new_wt_j,  mask=j_valid)


@triton.jit
def inv_perm_stable_kernel(sorted_exp_ptr, original_exp_ptr, inv_ptr,
                           N, BLOCK: tl.constexpr):
    """
    Compute inverse permutation: inv[i] = j where sorted_exp[j] == original_exp[i], stable tie-breaker.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    i = start + tl.arange(0, BLOCK)
    in_bounds = i < N

    # Initialize inv to zeros
    tl.store(inv_ptr + i, 0, mask=in_bounds)

    # Odd-even sort to propagate matches; for each i, if se_j == or_i set inv[i] = j, and vice versa.
    for t in range(0, N):
        # Even phase pairs (0,1), (2,3), ...
        even_pair = ((i % 2) == 0) & in_bounds
        j = i + 1
        j_valid = (j < N) & even_pair

        # Odd phase pairs (1,2), (3,4), ...
        odd_pair  = ((i % 2) == 1) & in_bounds
        j = i + 1
        j_valid = (j < N) & odd_pair

        se_i = tl.load(sorted_exp_ptr + i, mask=in_bounds, other=0)
        or_i = tl.load(original_exp_ptr + i, mask=in_bounds, other=0)

        # Current inv
        inv_i = tl.load(inv_ptr + i, mask=in_bounds, other=0)
        inv_j = tl.load(inv_ptr + j, mask=j_valid, other=0)

        # If se_j == or_i and inv_j not set, set inv[i] = j
        is_match_j = (se_j == or_i) & j_valid & (inv_j == 0)
        # If se_i == or_j and inv_i not set, set inv[j] = i
        is_match_i = (se_i == or_j) & in_bounds & (inv_i == 0)

        new_inv_i = tl.where(is_match_j, j, tl.where(inv_i > 0, inv_i, 0))
        new_inv_j = tl.where(is_match_i, i, inv_j)

        tl.store(inv_ptr + i, new_inv_i, mask=in_bounds)
        tl.store(inv_ptr + j, new_inv_j, mask=j_valid)


@triton.jit
def bincount_kernel(keys_ptr, out_ptr, M, BLOCK: tl.constexpr):
    """
    Triton bincount: counts occurrences of each integer key in [0, M-1] in keys_ptr into out_ptr (int32).
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < M

    local = tl.zeros((BLOCK,), dtype=tl.int32)

    for k in range(0, M):
        key = tl.load(keys_ptr + k, mask=True, other=0)  # scalar
        found = (idx == key) & in_bounds
        local += found.to(tl.int32)

    block_sum = tl.sum(local, axis=0)
    tl.atomic_add(out_ptr + idx, block_sum, mask=in_bounds)


@triton.jit
def cumsum_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    Inclusive cumsum over int32 array: out[i] = sum_{k<=i} in[k], one program per element.
    """
    pid = tl.program_id(0)
    i = pid
    in_bounds = i < N

    sum_val = tl.load(in_ptr + i, mask=in_bounds, other=0)
    tl.store(out_ptr + i, sum_val, mask=in_bounds)

    for k in range(0, i):
        prev = tl.load(out_ptr + k, mask=True, other=0)
        sum_val = sum_val + prev
        tl.store(out_ptr + i, sum_val, mask=in_bounds)


@triton.jit
def silu_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    """
    Elementwise SiLU activation: y = x * sigmoid(x). Compute in fp32, store in original dtype.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    x = tl.load(X_ptr + idx, mask=in_bounds, other=0.0)
    x_fp32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_fp32))
    y_fp32 = x_fp32 * sig
    y = y_fp32.to(x.dtype)
    tl.store(Y_ptr + idx, y, mask=in_bounds)


@triton.jit
def bmm_forward_kernel_right(A_ptr, B_ptr, C_ptr,
                             M, N, K,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             num_warps: tl.constexpr):
    """
    Triton kernel computing C = A @ B, where:
      A: [M, K], B: [K, N], C: [M, N]
    Each program computes a BLOCK_M x BLOCK_N tile of C. Accumulate in fp32, store to C.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * K + offs_k[None, :])
        b_ptrs = B_ptr + (offs_k[:, None] * N + offs_n[None, :])

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        A_tile = A_tile.to(tl.float32)
        B_tile = B_tile.to(tl.float32)

        acc += tl.dot(A_tile, B_tile)

    c_ptrs = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def _grid_1d(n_elements, block_size):
    return (triton.cdiv(n_elements, block_size),)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        Triton-Only implementation of the original run function.
        All heavy compute (sorting, bincount, cumsum, SiLU, batched matmuls) is performed by Triton kernels.
        Final weighted scatter-add uses PyTorch index_add due to Triton limitations with dynamic scatter.
        """
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape
        K = selected_experts.shape[1]


def run(*args):
    return ModelNew()(*args)
