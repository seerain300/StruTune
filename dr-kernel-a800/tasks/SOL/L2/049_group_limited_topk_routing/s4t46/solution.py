import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_AxB_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [N, K], float32 (we load as B[k, n] via strides)
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # strides for B[n, k]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + ((k0 + offs_k)[None, :] * stride_ak)
        a_mask = (offs_m < M)[:, None] & ((k0 + offs_k) < K)[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile as [BLOCK_K, BLOCK_N] from B[n, k] with strides
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + ((k0 + offs_k)[:, None] * stride_bk)
        b_mask = ((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

        k0 += BLOCK_K

    # Store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,    # [M, N], float32
    B_ptr,    # [N], float32
    Y_ptr,    # [M, N], float32
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK: tl.constexpr,
):
    # One program per row, iterate columns in chunks
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    j = 0
    while j < N:
        offs = j + tl.arange(0, BLOCK)
        mask = offs < N
        x_ptrs = X_ptr + pid_m * stride_xm + offs * stride_xn
        y_ptrs = Y_ptr + pid_m * stride_ym + offs * stride_yn
        b_ptrs = B_ptr + offs

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        b = tl.load(b_ptrs, mask=mask, other=0.0)

        # sigmoid(x) = 1 / (1 + exp(-x))
        y = 1.0 / (1.0 + tl.exp(-x))
        y = y + b  # broadcast add bias

        tl.store(y_ptrs, y, mask=mask)
        j += BLOCK


@triton.jit
def _group_top2_sum_kernel(
    S_ptr,     # [M, N], float32 scores
    GROUPS: tl.constexpr,     # 8
    EXP_PER_GRP: tl.constexpr,  # 32
    GS_ptr,    # [M, GROUPS], float32 output group_scores
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # For each group g in 0..GROUPS-1: find top-2 scores in its 32-expert slice and sum
    for g in range(GROUPS):
        col_start = g * EXP_PER_GRP
        # We'll scan EXP_PER_GRP and keep two best values
        best1 = -float('inf')
        best2 = -float('inf')

        j = 0
        while j < EXP_PER_GRP:
            offs = col_start + j
            val = tl.load(S_ptr + pid_m * stride_sm + offs * stride_sn)
            if val > best1:
                best2 = best1
                best1 = val
            elif val > best2:
                best2 = val
            j += 1

        group_score = best1 + best2
        tl.store(GS_ptr + pid_m * stride_gm + g * stride_gn, group_score)


@triton.jit
def _group_top4_select_kernel(
    GS_ptr,   # [M, GROUPS], float32 group scores
    GROUPS: tl.constexpr,  # 8
    TOPK_GROUPS: tl.constexpr,  # 4
    GIDX_ptr, # [M, TOPK_GROUPS], int32 output indices
    M,
    stride_gm, stride_gn,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Use Triton's topk to select top-4 per row over GROUPS columns
    gs = tl.load(GS_ptr + pid_m * stride_gm + tl.arange(0, GROUPS) * stride_gn)
    values = gs
    indices = tl.arange(0, GROUPS)
    # Triton supports selecting topk over a vector; we need to return indices and values
    # For topk, we can't directly access return indices, so we emulate:
    # Perform iterative selection: get max, mark excluded, repeat for TOPK_GROUPS
    # Here we implement it manually by scanning groups since GROUPS is small.
    selected = tl.zeros((GROUPS,), dtype=tl.int1)
    k = 0
    while k < TOPK_GROUPS:
        max_val = -float('inf')
        max_pos = 0
        j = 0
        while j < GROUPS:
            # Access scalar value at position j
            # Triton does not support dynamic indexing into vectors easily; emulate by recomputing max across all j
            # We'll recompute via reduction: find max_val among unselected
            unselected_vals = values * (1 - selected[j])  # not a proper mask, but we'll handle via loop
            # Instead, keep simple approach: for each j, compare max_val
            val_j = values[j]
            if (not selected[j]) and val_j > max_val:
                max_val = val_j
                max_pos = j
            j += 1
        # Record max_pos
        tl.store(GIDX_ptr + pid_m * stride_om + k * stride_on, max_pos)
        # Mark this position as selected
        selected = selected | (indices == max_pos)
        k += 1


@triton.jit
def _final_top8_and_normalize_kernel(
    S_ptr,         # [M, N], float32 masked scores (non-selected groups are -inf)
    GIDX_ptr,      # [M, TOPK_GROUPS], int32 group indices per token (TOPK_GROUPS=4)
    EXP_PER_GRP: tl.constexpr,   # 32
    N: tl.constexpr,
    TOPK_FINAL: tl.constexpr,    # 8
    IDX_ptr,       # [M, TOPK_FINAL], int32 output indices
    WT_ptr,        # [M, TOPK_FINAL], float32 output weights
    M,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    stride_im, stride_in,
    stride_wm, stride_wn,
    routed_scaling_factor: tl.float32,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # We assume non-selected groups in S_ptr have been set to -inf by host-side preprocessing.

    # Use Triton topk to select top-8 over N columns
    scores_row = tl.load(S_ptr + pid_m * stride_sm + tl.arange(0, N) * stride_sn)
    vals = scores_row
    idxs = tl.arange(0, N)
    # Triton does not provide topk returning indices directly in many examples; emulate iterative top-k.
    # Since N=256 and TOPK_FINAL=8, we can do manual selection:
    selected = tl.zeros((N,), dtype=tl.int1)
    selected_vals = tl.zeros((TOPK_FINAL,), dtype=tl.float32) - 1.0
    selected_idx = tl.zeros((TOPK_FINAL,), dtype=tl.int32) - 1

    k = 0
    while k < TOPK_FINAL:
        max_val = -float('inf')
        max_pos = -1
        j = 0
        while j < N:
            val_j = vals[j]
            # Consider only unselected
            if (not selected[j]) and val_j > max_val:
                max_val = val_j
                max_pos = j
            j += 1
        # Store index
        tl.store(IDX_ptr + pid_m * stride_im + k * stride_in, max_pos)
        tl.store(WT_ptr + pid_m * stride_wm + k * stride_wn, max_val)
        # Mark selected
        selected = selected | (idxs == max_pos)
        k += 1

    # Normalize and scale
    sum_vals = 0.0
    k = 0
    while k < TOPK_FINAL:
        val = tl.load(WT_ptr + pid_m * stride_wm + k * stride_wn)
        sum_vals += val
        k += 1

    k = 0
    while k < TOPK_FINAL:
        val = tl.load(WT_ptr + pid_m * stride_wm + k * stride_wn)
        norm = val / sum_vals
        scaled = norm * routed_scaling_factor
        tl.store(WT_ptr + pid_m * stride_wm + k * stride_wn, scaled)
        k += 1


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,  # PyTorch's nn.Linear stores [num_experts, hidden_dim]
        expert_bias: torch.Tensor,  # [num_experts]
        routed_scaling_factor: float,
    ):
        device = hidden_states.device
        dtype = torch.float32

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]  # hidden_dim
        N = 256  # num_experts
        EXP_PER_GRP = 32
        GROUPS = 8
        TOPK_GROUPS = 4
        TOPK_FINAL = 8


def run(*args):
    return ModelNew()(*args)
