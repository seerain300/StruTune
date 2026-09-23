import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_rows_kernel(
    X_ptr,          # *const bfloat16: input hidden [num_patches, hidden_size]
    W_ptr,          # *const bfloat16: ln_weight [hidden_size]
    B_ptr,          # *const bfloat16: ln_bias [hidden_size]
    Y_ptr,          # *bfloat16: output normalized [num_patches, hidden_size]
    NUM_PATCHES: tl.constexpr,  # int
    HIDDEN_SIZE,    # int
    EPS,            # float32
    BLOCK_SIZE: tl.constexpr,   # must be >= hidden_size
):
    # One program per row
    row_id = tl.program_id(0)
    if row_id >= NUM_PATCHES:
        return
    # Compute row base pointers
    x_row_ptr = X_ptr + row_id * HIDDEN_SIZE
    y_row_ptr = Y_ptr + row_id * HIDDEN_SIZE

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE

    # Load row in bf16, cast to fp32 for compute
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / HIDDEN_SIZE
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / HIDDEN_SIZE
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Normalize and apply ln_weight + ln_bias
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * w + b  # fp32

    # Store as bf16
    tl.store(y_row_ptr + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _pack_rows_kernel(
    X_ptr,           # *const bfloat16: input packed rows [num_patches * HIDDEN_SIZE] but we index by row
    OUT_ptr,         # *bfloat16: output packed 1D vector [num_patches * HIDDEN_SIZE_EXPANDED]
    ROWS,            # int: num_patches
    HIDDEN_SIZE,     # int
    HIDDEN_EXPANDED, # int
    BLOCK: tl.constexpr,  # must be >= HIDDEN_EXPANDED
):
    # One program per output slot
    slot_id = tl.program_id(0)
    if slot_id >= ROWS * HIDDEN_EXPANDED:
        return
    row = slot_id // HIDDEN_EXPANDED
    col = slot_id % HIDDEN_EXPANDED
    in_cols = tl.arange(0, BLOCK)
    mask = in_cols < HIDDEN_EXPANDED
    x = tl.load(X_ptr + row * HIDDEN_SIZE + in_cols, mask=mask, other=0.0)
    tl.store(OUT_ptr + slot_id, x.to(tl.bfloat16), mask=mask)


@triton.jit
def _gemm_rows_cols_kernel(
    A_ptr,          # *const bfloat16: [M, K]
    W_ptr,          # *const bfloat16: [N, K]
    B_ptr,          # *bfloat16: [M, N]
    M,              # int
    N,              # int
    K,              # int
    stride_am,      # int: stride for A in M (usually K)
    stride_ak,      # int: stride for A in K (usually 1)
    stride_wn,      # int: stride for W in N (usually 1)
    stride_wk,      # int: stride for W in K (usually N)
    stride_bm,      # int: stride for B in M (usually N)
    stride_bn,      # int: stride for B in N (usually 1)
    HAS_BIAS,       # int: 0 or 1
    BIAS_ptr,       # *const bfloat16: bias [N]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: one program handles a BLOCK_M x BLOCK_N tile of B
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    mask_n = off_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        off_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = off_k < K

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load W tile: shape [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + off_n[None, :] * stride_wn + off_k[:, None] * stride_wk
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, w)

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(BIAS_ptr + off_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # Store result in B (bf16)
    b_ptrs = B_ptr + off_m[:, None] * stride_bm + off_n[None, :] * stride_bn
    tl.store(b_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _gelu_tanh_kernel(
    X_ptr,          # *const bfloat16: input [M, N]
    Y_ptr,          # *bfloat16: output [M, N]
    M,              # int
    N,              # int
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D launch: one program handles a tile of output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)

    x = tl.load(X_ptr + off_m[:, None] * N + off_n[None, :], mask=mask, other=0.0).to(tl.float32)

    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    gelu = 0.5 * x * (1.0 + tl.tanh(c0 * (x + c1 * x3)))

    tl.store(Y_ptr + off_m[:, None] * N + off_n[None, :], gelu.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float):
        """
        Triton-only forward:
        - LayerNorm (per-row) via Triton
        - Pack rows into 1D vector via Triton
        - First Linear via Triton GEMM
        - GELU via Triton elementwise
        - Second Linear via Triton GEMM
        No torch ops are used for the core computation.
        """
        device = hidden.device
        # Ensure contiguous
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()
        grid_thw = grid_thw.contiguous()  # not used in packing but kept for signature

        num_patches, hidden_size = hidden.shape
        hidden_size_expanded = 4 * hidden_size  # merge 2x2 -> 4 features per original hidden_size
        num_merged_patches = num_patches // 4   # as per provided workloads

        # 1) LayerNorm in Triton
        hidden_norm = torch.empty((num_patches, hidden_size), dtype=torch.bfloat16, device=device)
        grid_layernorm = (num_patches,)
        _layernorm_rows_kernel[grid_layernorm](
            hidden, ln_weight, ln_bias, hidden_norm,
            num_patches, hidden_size, eps,
            BLOCK_SIZE=hidden_size  # must be >= hidden_size
        )

        # 2) Pack rows into 1D vector [num_patches * hidden_size_expanded] via Triton
        hidden_pack = torch.empty(num_patches * hidden_size_expanded, dtype=torch.bfloat16, device=device)
        grid_pack = (num_patches * hidden_size_expanded,)
        _pack_rows_kernel[grid_pack](
            hidden_norm, hidden_pack, num_patches, hidden_size, hidden_size_expanded,
            BLOCK=hidden_size_expanded  # must be >= hidden_size_expanded
        )

        # Reshape into [num_merged_patches, hidden_size_expanded]
        hidden_linear1 = hidden_pack.view(num_merged_patches, hidden_size_expanded)

        # 3) First Linear: (num_merged_patches, 6144) @ (6144, 6144) -> (num_merged_patches, 6144)
        B1 = torch.empty((num_merged_patches, hidden_size_expanded), dtype=torch.bfloat16, device=device)
        grid_gemm1 = (triton.cdiv(num_merged_patches, 128), triton.cdiv(hidden_size_expanded, 128))
        _gemm_rows_cols_kernel[grid_gemm1](
            hidden_linear1, fc1_weight, B1,
            num_merged_patches, hidden_size_expanded, hidden_size_expanded,
            hidden_linear1.stride(0), hidden_linear1.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            B1.stride(0), B1.stride(1),
            1, fc1_bias,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )

        # 4) GELU activation via Triton
        B1_gelu = torch.empty_like(B1, dtype=torch.bfloat16, device=device)
        grid_gelu = (triton.cdiv(num_merged_patches, 64), triton.cdiv(hidden_size_expanded, 128))
        _gelu_tanh_kernel[grid_gelu](
            B1, B1_gelu,
            num_merged_patches, hidden_size_expanded,
            BLOCK_M=64, BLOCK_N=128
        )

        # 5) Second Linear: (num_merged_patches, 6144) @ (3584, 6144) -> (num_merged_patches, 3584)
        out_hidden_size = fc2_weight.shape[0]
        output = torch.empty((num_merged_patches, out_hidden_size), dtype=torch.bfloat16, device=device)
        grid_gemm2 = (triton.cdiv(num_merged_patches, 128), triton.cdiv(out_hidden_size, 128))
        _gemm_rows_cols_kernel[grid_gemm2](
            B1_gelu, fc2_weight, output,
            num_merged_patches, out_hidden_size, hidden_size_expanded,
            B1_gelu.stride(0), B1_gelu.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output.stride(0), output.stride(1),
            1, fc2_bias,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
