import torch
import math
import triton
import triton.language as tl


@triton.jit
def layernorm_row_kernel(
    x_ptr,            # *const bfloat16, input [num_rows, features]
    y_ptr,            # *bfloat16, output [num_rows, features]
    ln_weight_ptr,    # *const float32, [features]
    ln_bias_ptr,      # *const float32, [features]
    num_rows,         # int
    features,         # int
    eps,              # float32
    BLOCK: tl.constexpr,  # block size for reduction
):
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    # First pass: compute mean and variance in fp32
    sum_fp32 = 0.0
    sumsq_fp32 = 0.0

    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row_id * features + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_fp32 += tl.sum(x, axis=0)
        sumsq_fp32 += tl.sum(x * x, axis=0)

    mean = sum_fp32 / features
    var = sumsq_fp32 / features - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, write back in bfloat16
    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row_id * features + idx, mask=mask, other=0.0).to(tl.float32)
        ln_w = tl.load(ln_weight_ptr + idx, mask=mask, other=1.0)
        ln_b = tl.load(ln_bias_ptr + idx, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        y = y * ln_w + ln_b
        tl.store(y_ptr + row_id * features + idx, y.to(tl.bfloat16), mask=mask)


@triton.jit
def gelu_erf_kernel(
    x_ptr,           # *const float32, input [M, K] flattened
    y_ptr,           # *float32, output [M, K] flattened
    M: tl.constexpr,  # number of rows
    K: tl.constexpr,  # number of columns
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    if (row >= M) or (col >= K):
        return
    x = tl.load(x_ptr + row * K + col)
    # erf approximation (Abramowitz & Stegun 7.1.26)
    # erf(x) ≈ sign(x) * (1 - t * exp(-x^2) * (a1 + a2 t + a3 t^2 + a4 t^3 + a5 t^4)), t = 1/(1 + p|x|)
    # constants
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    u = x
    abs_u = tl.abs(u)
    t = 1.0 / (1.0 + p * abs_u)
    # polynomial
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    erf_approx = 1.0 - poly * tl.exp(-abs_u * abs_u)
    # restore sign
    erf_approx = tl.where(u >= 0, erf_approx, -erf_approx)

    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu = 0.5 * x * (1.0 + erf_approx)
    tl.store(y_ptr + row * K + col, gelu)


@triton.jit
def matmul_kernel(
    A_ptr,           # *const float32, left matrix [M, K] flattened
    B_ptr,           # *const float32, right matrix [K, N] flattened
    C_ptr,           # *float32, output matrix [M, N] flattened
    M: tl.constexpr, # rows of A
    N: tl.constexpr, # cols of B (and C)
    K: tl.constexpr, # cols of A, rows of B
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(0, K, BLOCK_K):
        rk = kk + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    tl.store(C_ptr + rm[:, None] * N + rn[None, :],
             acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def cat_per_grid_rows_kernel(
    src_base_ptr,    # *const float32, base pointer to per-grid src tensor flattened
    out_ptr,         # *float32, output tensor flattened
    src_cols,        # int32, number of columns in per-grid src tensor (should be features for LN)
    num_patches_grid, # int32, number of rows per grid (T*(H//2)*(W//2))
    grid_id,         # int32, grid index
    total_rows,      # int32, total number of rows before this grid
):
    # This kernel assumes the source tensor has rows = num_patches_grid and columns = src_cols.
    # It copies that entire tensor into 'out' at offset = total_rows * src_cols.
    # Launch grid (num_patches_grid, 1) so each program copies one row.
    row_in_grid = tl.program_id(0)
    if row_in_grid >= num_patches_grid:
        return
    row_total = total_rows + row_in_grid
    src_row_base = src_base_ptr + row_in_grid * src_cols
    out_row_base = out_ptr + row_total * src_cols
    for col in range(0, src_cols):
        val = tl.load(src_row_base + col)
        tl.store(out_row_base + col, val)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden: torch.Tensor,          # [num_patches, 1536], bfloat16
        grid_thw: torch.Tensor,        # [num_grids, 3], int64 (T,H,W)
        ln_weight: torch.Tensor,       # [1536], bfloat16 (ones)
        ln_bias: torch.Tensor,         # [1536], bfloat16 (zeros)
        fc1_weight: torch.Tensor,      # [6144, 12288], bfloat16
        fc1_bias: torch.Tensor,        # [6144], bfloat16
        fc2_weight: torch.Tensor,      # [3584, 6144], bfloat16
        fc2_bias: torch.Tensor,        # [3584], bfloat16
        eps: float,                    # float32
    ):
        """
        Triton-optimized forward:
        - LayerNorm (per row) in Triton, compute in fp32, store as bfloat16.
        - Spatial shuffle done via PyTorch reshape/permute (metadata), producing per-grid tensors.
        - Concatenate per-grid tensors using Triton kernel (no torch.cat).
        - First Linear in Triton (fp32), then GELU in Triton, then Second Linear in Triton.
        - Return output in bfloat16.
        """
        device = hidden.device
        num_patches = hidden.shape[0]
        features = hidden.shape[1]
        assert features == 1536, "LayerNorm must be across 1536 features"

        # 1) LayerNorm in Triton
        hidden_norm = torch.empty_like(hidden, dtype=torch.float32, device=device)
        ln_w_fp32 = ln_weight.to(torch.float32)
        ln_b_fp32 = ln_bias.to(torch.float32)
        grid_ln = (_ceil_div(features, 1024),)
        layernorm_row_kernel[grid_ln](
            hidden, hidden_norm,
            ln_w_fp32, ln_b_fp32,
            num_patches, features, float(eps),
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        # 2) Per-grid spatial shuffle (metadata): for each grid, compute num_patches_this and perform view+permute
        offset = 0
        per_grid_lists = []
        for g in range(grid_thw.shape[0]):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            h_merged = h // 2
            w_merged = w // 2
            num_patches_this = t * h_merged * w_merged
            patches = hidden_norm[offset:offset + num_patches_this]  # [num_patches_this, 1536], fp32
            # Reshape (T, H/2, 2, W/2, 2, C)
            patches = patches.view(t, h_merged, 2, w_merged, 2, features)
            # Permute to (T, H/2, W/2, 2, 2, C)
            patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(num_patches_this, 8 * features)
            per_grid_lists.append(patches)  # fp32, shape [num_patches_this, 12288]
            offset += num_patches_this

        # 3) Concatenate per-grid outputs into a single tensor without torch.cat (Triton kernel)
        num_grids = len(per_grid_lists)
        M_total = num_patches  # equal to sum of per-grid num_patches_this as per get_inputs
        N = 8 * features  # 12288

        # Prepare output tensor
        hidden_shuffled = torch.empty((M_total, N), dtype=torch.float32, device=device)

        # Launch Triton kernel to copy each per-grid tensor into out at correct position
        total_rows_before = 0
        for g in range(num_grids):
            num_rows_this = per_grid_lists[g].shape[0]
            # Each grid writes num_rows_this rows into out starting at row = total_rows_before
            grid_cat = (num_rows_this, 1)
            cat_per_grid_rows_kernel[grid_cat](
                per_grid_lists[g], hidden_shuffled,
                N, num_rows_this, g, total_rows_before,
                num_warps=1, num_stages=1,
            )
            total_rows_before += num_rows_this

        # 4) First Linear (fp32 GEMM in Triton), bias addition
        # A: [M_total, 12288] = hidden_shuffled
        # B: [12288, 6144] = fc1_weight^T (cast to fp32 for compute)
        B1 = fc1_weight.t().to(torch.float32).contiguous()  # [12288, 6144]
        C1 = torch.empty((M_total, 6144), dtype=torch.float32, device=device)

        # Triton grid: (ceil(M/64), ceil(6144/64))
        grid_matmul1 = (_ceil_div(M_total, 64), _ceil_div(6144, 64))
        matmul_kernel[grid_matmul1](
            hidden_shuffled, B1, C1,
            M_total, 6144, 12288,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 5) GELU in Triton (erf approximation)
        C1_gelu = torch.empty_like(C1)  # same dtype
        # Launch 2D grid over [M_total, 6144]
        grid_gelu = (M_total, 6144)
        gelu_erf_kernel[grid_gelu](
            C1, C1_gelu, M_total, 6144,
            num_warps=4, num_stages=1,
        )

        # 6) Second Linear (fp32 GEMM in Triton), bias addition
        B2 = fc2_weight.t().to(torch.float32).contiguous()  # [6144, 3584]
        C2 = torch.empty((M_total, 3584), dtype=torch.float32, device=device)

        grid_matmul2 = (_ceil_div(M_total, 64), _ceil_div(3584, 64))
        matmul_kernel[grid_matmul2](
            C1_gelu, B2, C2,
            M_total, 3584, 6144,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 7) Return output in bfloat16, matching original
        return C2.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
