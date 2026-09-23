import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X: [rows, N] where rows = batch_size * seq_len, N = intermediate_size.
# MEAN: fp32 [rows], STD: fp32 [rows]
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32, contiguous [rows, N]
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    sum_val = 0.0
    sum_sq = 0.0
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    # population std
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: compute per-row threshold and write to THRESH [rows]
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    Q_ptr,           # *f32, scalar (1 element) containing ndtri(target_sparsity)
    THRESH_ptr,      # *f32, [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    q = tl.load(Q_ptr)  # scalar
    thr = mean + std * q
    tl.store(THRESH_ptr + row_id, thr)


# Kernel 3: apply ReLU(x - threshold[row]) elementwise, write to OUT
@triton.jit
def relu_threshold_kernel(
    X_ptr,           # *f32, linearized [rows, N]
    THRESH_ptr,      # *f32, [rows]
    OUT_ptr,         # *f32, linearized [rows, N]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    thresh = tl.load(THRESH_ptr + row_id)
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = tl.maximum(x - thresh, 0.0)
        tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, input: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous fp32 input for Triton kernels
        assert input.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = input.contiguous()
        B, S, N = x.shape
        rows = B * S

        # Compute in fp32 for numerical stability
        x_f32 = x.to(torch.float32)

        # Allocate per-row stats
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)

        # Launch row-wise reduction
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_f32, mean, std,
            rows=rows, N=N,
            BLOCK=1024,
            num_warps=8,
        )

        # Compute q = ndtri(target_sparsity) using torch to ensure stable scalar
        # Abramowitz & Stegun approximation for ndtri (standard normal inverse CDF)
        # This is a scalar; torch ops are fine here.
        # Constants
        a1 = -3.969683028665376e+01
        a2 = 2.209460984245205e+02
        a3 = -2.759285104469687e+02
        a4 = 1.383577518672690e+02
        a5 = -3.066479806614716e+01
        a6 = 2.506628277459239e+00

        b1 = -5.447609879822406e+01
        b2 = 1.615858368580409e+02
        b3 = -1.556989798598866e+02
        b4 = 6.680131188771972e+01
        b5 = -1.328068155288572e+01

        c1 = -7.784894002430293e-03
        c2 = -3.223964580411365e-01
        c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00
        c5 = 4.374664141464968e+00
        c6 = 2.938163982698783e+00

        d1 = 7.784695709041462e-03
        d2 = 3.224671290700398e-01
        d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00

        p_low = 0.02425
        p_high = 1.0 - p_low

        # Handle regions
        p = torch.tensor(target_sparsity, device=x.device, dtype=torch.float32)
        q = torch.zeros((), device=x.device, dtype=torch.float32)  # scalar output

        # Lower region
        if p < p_low:
            # q = sqrt(-2 log(p))
            q = torch.sqrt(-2.0 * torch.log(p))
            approx = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                     ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        elif p <= p_high:
            # Central region
            q = p - 0.5
            r = q * q
            approx = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                     (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        else:
            # Upper region
            q = torch.sqrt(-2.0 * torch.log(1.0 - p))
            approx = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                     ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

        # 3) Compute per-row threshold
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise to OUT
        OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            x_f32, threshold, OUT,
            rows=rows, N=N, BLOCK=1024, num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
