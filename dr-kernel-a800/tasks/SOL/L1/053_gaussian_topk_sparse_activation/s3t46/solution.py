import torch
import triton
import triton.language as tl


# Kernel 1: compute per-feature sum and sumsq across all rows (B*S)
# x_ptr: *float32, flattened [rows, L]
# sum_ptr: *float32, [L]
# sumsq_ptr: *float32, [L]
@triton.jit
def reduce_feature_sum_sumsq_kernel(
    x_ptr,           # *float32, flattened [rows, L]
    sum_ptr,         # *float32, [L]
    sumsq_ptr,       # *float32, [L]
    rows: tl.int32,  # number of rows (B*S), runtime scalar
    L: tl.int32,     # number of features, runtime scalar
    BLOCK_R: tl.constexpr  # chunk size for rows
):
    pid = tl.program_id(axis=0)  # feature index 0..L-1
    if pid >= L:
        return

    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    # Iterate over rows in chunks of BLOCK_R
    # We use static-range so Triton can unroll and handle it
    for chunk in range(0, rows, BLOCK_R):
        row_offsets = chunk + tl.arange(0, BLOCK_R)
        mask = row_offsets < rows
        # x is flattened as [rows, L], so index = row * L + f
        x_vals = tl.load(x_ptr + row_offsets * L + pid, mask=mask, other=0.0)
        acc_sum += tl.sum(x_vals, axis=0)
        acc_sumsq += tl.sum(x_vals * x_vals, axis=0)

    tl.store(sum_ptr + pid, acc_sum)
    tl.store(sumsq_ptr + pid, acc_sumsq)


# Kernel 2: compute threshold per feature: thr = mean + std * multiplier
# mean_ptr, std_ptr: *float32, [L]
# thr_ptr: *float32, [L]
# multiplier: scalar float32 (ndtri(target_sparsity))
@triton.jit
def compute_threshold_kernel(
    mean_ptr,        # *float32, [L]
    std_ptr,         # *float32, [L]
    thr_ptr,         # *float32, [L]
    multiplier,      # scalar float32
    L: tl.int32
):
    pid = tl.program_id(axis=0)
    if pid >= L:
        return
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    thr = mean + std * multiplier
    tl.store(thr_ptr + pid, thr)


# Kernel 3: apply sparse ReLU elementwise with per-feature threshold
# x_ptr: *float32, flattened [rows, L]
# thr_ptr: *float32, [L]
# out_ptr: *float32, flattened [rows, L]
@triton.jit
def sparse_relu_kernel(
    x_ptr,           # *float32, [rows, L]
    thr_ptr,         # *float32, [L]
    out_ptr,         # *float32, [rows, L]
    rows: tl.int32,
    L: tl.int32
):
    pid_row = tl.program_id(axis=0)  # row index
    pid_col = tl.program_id(axis=1)  # feature index
    if pid_row >= rows or pid_col >= L:
        return
    x_val = tl.load(x_ptr + pid_row * L + pid_col)
    thr = tl.load(thr_ptr + pid_col)
    y = x_val - thr
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + pid_row * L + pid_col, y)


# Kernel 4: cast float32 output to bfloat16
# out_ptr_f32: *float32, flattened [rows, L]
# out_ptr_bf16: *bfloat16, flattened [rows, L]
@triton.jit
def cast_bf16_kernel(
    out_ptr_f32,     # *float32, [rows, L]
    out_ptr_bf16,    # *bfloat16, [rows, L]
    rows: tl.int32,
    L: tl.int32
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)
    if pid_row >= rows or pid_col >= L:
        return
    val = tl.load(out_ptr_f32 + pid_row * L + pid_col)
    tl.store(out_ptr_bf16 + pid_row * L + pid_col, tl.cast(val, tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-only forward:
        - Launch reduce_feature_sum_sumsq_kernel to compute per-feature sum and sumsq.
        - Launch compute_threshold_kernel to compute thr = mean + std * ndtri(target_sparsity).
        - Launch sparse_relu_kernel to apply ReLU with per-feature threshold.
        - Launch cast_bf16_kernel to cast output to bfloat16.
        Return tensor with same shape as x, dtype bfloat16.
        """
        # Ensure dtype float32 and contiguity for Triton
        x_f32 = x.contiguous().to(torch.float32)
        B, S, L = x_f32.shape
        rows = B * S

        # Flatten to [rows, L]
        x_flat = x_f32.view(rows, L).contiguous()

        # Allocate buffers for per-feature sums and sumsq
        sum_buf = torch.empty(L, dtype=torch.float32, device=x.device)
        sumsq_buf = torch.empty(L, dtype=torch.float32, device=x.device)

        # Launch kernel to compute per-feature sum and sumsq
        # Use a chunk size that balances occupancy; 256 is a good default
        BLOCK_R = 256
        grid_reduce = (L,)  # one program per feature
        reduce_feature_sum_sumsq_kernel[grid_reduce](
            x_flat, sum_buf, sumsq_buf, rows, L, BLOCK_R
        )

        # Compute mean and std via simple PyTorch ops on device (tiny vs elementwise)
        # mean = sum / rows, var = sumsq / rows - mean^2
        mean = sum_buf / rows
        var = sumsq_buf / rows - mean * mean
        var = torch.clamp(var, min=0.0)  # numerical safety
        std = torch.sqrt(var)

        # Compute ndtri(target_sparsity) using Abramowitz & Stegun 26.2.23 approximation.
        # This is done in forward with minimal device ops; no torch elementwise on x.
        p = float(target_sparsity)
        p_low = 0.02425
        p_high = 1.0 - p_low

        if p < p_low:
            q = torch.sqrt(-2.0 * torch.log(torch.tensor(p, dtype=torch.float32, device=x.device)))
            # Rational approximation in one expression
            m = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        elif p > p_high:
            q = torch.sqrt(-2.0 * torch.log(torch.tensor(1.0 - p, dtype=torch.float32, device=x.device)))
            m = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        else:
            q = p - 0.5
            r = q * q
            m = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

        # Allocate threshold buffer
        thr = torch.empty(L, dtype=torch.float32, device=x.device)

        # Launch threshold kernel (one program per feature)
        grid_thr = (L,)
        compute_threshold_kernel[grid_thr](mean, std, thr, m, L)

        # Prepare output buffer for ReLU result (FP32)
        out_flat = torch.empty((rows, L), dtype=torch.float32, device=x.device)

        # Launch ReLU kernel over 2D grid (rows, features)
        grid_relu = (rows, L)
        sparse_relu_kernel[grid_relu](x_flat, thr, out_flat, rows, L)

        # Cast to bfloat16 using Triton
        out_bf16_flat = torch.empty((rows, L), dtype=torch.bfloat16, device=x.device)
        cast_bf16_kernel[grid_relu](out_flat, out_bf16_flat, rows, L)

        # Reshape back to [B, S, L]
        out = out_bf16_flat.view(B, S, L)
        return out

# Constants for A&S approximation
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
c2 = -3.223964580411395e-01
c3 = -2.400758277161838e+00
c4 = -2.549732539343734e+00
c5 = 4.374664141464968e+00
c6 = 2.938163982698783e+00

d1 = 7.784695709041462e-03
d2 = 3.224671290700398e-01
d3 = 2.445134137142996e+00
d4 = 3.754408661907416e+00


def run(*args):
    return ModelNew()(*args)
