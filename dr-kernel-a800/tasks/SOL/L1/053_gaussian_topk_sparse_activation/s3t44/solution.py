import torch
import triton
import triton.language as tl


@triton.jit
def sum_sumsq_per_feature(x_ptr,          # *fp32
                           sum_ptr,        # *fp32 (length L)
                           sumsq_ptr,      # *fp32 (length L)
                           rows,           # int32
                           L,              # int32
                           BLOCK_R: tl.constexpr):
    """
    2D grid over features and row chunks:
    - program_id(0): feature f
    - program_id(1): chunk id over rows
    Each program accumulates partial sum and sumsq for feature f across its rows chunk
    and atomically adds into sum_ptr[f] and sumsq_ptr[f].
    """
    pid_f = tl.program_id(0)
    pid_c = tl.program_id(1)
    if pid_f >= L:
        return

    # Rows this program handles
    base = pid_c * BLOCK_R
    rows_idx = base + tl.arange(0, BLOCK_R)
    mask = rows_idx < rows

    # Load x[rows_idx, f] as a vector
    x_vals = tl.load(x_ptr + rows_idx * L + pid_f, mask=mask, other=0.0)

    # Accumulate partial sums
    s = tl.sum(x_vals, axis=0)
    ss = tl.sum(x_vals * x_vals, axis=0)

    # Atomically add to global accumulators
    tl.atomic_add(sum_ptr + pid_f, s)
    tl.atomic_add(sumsq_ptr + pid_f, ss)


@triton.jit
def compute_mean_std_per_feature(sum_ptr,    # *fp32 (length L)
                                  sumsq_ptr,  # *fp32 (length L)
                                  mean_ptr,   # *fp32 (length L)
                                  std_ptr,    # *fp32 (length L)
                                  rows,       # int32
                                  L: tl.constexpr):
    """
    One program per feature f.
    Compute mean and std from sum and sumsq.
    mean_f = sum_f / rows
    var_f = sumsq_f / rows - mean_f^2
    std_f = sqrt(max(var_f, 0))
    """
    pid_f = tl.program_id(0)
    if pid_f >= L:
        return

    sum_f = tl.load(sum_ptr + pid_f)
    sumsq_f = tl.load(sumsq_ptr + pid_f)

    mean_f = sum_f / rows
    var_f = sumsq_f / rows - mean_f * mean_f
    std_f = tl.sqrt(var_f)

    # Clamp std to non-negative to avoid tiny negative due to rounding
    std_f = tl.where(std_f >= 0.0, std_f, 0.0)

    tl.store(mean_ptr + pid_f, mean_f)
    tl.store(std_ptr + pid_f, std_f)


@triton.jit
def compute_thr_per_feature(mean_ptr, std_ptr, thr_ptr, multiplier, L: tl.constexpr):
    """
    Compute per-feature threshold: thr[f] = mean_f + std_f * multiplier.
    One program per feature.
    """
    pid_f = tl.program_id(0)
    if pid_f >= L:
        return

    mean_f = tl.load(mean_ptr + pid_f)
    std_f = tl.load(std_ptr + pid_f)
    thr_f = mean_f + std_f * multiplier
    tl.store(thr_ptr + pid_f, thr_f)


@triton.jit
def sparse_relu_per_feature(x_ptr,        # *fp32, [rows, L]
                             thr_ptr,      # *fp32, [L]
                             out_ptr,      # *fp32, [rows, L]
                             rows,         # int32
                             L,            # int32
                             BLOCK_R: tl.constexpr):
    """
    Apply elementwise sparse ReLU with per-feature threshold:
      out[row, f] = max(x[row, f] - thr[f], 0)
    2D grid: program_id(0) over row chunks, program_id(1) over features.
    """
    pid_r = tl.program_id(0)
    pid_f = tl.program_id(1)
    if pid_r >= rows or pid_f >= L:
        return

    base = pid_r * BLOCK_R
    rows_idx = base + tl.arange(0, BLOCK_R)
    mask_r = rows_idx < rows

    # Load feature vector
    x_vals = tl.load(x_ptr + rows_idx * L + pid_f, mask=mask_r, other=0.0)
    thr_f = tl.load(thr_ptr + pid_f)

    out_vals = tl.maximum(x_vals - thr_f, 0.0)
    tl.store(out_ptr + rows_idx * L + pid_f, out_vals, mask=mask_r)


@triton.jit
def cast_bf16_kernel(in_ptr,  # *fp32
                     out_ptr,  # *bf16
                     numel,    # int32
                     BLOCK_SIZE: tl.constexpr):
    """
    Cast fp32 to bf16 elementwise.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    vals = tl.load(in_ptr + offs, mask=mask, other=0.0)
    vals_bf16 = vals.to(tl.bfloat16)
    tl.store(out_ptr + offs, vals_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float, multiplier: float):
        super().__init__()
        # multiplier is ndtri(target_sparsity) precomputed
        self.target_sparsity = float(target_sparsity)
        self.multiplier = float(multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, L] float32 CUDA tensor.
        Returns: [B, S, L] bfloat16 tensor (sparsified and cast).
        """
        assert x.is_cuda, "ModelNew.forward requires a CUDA tensor"
        assert x.dtype == torch.float32, "Expected input dtype to be float32"

        B, S, L = x.shape
        rows = B * S

        # Flatten to [rows, L] contiguous
        x_flat = x.contiguous().view(rows, L)

        # 1) Per-feature reduction: sum and sumsq using Triton
        sum_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)
        sumsq_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)

        BLOCK_R = 1024  # process rows in chunks
        grid_sum = (L, triton.cdiv(rows, BLOCK_R))
        sum_sumsq_per_feature[grid_sum](x_flat, sum_per_feature, sumsq_per_feature, rows, L, BLOCK_R=BLOCK_R)

        # 2) Compute mean and std per feature using Triton
        mean_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)
        std_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)
        compute_mean_std_per_feature[(L,)](sum_per_feature, sumsq_per_feature, mean_per_feature, std_per_feature, rows, L=L)

        # 3) Compute per-feature threshold using Triton
        thr_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)
        compute_thr_per_feature[(L,)](mean_per_feature, std_per_feature, thr_per_feature, self.multiplier, L=L)

        # 4) Apply sparse ReLU using Triton kernel
        out_fp32 = torch.empty((rows, L), dtype=torch.float32, device=x.device)
        grid_relu = (triton.cdiv(rows, BLOCK_R), L)
        sparse_relu_per_feature[grid_relu](x_flat, thr_per_feature, out_fp32, rows, L, BLOCK_R=BLOCK_R)

        # 5) Cast to bfloat16 using Triton kernel
        out_bf16 = torch.empty((rows, L), dtype=torch.bfloat16, device=x.device)
        BLOCK_SIZE = 4096
        grid_cast = (triton.cdiv(rows * L, BLOCK_SIZE),)
        cast_bf16_kernel[grid_cast](out_fp32, out_bf16, rows * L, BLOCK_SIZE=BLOCK_SIZE)

        # Reshape to [B, S, L]
        out = out_bf16.view(B, S, L)
        return out


def run(*args):
    return ModelNew()(*args)
