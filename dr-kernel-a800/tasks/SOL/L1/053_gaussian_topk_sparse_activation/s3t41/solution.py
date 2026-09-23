import torch
import triton
import triton.language as tl


@triton.jit
def sum_sumsq_per_feature_kernel(
    x_ptr,                  # *fp32, input flattened as [rows, L]
    sum_out_ptr,            # *fp32, output per-feature sum (length L)
    sumsq_out_ptr,          # *fp32, output per-feature sum of squares (length L)
    rows,                   # int32, total number of rows B*S
    L,                      # int32, number of features
    BLOCK_F: tl.constexpr,  # chunk size over features
    BLOCK_R: tl.constexpr,  # chunk size over rows
):
    """
    2D grid over feature chunks and row chunks.
    Each program handles one feature chunk and one row chunk:
      - loads tile [BLOCK_R, BLOCK_F]
      - computes partial sum and sumsq over rows in this chunk
      - atomically adds into global per-feature accumulators
    """
    pid_f = tl.program_id(0)
    pid_r = tl.program_id(1)

    feature_offsets = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    row_offsets = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)

    mask_f = feature_offsets < L
    mask_r = row_offsets < rows

    # Build 2D pointers: [BLOCK_R, BLOCK_F]
    ptrs = x_ptr + row_offsets[:, None] * L + feature_offsets[None, :]
    mask = mask_r[:, None] & mask_f[None, :]

    tile = tl.load(ptrs, mask=mask, other=0.0)

    sum_acc = tl.sum(tile, axis=0)         # shape [BLOCK_F]
    sumsq_acc = tl.sum(tile * tile, axis=0)  # shape [BLOCK_F]

    # Atomically add partials to per-feature accumulators
    tl.atomic_add(sum_out_ptr + feature_offsets, sum_acc, mask=mask_f)
    tl.atomic_add(sumsq_out_ptr + feature_offsets, sumsq_acc, mask=mask_f)


@triton.jit
def compute_mean_std_per_feature_kernel(
    sum_ptr,                # *fp32, per-feature sum (length L)
    sumsq_ptr,              # *fp32, per-feature sum of squares (length L)
    mean_ptr,               # *fp32, output mean (length L)
    std_ptr,                # *fp32, output std (length L)
    rows,                   # int32
    L,                      # int32
):
    """
    Compute mean and std per feature:
      mean = sum / rows
      var = sumsq / rows - mean^2
      std = sqrt(max(var, 0))
    Grid: (L,)
    """
    pid = tl.program_id(0)
    if pid >= L:
        return
    sum_f = tl.load(sum_ptr + pid)
    sumsq_f = tl.load(sumsq_ptr + pid)
    mean = sum_f / rows
    var = sumsq_f / rows - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_thr_per_feature_kernel(
    mean_ptr,               # *fp32, per-feature mean (length L)
    std_ptr,                # *fp32, per-feature std (length L)
    thr_ptr,                # *fp32, output per-feature threshold (length L)
    multiplier,             # fp32 scalar, ndtri(target_sparsity)
    L,                      # int32
):
    """
    Compute threshold per feature:
      thr = mean + std * multiplier
    Grid: (L,)
    """
    pid = tl.program_id(0)
    if pid >= L:
        return
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    thr = mean + std * multiplier
    tl.store(thr_ptr + pid, thr)


@triton.jit
def sparse_relu_per_feature_kernel(
    x_ptr,                  # *fp32, input [rows, L]
    thr_ptr,                # *fp32, per-feature threshold [L]
    out_ptr,                # *fp32, output [rows, L]
    rows,                   # int32
    L,                      # int32
):
    """
    Apply sparse ReLU per feature:
      out[row, f] = max(x[row, f] - thr[f], 0)
    Grid: (rows, L)
    """
    pid_r = tl.program_id(0)
    pid_f = tl.program_id(1)
    if pid_r >= rows or pid_f >= L:
        return
    val = tl.load(x_ptr + pid_r * L + pid_f)
    thr = tl.load(thr_ptr + pid_f)
    out_val = tl.maximum(val - thr, 0.0)
    tl.store(out_ptr + pid_r * L + pid_f, out_val)


@triton.jit
def cast_bf16_kernel(in_ptr,  # *fp32
                     out_ptr,  # *bf16
                     numel,    # int32
                     BLOCK_SIZE: tl.constexpr):
    """
    Cast fp32 to bf16 elementwise.
    Grid: (ceil_div(numel, BLOCK_SIZE),)
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
        # multiplier = ndtri(target_sparsity) provided; no torch ops in forward.
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

        # 1) Per-feature accumulators for sum and sumsq
        sum_per_feature = torch.zeros(L, dtype=torch.float32, device=x.device)
        sumsq_per_feature = torch.zeros(L, dtype=torch.float32, device=x.device)

        # 2) Launch Triton kernel to compute partial sums/sumsq across chunks
        BLOCK_F = 128
        BLOCK_R = 1024
        grid_f = triton.cdiv(L, BLOCK_F)
        grid_r = triton.cdiv(rows, BLOCK_R)
        sum_sumsq_per_feature_kernel[(grid_f, grid_r)](
            x_flat, sum_per_feature, sumsq_per_feature, rows, L,
            BLOCK_F=BLOCK_F, BLOCK_R=BLOCK_R,
        )

        # 3) Compute mean and std per feature in Triton kernel
        mean_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)
        std_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)
        compute_mean_std_per_feature_kernel[(L,)](sum_per_feature, sumsq_per_feature,
                                                 mean_per_feature, std_per_feature, rows, L)

        # 4) Compute per-feature threshold in Triton kernel
        thr_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)
        compute_thr_per_feature_kernel[(L,)](mean_per_feature, std_per_feature,
                                             thr_per_feature, self.multiplier, L)

        # 5) Apply sparse ReLU using Triton kernel
        out_fp32 = torch.empty((rows, L), dtype=torch.float32, device=x.device)
        sparse_relu_per_feature_kernel[(rows, L)](x_flat, thr_per_feature,
                                                  out_fp32, rows, L)

        # 6) Cast to bfloat16 using Triton kernel
        out_bf16 = torch.empty((rows, L), dtype=torch.bfloat16, device=x.device)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(rows * L, BLOCK_SIZE),)
        cast_bf16_kernel[grid](out_fp32, out_bf16, rows * L, BLOCK_SIZE=BLOCK_SIZE)

        # Reshape to [B, S, L]
        out = out_bf16.view(B, S, L)
        return out


def run(*args):
    return ModelNew()(*args)
