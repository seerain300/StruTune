import torch
import triton
import triton.language as tl


@triton.jit
def sum_sumsq_per_feature(x_ptr,      # *fp32, input flattened [rows, L]
                           out_sum_ptr,  # *fp32, length L
                           out_sumsq_ptr,  # *fp32, length L
                           rows,        # int32, number of rows (B*S)
                           L,           # int32, number of features
                           BLOCK_R: tl.constexpr):
    """
    One Triton program per feature f. Loop over all rows in chunks of BLOCK_R,
    accumulate sum and sumsq for that feature, and write results.
    """
    f = tl.program_id(0)
    if f >= L:
        return

    sum_acc = 0.0
    sumsq_acc = 0.0

    r = 0
    while r < rows:
        for i in range(BLOCK_R):
            row_idx = r + i
            row_mask = row_idx < rows
            # Load x[row_idx, f] as fp32
            val = tl.load(x_ptr + row_idx * L + f, mask=row_mask, other=0.0)
            sum_acc += val
            sumsq_acc += val * val
        r += BLOCK_R

    tl.store(out_sum_ptr + f, sum_acc)
    tl.store(out_sumsq_ptr + f, sumsq_acc)


@triton.jit
def compute_mean_std_per_feature(sum_ptr,     # *fp32, length L
                                 sumsq_ptr,   # *fp32, length L
                                 mean_ptr,    # *fp32, length L
                                 std_ptr,     # *fp32, length L
                                 rows,        # int32
                                 L):          # int32
    """
    Compute mean and std per feature:
      mean = sum / rows
      var = sumsq / rows - mean^2
      std = sqrt(max(var, 0))
    """
    f = tl.program_id(0)
    if f >= L:
        return
    sum_f = tl.load(sum_ptr + f)
    sumsq_f = tl.load(sumsq_ptr + f)
    mean_f = sum_f / rows
    var_f = sumsq_f / rows - mean_f * mean_f
    var_f = tl.maximum(var_f, 0.0)
    std_f = tl.sqrt(var_f)
    tl.store(mean_ptr + f, mean_f)
    tl.store(std_ptr + f, std_f)


@triton.jit
def compute_thr_per_feature(mean_ptr,     # *fp32, length L
                            std_ptr,      # *fp32, length L
                            thr_ptr,      # *fp32, length L
                            multiplier,   # fp32 scalar
                            L):           # int32
    """
    Compute per-feature threshold: thr = mean + std * multiplier
    """
    f = tl.program_id(0)
    if f >= L:
        return
    mean_f = tl.load(mean_ptr + f)
    std_f = tl.load(std_ptr + f)
    thr_f = mean_f + std_f * multiplier
    tl.store(thr_ptr + f, thr_f)


@triton.jit
def sparse_relu_per_feature(x_ptr,       # *fp32, input flattened [rows, L]
                            thr_ptr,     # *fp32, length L
                            out_ptr,     # *fp32, output flattened [rows, L]
                            rows,        # int32
                            L):          # int32
    """
    Apply elementwise ReLU with per-feature threshold:
      out[row, f] = max(x[row, f] - thr[f], 0)
    2D grid: program_id(0) over rows, program_id(1) over features.
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
        # multiplier = ndtri(target_sparsity) precomputed. Keep as Python float.
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

        # 1) Per-feature reduction: sum and sumsq (Triton kernel, one program per feature)
        sum_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)
        sumsq_per_feature = torch.empty(L, dtype=torch.float32, device=x.device)

        sum_sums


def run(*args):
    return ModelNew()(*args)
