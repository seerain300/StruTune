import torch
import triton
import triton.language as tl


@triton.jit
def row_reduce_sum_sumsq_kernel(X_ptr, sum_out_ptr, sumsq_out_ptr,
                                 F: tl.int32,
                                 BLOCK_SIZE: tl.constexpr):
    """
    Each program handles one row. It iterates over the feature dimension in tiles
    and accumulates sum and sum of squares for that row.
    Assumes input is laid out as [rows, F] with rows = B * S and X_ptr points to such data.
    """
    pid = tl.program_id(axis=0)  # row index
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate across features in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        offs = pid * F + cols
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    tl.store(sum_out_ptr + pid, sum_val)
    tl.store(sumsq_out_ptr + pid, sum_sq)


@triton.jit
def apply_relu_threshold_kernel(X_ptr, Y_ptr, mean_ptr, std_ptr, multiplier,
                                F: tl.int32,
                                BLOCK_SIZE: tl.constexpr):
    """
    For each row (program_id=axis=0), load mean[row] and std[row], compute threshold =
    mean + std * multiplier, then for each feature col: y = max(0, X[row, col] - threshold).
    Stores y to Y_ptr.
    Assumes X and Y are laid out as [rows, F].
    mean_ptr, std_ptr are 1D tensors of length rows.
    multiplier is a scalar (Python float passed in).
    """
    pid = tl.program_id(axis=0)  # row index
    threshold = tl.load(mean_ptr + pid) + tl.load(std_ptr + pid) * multiplier

    # Loop over features in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        offs = pid * F + cols
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size_reduce=1024, block_size_act=1024):
        super().__init__()
        self.block_size_reduce = block_size_reduce
        self.block_size_act = block_size_act

    @torch.no_grad()
    def run(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Gaussian-based top-k sparse activation (same behavior as the original).
        - Computes per-row mean and std across the feature dimension (unbiased=False).
        - Computes inverse standard normal CDF of target_sparsity (scalar multiplier).
        - Applies y = max(0, x - (mean + std * multiplier)) per element.
        - Returns tensor with same shape as inputs, cast to bfloat16.
        """
        # Handle edge case: no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Flatten to [rows, F], rows = B * S
        x = inputs
        B, S, F = x.shape
        rows = B * S

        # Ensure contiguous and compute in float32
        x32 = x.to(torch.float32).contiguous()

        # 1) Per-row reduction: sum and sum of squares
        sum_buf = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq_buf = torch.empty(rows, dtype=torch.float32, device=x32.device)

        x2d = x32.view(rows, F)
        row_reduce_sum_sumsq_kernel[(rows,)](
            x2d, sum_buf, sumsq_buf,
            F=F,
            BLOCK_SIZE=self.block_size_reduce,
            num_warps=8,
            num_stages=2,
        )

        # 2) Compute mean and std per row (PyTorch, on GPU)
        mean = sum_buf / F
        var = sumsq_buf / F - mean * mean
        # Avoid negative due to numerical error
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        # 3) Compute inverse standard normal CDF for the scalar target_sparsity in PyTorch
        # Using torch's built-in qnorm for reliability and exactness.
        # torch.distributions.normal has .icdf; ensure availability.
        try:
            from torch.distributions import normal
            std_multiplier = normal.Normal(0, 1).icdf(torch.tensor(target_sparsity, dtype=torch.float32, device=x.device))
        except Exception:
            # Fallback to stats: norm.ppf if available
            std_multiplier = torch.tensor(0.0, dtype=torch.float32, device=x.device)  # default, will be overwritten
            try:
                import scipy.stats
                multiplier_value = float(scipy.stats.norm.ppf(target_sparsity))
                std_multiplier = torch.tensor(multiplier_value, dtype=torch.float32, device=x.device)
            except Exception:
                # Fallback: use a reasonable default if icdf fails
                std_multiplier = torch.tensor(0.0, dtype=torch.float32, device=x.device)

        # Extract Python float from 1-element tensor
        multiplier_value = float(std_multiplier.item())

        # 4) Apply activation in Triton
        y = torch.empty(rows * F, dtype=torch.float32, device=x32.device)
        apply_relu_threshold_kernel[(rows,)](
            x2d,
            y,
            mean,  # 1D tensor of size rows
            std,   # 1D tensor of size rows
            multiplier_value,  # scalar passed in
            F=F,
            BLOCK_SIZE=self.block_size_act,
            num_warps=8,
            num_stages=2,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
