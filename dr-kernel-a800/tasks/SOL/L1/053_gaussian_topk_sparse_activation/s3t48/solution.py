import torch
import triton
import triton.language as tl


# Kernel 1: reduce per-feature sum and sum of squares across all rows
@triton.jit
def reduce_feature_sum_sumsq_kernel(
    x_ptr,                 # *float32, flattened [rows, L]
    sum_ptr,               # *float32, length L
    sumsq_ptr,             # *float32, length L
    L: tl.constexpr,       # number of features
    rows,                  # int32 runtime: total number of rows (B*S)
    BLOCK_ROWS: tl.constexpr,  # chunk size for rows
):
    pid_f = tl.program_id(axis=0)  # feature id
    # Accumulators in FP32
    total_sum = 0.0
    total_sumsq = 0.0
    # Iterate over rows in chunks
    r_start = 0
    while r_start < rows:
        r_idx = r_start + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
        mask = r_idx < rows
        # Compute pointers for this chunk: x[r_idx, pid_f]
        # For flattened [rows, L], linear index = r_idx * L + pid_f
        x_vals = tl.load(x_ptr + r_idx * L + pid_f, mask=mask, other=0.0)
        total_sum += tl.sum(x_vals, axis=0)
        total_sumsq += tl.sum(x_vals * x_vals, axis=0)
        r_start += BLOCK_ROWS
    # Atomically add into global sums
    tl.atomic_add(sum_ptr + pid_f, total_sum)
    tl.atomic_add(sumsq_ptr + pid_f, total_sumsq)


# Kernel 2: compute mean and std per feature (from sums and sumsqs)
@triton.jit
def compute_mean_std_kernel(
    sum_ptr,               # *float32, length L
    sumsq_ptr,             # *float32, length L
    mean_ptr,              # *float32, length L
    std_ptr,               # *float32, length L
    L: tl.constexpr,       # number of features
    rows,                  # int32 runtime
):
    pid_f = tl.program_id(axis=0)
    s = tl.load(sum_ptr + pid_f)
    ss = tl.load(sumsq_ptr + pid_f)
    mean = s / rows
    var = ss / rows - mean * mean
    # numerical safeguard
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(mean_ptr + pid_f, mean)
    tl.store(std_ptr + pid_f, std)


# Kernel 3: compute threshold thr[f] = mean[f] + std[f] * std_multiplier
@triton.jit
def compute_threshold_kernel(
    mean_ptr,              # *float32, length L
    std_ptr,               # *float32, length L
    thr_ptr,               # *float32, length L
    std_multiplier,        # float32 scalar
    L: tl.constexpr,
):
    pid_f = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + pid_f)
    std = tl.load(std_ptr + pid_f)
    thr = mean + std * std_multiplier
    tl.store(thr_ptr + pid_f, thr)


# Kernel 4: apply sparse ReLU per element: y = max(x - thr[f], 0)
@triton.jit
def sparse_relu_kernel(
    x_ptr,                 # *float32, flattened [rows, L]
    thr_ptr,               # *float32, length L
    out_ptr,               # *float32, flattened [rows, L]
    rows,                  # int32 runtime
    L: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)  # row id
    pid_f = tl.program_id(axis=1)    # feature id
    # Bounds check
    if pid_row >= rows:
        return
    x_val = tl.load(x_ptr + pid_row * L + pid_f)
    thr = tl.load(thr_ptr + pid_f)
    y = x_val - thr
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_ptr + pid_row * L + pid_f, y)


# Kernel 5: cast FP32 output to BF16
@triton.jit
def cast_bf16_kernel(
    in_ptr_fp32,           # *float32, flattened [rows, L]
    out_ptr_bf16,          # *bfloat16, flattened [rows, L]
    rows,                  # int32 runtime
    L: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)  # row id
    pid_f = tl.program_id(axis=1)    # feature id
    if pid_row >= rows:
        return
    val = tl.load(in_ptr_fp32 + pid_row * L + pid_f)
    # cast to bfloat16
    val_bf16 = tl.cast(val, tl.bfloat16)
    tl.store(out_ptr_bf16 + pid_row * L + pid_f, val_bf16)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float, block_rows: int = 1024, block_f: int = 128):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        self.block_rows = int(block_rows)
        self.block_f = int(block_f)  # used for grid in ReLU/cast kernels; not strictly required

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous and float32 for robust Triton loads
        x = inputs.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        B, S, L = x.shape
        rows = B * S

        # Flatten to [rows, L] for reduction
        x_flat = x.view(rows, L).contiguous()

        # Allocate global accumulators
        sum_buf = torch.zeros(L, dtype=torch.float32, device=x.device)
        sumsq_buf = torch.zeros(L, dtype=torch.float32, device=x.device)
        mean_buf = torch.empty(L, dtype=torch.float32, device=x.device)
        std_buf = torch.empty(L, dtype=torch.float32, device=x.device)
        thr_buf = torch.empty(L, dtype=torch.float32, device=x.device)

        # Prepare std multiplier as device scalar (float32)
        std_multiplier = torch.tensor(self.target_sparsity, dtype=torch.float32, device=x.device)

        # 1) Reduce per-feature sum and sumsq
        grid_reduce = (L,)
        reduce_feature_sum_sumsq_kernel[grid_reduce](
            x_flat,
            sum_buf, sumsq_buf,
            L,
            rows,
            BLOCK_ROWS=self.block_rows,
        )

        # 2) Compute mean and std per feature
        grid_meanstd = (L,)
        compute_mean_std_kernel[grid_meanstd](
            sum_buf, sumsq_buf,
            mean_buf, std_buf,
            L,
            rows,
        )

        # 3) Compute per-feature threshold thr = mean + std * std_multiplier
        grid_thr = (L,)
        compute_threshold_kernel[grid_thr](
            mean_buf, std_buf,
            thr_buf,
            std_multiplier,
            L,
        )

        # 4) Apply sparse ReLU: y = max(x - thr[f], 0) in FP32
        out_fp32 = torch.empty((rows, L), dtype=torch.float32, device=x.device)
        grid_relu = (rows, L)
        sparse_relu_kernel[grid_relu](
            x_flat, thr_buf, out_fp32,
            rows, L,
        )

        # 5) Cast to bfloat16 in Triton
        out_bf16 = torch.empty((rows, L), dtype=torch.bfloat16, device=x.device)
        cast_bf16_kernel[grid_relu](
            out_fp32, out_bf16,
            rows, L,
        )

        # Reshape back to [B, S, L]
        out_bf16 = out_bf16.view(B, S, L)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
