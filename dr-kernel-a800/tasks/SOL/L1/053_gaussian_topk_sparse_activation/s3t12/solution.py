import math
import triton
import triton.language as tl


@triton.jit
def add_to_row_sum_sumsq_kernel(x_ptr, sum_ptr, sumsq_ptr, total_rows, L, row_id):
    # For each program instance (over flattened idx), if idx belongs to row row_id, add to sum[row_id] and sumsq[row_id].
    acc_sum = 0.0
    acc_sumsq = 0.0
    # We loop over idx using a vectorized offset; Triton will handle iteration. We don't rely on while here.
    # Instead, we use masks to add only elements belonging to this row.
    # Note: Triton does not support "break" in loops; we rely on mask to make loads/stores inactive.
    # However, Triton requires a compile-time loop bound; we emulate by iterating over total_rows * L with a fixed
    # BLOCK and use mask to target row_id. This pattern is valid in Triton.
    # Define a large block size to cover typical vectors.
    BLOCK = 4096
    for start in range(0, total_rows * L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        # Determine if this offs belongs to row row_id
        row_idx = offs // L  # elementwise integer division: feature index -> row
        mask_row = row_idx == row_id
        mask = offs < (total_rows * L) & mask_row
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(vals, axis=0)
        acc_sumsq += tl.sum(vals * vals, axis=0)

    # Atomically add to per-row accumulators (use float32 pointers). If no elements belong, mask zeros add nothing.
    tl.atomic_add(sum_ptr + row_id, acc_sum)
    tl.atomic_add(sumsq_ptr + row_id, acc_sumsq)


@triton.jit
def compute_mean_std_threshold_rows_kernel(sum_ptr, sumsq_ptr, threshold_ptr, total_rows, L, multiplier):
    # One program per row
    row = tl.program_id(axis=0)
    sum_row = tl.load(sum_ptr + row)
    sumsq_row = tl.load(sumsq_ptr + row)
    L_f = tl.full((), L, tl.float32)
    mean = sum_row / L_f
    var = sumsq_row / L_f - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    thr = mean + std * multiplier
    tl.store(threshold_ptr + row, thr)


@triton.jit
def sparse_relu_rows_kernel(x_ptr, threshold_ptr, out_ptr, total_rows, L):
    # One program per row
    row = tl.program_id(axis=0)
    thr = tl.load(threshold_ptr + row)
    for start in range(0, L, 4096):
        offs = start + tl.arange(0, 4096)
        idx = row * L + offs
        mask = offs < L
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + idx, y, mask=mask)


@triton.jit
def cast_bf16_kernel(in_fp32_ptr, out_bf16_ptr, N, BLOCK: tl.constexpr):
    # Cast FP32 to BF16 elementwise
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_fp32_ptr + offs, mask=mask, other=0.0)
    # Store to BF16 pointer; Triton will convert appropriately.
    tl.store(out_bf16_ptr + offs, x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        # Compute ndtri(target_sparsity) approximation as a Python float (no torch in forward).
        # For p in (0,1), a simple approximation: for p <= 0.5, use -sqrt(-2*log(p)); else use sqrt(-2*log(1-p)).
        if self.target_sparsity <= 0.5:
            self.multiplier = -math.sqrt(-2.0 * math.log(self.target_sparsity))
        else:
            self.multiplier = math.sqrt(-2.0 * math.log(1.0 - self.target_sparsity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, L], float32, on CUDA device
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        B, S, L = x.shape
        total_rows = B * S
        device = x.device

        # Flatten
        x_flat = x.view(-1)  # [total_rows * L]

        # 1) Per-row sums and sumsqs using Triton kernels (no torch ops)
        sum_rows = torch.zeros(total_rows, dtype=torch.float32, device=device)
        sumsq_rows = torch.zeros(total_rows, dtype=torch.float32, device=device)

        # Launch one program per row; within kernel, each program adds its row's contributions
        grid_rows = (total_rows,)
        add_to_row_sum_sumsq_kernel[grid_rows](x_flat, sum_rows, sumsq_rows, total_rows, L, total_rows * L, num_warps=4)

        # 2) Compute per-row mean, std, threshold
        threshold_rows = torch.empty(total_rows, dtype=torch.float32, device=device)
        compute_mean_std_threshold_rows_kernel[grid_rows](
            sum_rows, sumsq_rows, threshold_rows, total_rows, L, self.multiplier, num_warps=1
        )

        # 3) Sparse ReLU per row
        y_flat = torch.empty(total_rows * L, dtype=torch.float32, device=device)
        sparse_relu_rows_kernel[grid_rows](
            x_flat, threshold_rows, y_flat, total_rows, L, num_warps=4
        )

        # 4) Cast to bfloat16 using Triton (forward must invoke this kernel)
        y_bf16 = torch.empty(total_rows * L, dtype=torch.bfloat16, device=device)
        grid_cast = (triton.cdiv(total_rows * L, 4096),)
        cast_bf16_kernel[grid_cast](y_flat, y_bf16, total_rows * L, BLOCK=4096, num_warps=4)

        # Reshape back to [B, S, L]
        y = y_bf16.view(B, S, L)
        return y


def run(*args):
    return ModelNew()(*args)
