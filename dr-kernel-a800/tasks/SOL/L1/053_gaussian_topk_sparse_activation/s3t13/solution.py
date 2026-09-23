import math
import triton
import triton.language as tl


@triton.jit
def reduce_sum_rows_kernel(x_ptr, sum_ptr, N, L, BLOCK: tl.constexpr):
    # One program per row (pid in [0, N) where N = B*S)
    row = tl.program_id(axis=0)
    acc = 0.0
    # Iterate over features in chunks of BLOCK=1024
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        # Base pointer for this row; x_ptr is a 1D flattened pointer of length N*L
        # index = row * L + offs
        x_idx = row * L + offs
        vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(sum_ptr + row, acc)


@triton.jit
def reduce_sumsq_rows_kernel(x_ptr, sumsq_ptr, N, L, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    acc = 0.0
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        x_idx = row * L + offs
        vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(sumsq_ptr + row, acc)


@triton.jit
def compute_mean_std_threshold_rows_kernel(sum_ptr, sumsq_ptr, threshold_ptr, N, L, rows, multiplier):
    row = tl.program_id(axis=0)
    s = tl.load(sum_ptr + row)
    ss = tl.load(sumsq_ptr + row)
    mean = s / rows
    var = ss / rows - mean * mean
    # Ensure var >= 0 before sqrt (numerical safety)
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    thr = mean + std * multiplier
    tl.store(threshold_ptr + row, thr)


@triton.jit
def sparse_relu_rows_kernel(x_ptr, threshold_ptr, out_ptr, N, L, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    thr = tl.load(threshold_ptr + row)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        x_idx = row * L + offs
        vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        out_vals = tl.maximum(vals - thr, 0.0)
        tl.store(out_ptr + x_idx, out_vals, mask=mask)


@triton.jit
def cast_bf16_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    # Cast float32 to bfloat16 in chunks
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        vals = tl.load(in_ptr + offs, mask=mask, other=0.0)
        # Triton will infer types; ensure out_ptr is bfloat16
        tl.store(out_ptr + offs, vals.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure contiguity and dtype
        x = inputs.contiguous()
        # Flatten to [N_rows, L]
        B, S, L = x.shape
        N_rows = B * S
        x_flat = x.view(N_rows, L).to(torch.float32).reshape(-1)  # [N_rows*L] float32
        total_elems = N_rows * L

        # Allocate accumulators for sums (FP32)
        sum_rows = torch.empty(N_rows, dtype=torch.float32, device=x.device)
        sumsq_rows = torch.empty(N_rows, dtype=torch.float32, device=x.device)
        threshold_rows = torch.empty(N_rows, dtype=torch.float32, device=x.device)

        # 1) Compute per-row sum and sumsq
        grid = (N_rows,)
        reduce_sum_rows_kernel[grid](x_flat, sum_rows, N_rows, L, BLOCK=1024, num_warps=4)
        reduce_sumsq_rows_kernel[grid](x_flat, sumsq_rows, N_rows, L, BLOCK=1024, num_warps=4)

        # 2) Compute per-row mean, std, and threshold
        compute_mean_std_threshold_rows_kernel[grid](
            sum_rows, sumsq_rows, threshold_rows, N_rows, L, float(N_rows), self.target_sparsity
        )

        # 3) Allocate FP32 output and perform ReLU with per-row threshold
        out_flat_fp32 = torch.empty(total_elems, dtype=torch.float32, device=x.device)
        sparse_relu_rows_kernel[grid](
            x_flat, threshold_rows, out_flat_fp32, N_rows, L, BLOCK=1024, num_warps=4
        )

        # 4) Cast to bfloat16 via Triton (forward MUST invoke this kernel; no torch cast in forward)
        out_bf16 = torch.empty(total_elems, dtype=torch.bfloat16, device=x.device)
        grid_cast = (triton.cdiv(total_elems, 4096),)
        cast_bf16_kernel[grid_cast](out_flat_fp32, out_bf16, total_elems, BLOCK=4096, num_warps=4)

        # 5) Reshape back to [B, S, L]
        out = out_bf16.view(B, S, L)
        return out


def run(*args):
    return ModelNew()(*args)
