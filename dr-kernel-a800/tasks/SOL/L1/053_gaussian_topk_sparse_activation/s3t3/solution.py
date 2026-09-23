import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_per_row(
    inp_ptr,              # *fp32, flattened input [rows * L]
    sum_ptr,              # *fp32, per-row sum [rows]
    sumsq_ptr,            # *fp32, per-row sum of squares [rows]
    L: tl.constexpr,      # last-dim length
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    cols = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < L
    base = row_id * L
    vals = tl.load(inp_ptr + base + cols, mask=mask, other=0.0)
    s = tl.sum(vals, axis=0)
    ss = tl.sum(vals * vals, axis=0)
    tl.atomic_add(sum_ptr + row_id, s)
    tl.atomic_add(sumsq_ptr + row_id, ss)


@triton.jit
def compute_mean_std(
    sum_ptr,              # *fp32, per-row sum [rows]
    sumsq_ptr,            # *fp32, per-row sum of squares [rows]
    mean_ptr,             # *fp32, output mean [rows]
    std_ptr,              # *fp32, output std [rows]
    L: tl.constexpr,      # last-dim length
):
    row_id = tl.program_id(0)
    s = tl.load(sum_ptr + row_id)
    ss = tl.load(sumsq_ptr + row_id)
    mean = s / L
    var = ss / L - mean * mean
    std = tl.sqrt(var)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_threshold(
    mean_ptr,             # *fp32, mean [rows]
    std_ptr,              # *fp32, std [rows]
    std_mult_ptr,         # *fp32, 1-element tensor (scalar) on device
    threshold_ptr,        # *fp32, output threshold [rows]
):
    row_id = tl.program_id(0)
    m = tl.load(mean_ptr + row_id)
    sd = tl.load(std_ptr + row_id)
    mult = tl.load(std_mult_ptr)  # scalar
    thr = m + sd * mult
    tl.store(threshold_ptr + row_id, thr)


@triton.jit
def sparse_relu_fp32(
    inp_ptr,              # *fp32, flattened input [rows * L]
    thr_ptr,              # *fp32, threshold [rows]
    out_ptr,              # *fp32, flattened output [rows * L]
    rows: tl.constexpr,   # total rows = B * S
    L: tl.constexpr,      # last-dim length
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    cols = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < L
    base = row_id * L
    inp = tl.load(inp_ptr + base + cols, mask=mask, other=0.0)
    thr = tl.load(thr_ptr + row_id)
    val = tl.maximum(inp - thr, 0.0)
    tl.store(out_ptr + base + cols, val, mask=mask)


@triton.jit
def cast_bf16_kernel(
    in_fp32_ptr,          # *fp32, input [N]
    out_bf16_ptr,         # *bf16, output [N]
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N
    vals = tl.load(in_fp32_ptr + offs, mask=mask, other=0.0)
    vals_bf16 = vals.to(tl.bfloat16)
    tl.store(out_bf16_ptr + offs, vals_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float, std_multiplier: torch.Tensor,
                 block_size_reduce: int = 256, block_size_act: int = 256):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        # std_multiplier must be a 0-dim tensor (scalar) on the same device as inputs.
        if std_multiplier.numel() != 1:
            raise RuntimeError("std_multiplier must be a scalar 0-dim tensor")
        self.std_multiplier = std_multiplier
        self.block_size_reduce = block_size_reduce
        self.block_size_act = block_size_act

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguity (no torch ops here)
        assert inputs.is_cuda, "ModelNew expects CUDA tensor inputs"
        inputs = inputs.contiguous()
        B, S, L = inputs.shape
        rows = B * S
        N = rows * L

        # Flatten and convert to FP32 (data movement, not torch op)
        inp_fp32 = inputs.view(-1).to(torch.float32)

        # Per-row accumulators
        sum_row = torch.zeros(rows, dtype=torch.float32, device=inputs.device)
        sumsq_row = torch.zeros(rows, dtype=torch.float32, device=inputs.device)

        # Reduction: compute per-row sum and sumsq
        grid_reduce = (rows, triton.cdiv(L, self.block_size_reduce))
        reduce_sum_sumsq_per_row[grid_reduce](
            inp_fp32, sum_row, sumsq_row, L, BLOCK_SIZE=self.block_size_reduce, num_warps=4
        )

        # Mean and std per row
        mean_row = torch.empty(rows, dtype=torch.float32, device=inputs.device)
        std_row = torch.empty(rows, dtype=torch.float32, device=inputs.device)
        compute_mean_std[(rows,)](sum_row, sumsq_row, mean_row, std_row, L, num_warps=1)

        # Threshold per row: mean + std * std_multiplier (std_multiplier is a 0-dim tensor on device)
        threshold = torch.empty(rows, dtype=torch.float32, device=inputs.device)
        compute_threshold[(rows,)](mean_row, std_row, self.std_multiplier, threshold, num_warps=1)

        # Elementwise sparse ReLU in FP32: out = max(inp - threshold[row], 0)
        out_fp32 = torch.empty(N, dtype=torch.float32, device=inputs.device)
        grid_act = (rows, triton.cdiv(L, self.block_size_act))
        sparse_relu_fp32[grid_act](
            inp_fp32, threshold, out_fp32, rows, L, BLOCK_SIZE=self.block_size_act, num_warps=4
        )

        # Cast to bfloat16 via Triton kernel (forward MUST invoke this kernel)
        out_bf16 = torch.empty(N, dtype=torch.bfloat16, device=inputs.device)
        grid_cast = (triton.cdiv(N, 1024),)
        cast_bf16_kernel[grid_cast](out_fp32, out_bf16, N, num_warps=8)

        # Reshape to [B, S, L]
        out = out_bf16.view(B, S, L)
        return out


def run(*args):
    return ModelNew()(*args)
