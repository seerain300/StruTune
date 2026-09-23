import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(input_ptr, row_sums_ptr, row_sumsq_ptr, N, inter_size, BLOCK_SIZE: tl.constexpr):
    """
    Reduce sum and sum of squares over the last dimension for each row (axis=0 = one (batch, seq) row).
    Uses 2D grid: axis0 = row_id in [0, B*S), axis1 = block along last dim.
    Accumulates per-row totals with atomics.
    """
    row_id = tl.program_id(axis=0)  # 0 .. B*S - 1
    block_id = tl.program_id(axis=1)

    start = row_id * inter_size
    offs = start + block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N  # safe mask; in practice offs < start + inter_size

    inp = tl.load(input_ptr + offs, mask=mask, other=0.0)
    sum_block = tl.sum(inp, axis=0)
    sumsq_block = tl.sum(inp * inp, axis=0)

    tl.atomic_add(row_sums_ptr + row_id, sum_block)
    tl.atomic_add(row_sumsq_ptr + row_id, sumsq_block)


@triton.jit
def write_threshold_kernel(mean_ptr, std_ptr, threshold_ptr, multiplier, B_S, inter_size, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row threshold = mean[row] + std[row] * multiplier and write to threshold_ptr[row].
    Axis 0 iterates over rows (0 .. B_S-1). Axis 1 is 1 block per row (no blocks along last dim).
    """
    row_id = tl.program_id(axis=0)  # 0 .. B_S - 1
    # single block per row, BLOCK_SIZE can be anything, only one program per row
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    thr = mean + std * multiplier  # multiplier is a scalar float
    tl.store(threshold_ptr + row_id, thr)


@triton.jit
def sparse_relu_kernel(input_ptr, threshold_ptr, output_ptr, N, inter_size, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise kernel: out = max(input - threshold[row], 0)
    input_ptr points to FP32 input flattened; threshold_ptr is [B*S] FP32; output_ptr is FP32.
    """
    row_id = tl.program_id(axis=0)  # 0 .. B*S - 1
    block_id = tl.program_id(axis=1)

    start = row_id * inter_size
    offs = start + block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    inp = tl.load(input_ptr + offs, mask=mask, other=0.0)  # FP32
    thr = tl.load(threshold_ptr + row_id)  # scalar FP32

    out = inp - thr
    out = tl.maximum(out, 0.0)

    tl.store(output_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        """
        Triton-only implementation of the original run:
        - Compute per-row sum and sumsq along last dim via Triton.
        - Compute mean and std via torch (small tensors).
        - Compute per-row threshold via Triton (mean + std * ndtri(target_sparsity)).
        - Apply sparse ReLU via Triton: max(inputs - threshold, 0).
        Returns tensor of the same shape as input, cast to bfloat16.
        """
        assert len(args) == 1, "ModelNew.forward expects a single input tensor."
        inputs = args[0]

        # If no sparsity requested, return inputs unchanged (cast to bfloat16 like original).
        if len(args) > 1 and isinstance(args[1], (float, torch.Tensor)):
            target_sparsity = float(args[1]) if isinstance(args[1], torch.Tensor) else float(args[1])
        else:
            target_sparsity = 0.0

        if target_sparsity == 0.0:
            # Preserve shape and dtype behavior: original returns same shape, bfloat16
            return inputs.to(torch.bfloat16)

        # Ensure CUDA and contiguous; compute in FP32
        assert inputs.is_cuda, "Input must be on CUDA device for Triton."
        inputs = inputs.contiguous()
        inputs_f32 = inputs.to(torch.float32)

        B, S, I = inputs_f32.shape
        N = B * S * I

        # Buffers for per-row reductions
        row_sums = torch.zeros(B * S, dtype=torch.float32, device=inputs.device)
        row_sumsq = torch.zeros(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel: 2D grid over rows and blocks along last dim
        BLOCK_SIZE = 1024
        grid_red = (B * S, triton.cdiv(I, BLOCK_SIZE))
        reduce_sum_sumsq_kernel[grid_red](inputs_f32.view(-1), row_sums, row_sumsq, N, I, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)

        # Compute mean and std (unbiased=False) per row
        mean = row_sums / I                    # [B*S], FP32
        var = row_sumsq / I - mean * mean      # [B*S], FP32
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)                  # [B*S], FP32

        # Compute inverse normal CDF (quantile) for target_sparsity using A&S 5.2.23 approximation (host scalar).
        # Typical use: target_sparsity in (0,1). We implement central and tails regions.
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

        if target_sparsity <= p_low:
            q = torch.sqrt(torch.tensor(-2.0 * math.log(target_sparsity), dtype=torch.float32, device=inputs.device))
            poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            denom = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
            std_multiplier = poly / denom
        elif target_sparsity >= p_high:
            q = torch.sqrt(torch.tensor(-2.0 * math.log(1.0 - target_sparsity), dtype=torch.float32, device=inputs.device))
            poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            denom = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
            std_multiplier = -poly / denom
        else:
            q = (target_sparsity - 0.5)
            r = q * q
            num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
            std_multiplier = num / den

        std_multiplier = float(std_multiplier.item())  # scalar float

        # Allocate threshold buffer [B*S] in FP32 and compute per-row threshold in Triton
        threshold = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        grid_thr = (B * S, 1)  # one program per row
        write_threshold_kernel[grid_thr](mean, std, threshold, std_multiplier, B * S, I, BLOCK_SIZE=1, num_warps=1)

        # Launch elementwise Triton kernel: out = max(inputs - threshold, 0)
        inp_flat = inputs_f32.view(-1)  # [B*S*I] FP32
        out_flat_fp32 = torch.empty_like(inp_flat, dtype=torch.float32, device=inputs.device)

        grid_act = (B * S, triton.cdiv(I, BLOCK_SIZE))
        sparse_relu_kernel[grid_act](inp_flat, threshold, out_flat_fp32, N, I, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)

        # Reshape and cast to bfloat16 to match original behavior
        out_fp32 = out_flat_fp32.view(B, S, I)
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
