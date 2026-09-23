import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def subtract_relu_kernel_2d(input_ptr, threshold_ptr, output_ptr,
                            inter_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton 2D kernel:
      - axis 0: over rows, where each row corresponds to a (batch, seq) pair. Total rows = batch * seq.
      - axis 1: over blocks along the last dimension of size BLOCK_SIZE.
    For each program instance, process a block of the last dimension for one row:
      inp = load(input_ptr + base + cols)
      thr = load(threshold_ptr + row_id)  # scalar threshold for this row
      out = max(inp - thr, 0)
      store(output_ptr + base + cols, out)
    """
    row_id = tl.program_id(axis=0)  # 0 .. batch*seq - 1
    col_block = tl.program_id(axis=1)

    base = row_id * inter_size
    offs = base + col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < base + inter_size  # always true except for the last partial block

    inp = tl.load(input_ptr + offs, mask=mask, other=0.0)
    thr = tl.load(threshold_ptr + row_id, mask=True, other=0.0)  # scalar per row

    out = inp - thr
    out = tl.maximum(out, 0.0)

    tl.store(output_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        """
        Triton-optimized version of the original run function.
        Expects a single input tensor of shape [batch_size, seq_len, intermediate_size].
        """
        assert len(args) == 1, "ModelNew.forward expects a single tensor input."
        inputs = args[0]

        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "Input must be on CUDA device for Triton."
        inputs = inputs.contiguous()

        # Compute statistics in FP32
        inputs_f32 = inputs.to(torch.float32)
        mean = torch.mean(inputs_f32, dim=-1, keepdim=True)   # [B, S, 1]
        std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)  # [B, S, 1]

        # target_sparsity: if not provided, default to 0.0
        target_sparsity = 0.0
        if len(args) > 1 and isinstance(args[1], (float, torch.Tensor)):
            target_sparsity = float(args[1]) if isinstance(args[1], torch.Tensor) else float(args[1])

        # Compute inverse normal CDF (quantile) for the target sparsity using A&S approximation.
        # This is a single scalar; not a bottleneck.
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
            # Small p: use tail approximation
            q = torch.sqrt(-2.0 * torch.log(torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)))
            std_multiplier = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        elif target_sparsity >= p_high:
            # Large p: use complementary tail approximation
            q = torch.sqrt(-2.0 * torch.log(torch.tensor(1.0 - target_sparsity, dtype=torch.float32, device=inputs.device)))
            std_multiplier = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        else:
            # Central region
            q = (target_sparsity - 0.5)
            r = q * q
            num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
            std_multiplier = num / den

        std_multiplier = float(std_multiplier.item())  # scalar

        # Compute per-row threshold: cutoff_threshold = mean + std * std_multiplier
        # shapes: mean, std: [B, S, 1]; std_multiplier: scalar
        cutoff_threshold = mean + std * std_multiplier  # [B, S, 1]

        # Prepare flattened pointers for Triton
        B, S, I = inputs_f32.shape
        inp_flat = inputs_f32.reshape(-1)  # [B*S*I]
        N = inp_flat.numel()

        # Threshold is per row; flatten to [B*S]
        threshold_flat = cutoff_threshold.reshape(B * S).contiguous()  # [B*S]

        # Output FP32 buffer
        out_flat = torch.empty(N, dtype=torch.float32, device=inputs.device)

        # Launch Triton kernel with 2D grid
        BLOCK_SIZE = 1024
        grid = (B * S, triton.cdiv(I, BLOCK_SIZE))
        subtract_relu_kernel_2d[grid](inp_flat, threshold_flat, out_flat, I, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)

        # Reshape back to [B, S, I] and cast to bfloat16
        out_f32 = out_flat.reshape(B, S, I)
        out_bf16 = out_f32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
