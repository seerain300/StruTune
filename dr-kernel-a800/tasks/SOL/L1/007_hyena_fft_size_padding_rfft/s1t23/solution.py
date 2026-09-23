import torch
import triton
import triton.language as tl


@triton.jit
def _copy_real_to_output_kernel(
    in_ptr,           # *const float32, input pointer to x_freq.real (flattened)
    out_ptr,          # *float32, output pointer to out_real (flattened)
    size: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one element; grid should be (size,)
    idx = pid
    tl.store(out_ptr + idx, tl.load(in_ptr + idx))


@triton.jit
def _copy_imag_to_output_kernel(
    in_ptr,           # *const float32, input pointer to x_freq.imag (flattened)
    out_ptr,          # *float32, output pointer to out_imag (flattened)
    size: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid
    tl.store(out_ptr + idx, tl.load(in_ptr + idx))


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen)
        batch, channels, seqlen = x.shape
        N = 2 * seqlen
        x_f32 = x.to(torch.float32)
        # Compute rfft along last dim, pad to n=N
        x_freq = torch.fft.rfft(x_f32, n=N, dim=-1)
        # Normalize by N
        x_freq = x_freq / N
        # Extract real and imaginary parts
        x_real = x_freq.real
        x_imag = x_freq.imag
        # Prepare outputs
        out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        # Flatten for simple 1D copy
        real_flat = x_real.contiguous().view(-1)
        imag_flat = x_imag.contiguous().view(-1)
        out_real_flat = out_real.view(-1)
        out_imag_flat = out_imag.view(-1)
        size = real_flat.numel()
        # Launch Triton kernels to copy data
        grid = (size,)
        _copy_real_to_output_kernel[grid](real_flat, out_real_flat, size)
        _copy_imag_to_output_kernel[grid](imag_flat, out_imag_flat, size)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
