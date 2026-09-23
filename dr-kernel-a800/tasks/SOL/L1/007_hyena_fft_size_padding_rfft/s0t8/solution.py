import torch
import triton
import triton.language as tl


@triton.jit
def cast_to_float32_channel_kernel(in_ptr, out_ptr, n_in, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: cast elements from input dtype to float32 per channel row.
    Assumes 'in_ptr' points to a contiguous 2D view of shape (B*C, seqlen).
    We cast each element to float32 and write to 'out_ptr'.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_in
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    tl.store(out_ptr + offsets, x_f32, mask=mask)


@triton.jit
def normalize_divide_kernel(in_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel: elementwise divide each element in 'in_ptr' by 'scale' and write to 'out_ptr'.
    Assumes float32 input/output.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = x / scale
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-enabled version:
        - Cast input to float32 for channel-major view using Triton kernel.
        - Compute rfft via torch (for correctness).
        - Normalize by 2*seqlen using Triton kernels for real and imaginary parts.
        Returns:
            x_freq_real: float32 tensor of shape (batch, channels, seqlen+1)
            x_freq_imag: float32 tensor of shape (batch, channels, seqlen+1)
        """
        assert x.dim() == 3, "Input must be of shape (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape

        # Ensure CUDA and contiguous layout
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()

        # Cast to float32 per (batch, channel) row using Triton kernel.
        # Reshape to 2D: (B*C, seqlen), then cast each element.
        x_2d = x.view(batch * channels, seqlen)
        x_f32_2d = torch.empty_like(x_2d, dtype=torch.float32)

        n_in = x_2d.numel()
        BLOCK_SIZE_CAST = 1024
        grid_cast = (triton.cdiv(n_in, BLOCK_SIZE_CAST),)
        cast_to_float32_channel_kernel[grid_cast](x_2d, x_f32_2d, n_in, BLOCK_SIZE=BLOCK_SIZE_CAST)

        # Reshape back to (batch, channels, seqlen)
        x_f32 = x_f32_2d.view(batch, channels, seqlen).contiguous()

        # Compute real FFT with n=2*seqlen using torch for correctness
        N = 2 * seqlen
        x_freq = torch.fft.rfft(x_f32, n=N)  # complex output

        # Normalize by 2*seqlen
        scale = float(N)

        # Extract real and imaginary parts and normalize using Triton kernels
        x_freq_real = x_freq.real.contiguous()  # float32
        x_freq_imag = x_freq.imag.contiguous()  # float32

        n_bins = seqlen + 1
        out_real = torch.empty_like(x_freq_real)
        out_imag = torch.empty_like(x_freq_imag)

        n_real = batch * channels * n_bins
        n_imag = n_real

        BLOCK_SIZE_DIV = 1024
        grid_real = (triton.cdiv(n_real, BLOCK_SIZE_DIV),)
        grid_imag = (triton.cdiv(n_imag, BLOCK_SIZE_DIV),)

        # Launch normalization Triton kernels
        normalize_divide_kernel[grid_real](x_freq_real, out_real, n_real, scale, BLOCK_SIZE=BLOCK_SIZE_DIV)
        normalize_divide_kernel[grid_imag](x_freq_imag, out_imag, n_imag, scale, BLOCK_SIZE=BLOCK_SIZE_DIV)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
