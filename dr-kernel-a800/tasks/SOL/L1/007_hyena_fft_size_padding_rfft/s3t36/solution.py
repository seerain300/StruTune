import torch
import triton
import triton.language as tl


@triton.jit
def real_part_copy_kernel(
    src_ptr,          # *f32, pointer to source real part (flattened)
    dst_ptr,          # *f32, pointer to destination real output (flattened)
    N: tl.int32,      # total number of elements to copy
    BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def imag_part_copy_kernel(
    src_ptr,          # *f32, pointer to source imag part (flattened)
    dst_ptr,          # *f32, pointer to destination imag output (flattened)
    N: tl.int32,      # total number of elements to copy
    BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def _copy_real_imag_via_triton(x_complex: torch.Tensor, real_out: torch.Tensor, imag_out: torch.Tensor, block_size: int = 1024):
    """
    Launch Triton kernels to copy real and imaginary parts from a complex tensor to float32 outputs.
    We pass torch.real(x_complex) and torch.imag(x_complex) as float32 sources for Triton kernels.
    """
    N = x_complex.real.numel()
    grid = (triton.cdiv(N, block_size),)
    real_part_copy_kernel[grid](
        x_complex.real,       # source real part
        real_out,             # destination real output
        N,
        BLOCK=block_size,
        num_warps=4,
    )
    imag_part_copy_kernel[grid](
        x_complex.imag,       # source imag part
        imag_out,             # destination imag output
        N,
        BLOCK=block_size,
        num_warps=4,
    )


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect input x of shape (batch, channels, seqlen)
        x = args[0]
        # Cast to float32 for numerical stability
        x_f32 = x.to(torch.float32)

        batch, channels, seqlen = x_f32.shape
        twoL = 2 * seqlen

        # Compute rfft with implicit zero-padding to 2*seqlen along the last dimension
        x_freq_complex = torch.fft.rfft(x_f32, n=twoL)  # complex tensor (B, C, L+1)

        # Normalize by 2*seqlen, matching original code
        x_freq_complex = x_freq_complex / twoL

        # Allocate outputs: real and imag parts, each (B, C, L+1), float32
        real_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Copy real and imaginary parts using Triton kernels
        _copy_real_imag_via_triton(x_freq_complex, real_out, imag_out, block_size=1024)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
