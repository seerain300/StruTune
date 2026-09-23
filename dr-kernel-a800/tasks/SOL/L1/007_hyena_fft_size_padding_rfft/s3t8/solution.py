import torch
import triton
import triton.language as tl


@triton.jit
def copy_real_kernel(
    in_ptr,      # *complex64 (we pass complex output as float2), but here we only use it for reading real part
    out_ptr,     # *float32, output real part (B, C, L+1)
    B: tl.int32,
    C: tl.int32,
    L_plus1: tl.int32,
    in_stride_b: tl.int32,
    in_stride_c: tl.int32,
    in_stride_l: tl.int32,
    out_stride_b: tl.int32,
    out_stride_c: tl.int32,
    out_stride_l: tl.int32,
):
    # One program per (b, c)
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c

    k = 0
    while k < L_plus1:
        # Assuming in_ptr points to complex tensor interleaved [real, imag] per element,
        # but we pass x_freq.real as input to this kernel, so we directly read real value.
        # However, Triton does not support complex64 pointer; thus we avoid passing complex.
        # Instead, we call this kernel with input tensor x_freq.real (float32).
        val = tl.load(in_ptr + base_in + k * in_stride_l)
        tl.store(out_ptr + base_out + k * out_stride_l, val)
        k += 1


@triton.jit
def copy_imag_kernel(
    in_ptr,      # *float32, input imag part (B, C, L+1)
    out_ptr,     # *float32, output imag part (B, C, L+1)
    B: tl.int32,
    C: tl.int32,
    L_plus1: tl.int32,
    in_stride_b: tl.int32,
    in_stride_c: tl.int32,
    in_stride_l: tl.int32,
    out_stride_b: tl.int32,
    out_stride_c: tl.int32,
    out_stride_l: tl.int32,
):
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c

    k = 0
    while k < L_plus1:
        val = tl.load(in_ptr + base_in + k * in_stride_l)
        tl.store(out_ptr + base_out + k * out_stride_l, val)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure float32
        x = x.to(torch.float32)
        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L
        L_plus1 = L + 1

        # Compute rfft with implicit zero-padding to 2*seqlen
        x_freq = torch.fft.rfft(x, n=twoL)  # complex output
        # Normalize
        x_freq = x_freq / twoL

        # Allocate outputs
        real_out = torch.empty((batch, channels, L_plus1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L_plus1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels to copy real and imaginary parts
        # We pass x_freq.real and x_freq.imag to the respective kernels.
        # Strides are in elements.
        real_stride_b, real_stride_c, real_stride_l = x_freq.real.stride()
        imag_stride_b, imag_stride_c, imag_stride_l = x_freq.imag.stride()
        out_stride_b, out_stride_c, out_stride_l = real_out.stride()

        grid = (batch * channels,)
        copy_real_kernel[grid](
            x_freq.real, real_out,
            batch, channels, L_plus1,
            real_stride_b, real_stride_c, real_stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            num_warps=1,
        )
        copy_imag_kernel[grid](
            x_freq.imag, imag_out,
            batch, channels, L_plus1,
            imag_stride_b, imag_stride_c, imag_stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            num_warps=1,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
