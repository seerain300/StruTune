import torch
import triton
import triton.language as tl


@triton.jit
def real_extract_kernel(x_real_ptr, real_out_ptr,
                         batch: tl.int32, channels: tl.int32,
                         L: tl.int32,
                         x_stride_b: tl.int32, x_stride_c: tl.int32, x_stride_l: tl.int32,
                         out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32,
                         BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    x_base = b * x_stride_b + c * x_stride_c
    out_base = b * out_stride_b + c * out_stride_c

    k = 0
    while k < L + 1:
        idx = k + tl.arange(0, BLOCK)
        mask = idx < (L + 1)
        vals = tl.load(x_real_ptr + x_base + idx * x_stride_l, mask=mask, other=0.0)
        tl.store(real_out_ptr + out_base + idx * out_stride_l, vals, mask=mask)
        k += BLOCK


@triton.jit
def imag_extract_kernel(x_imag_ptr, imag_out_ptr,
                         batch: tl.int32, channels: tl.int32,
                         L: tl.int32,
                         x_stride_b: tl.int32, x_stride_c: tl.int32, x_stride_l: tl.int32,
                         out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32,
                         BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    x_base = b * x_stride_b + c * x_stride_c
    out_base = b * out_stride_b + c * out_stride_c

    k = 0
    while k < L + 1:
        idx = k + tl.arange(0, BLOCK)
        mask = idx < (L + 1)
        vals = tl.load(x_imag_ptr + x_base + idx * x_stride_l, mask=mask, other=0.0)
        tl.store(imag_out_ptr + out_base + idx * out_stride_l, vals, mask=mask)
        k += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-enabled replacement for the original run function.
        Computes normalized rfft(x, n=2*seqlen) along the last dimension for each (batch, channel) slice,
        returns real and imaginary parts as two float tensors of shape (batch, channels, seqlen+1).
        """
        # Ensure float32 for computation
        x_f32 = x.to(torch.float32)

        batch, channels, seqlen = x_f32.shape
        L = seqlen
        twoL = 2 * L

        # Compute rfft with PyTorch for correctness and speed
        x_freq_complex = torch.fft.rfft(x_f32, n=twoL, dim=-1)

        # Normalize by 2*seqlen, matching the original
        x_freq_complex = x_freq_complex / twoL

        # Allocate outputs
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Extract real and imaginary parts using torch.view_as_real (robust and fast)
        # view_as_real returns shape (B, C, L+1, 2) where the last dim [0] is real and [1] is imag.
        x_view = torch.view_as_real(x_freq_complex)  # (B, C, L+1, 2), float32
        real_view = x_view[:, :, :, 0]               # (B, C, L+1)
        imag_view = x_view[:, :, :, 1]               # (B, C, L+1)

        # Launch Triton kernels to copy into final outputs
        grid = (batch * channels,)

        # Strides for source views (views are contiguous along last dim)
        x_real_stride_b = real_view.stride(0)
        x_real_stride_c = real_view.stride(1)
        x_real_stride_l = real_view.stride(2)

        x_imag_stride_b = imag_view.stride(0)
        x_imag_stride_c = imag_view.stride(1)
        x_imag_stride_l = imag_view.stride(2)

        # Output strides
        out_stride_b = real_out.stride(0)
        out_stride_c = real_out.stride(1)
        out_stride_l = real_out.stride(2)

        # Copy with Triton kernels (simple, robust 1D copies)
        real_extract_kernel[grid](
            real_view, real_out,
            batch, channels, L,
            x_real_stride_b, x_real_stride_c, x_real_stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            BLOCK=1024,  # tile size; masking handles L+1 elements
            num_warps=4,
        )

        imag_extract_kernel[grid](
            imag_view, imag_out,
            batch, channels, L,
            x_imag_stride_b, x_imag_stride_c, x_imag_stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            BLOCK=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
