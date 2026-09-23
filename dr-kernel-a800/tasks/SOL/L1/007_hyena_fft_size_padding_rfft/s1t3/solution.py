import torch
import triton
import triton.language as tl


@triton.jit
def _real_to_re_kernel(x_ptr, out_real_ptr, B, C, N, L,
                        stride_b, stride_c, stride_t,
                        out_stride_b, out_stride_c, out_stride_m):
    # Each program handles one (b, c) slice
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    if (pid_b >= B) or (pid_c >= C):
        return

    # Base offsets for this (b, c)
    x_base = pid_b * stride_b + pid_c * stride_c
    out_base = pid_b * out_stride_b + pid_c * out_stride_c

    inv_N = 1.0 / N

    # Loop over output frequency j from 0 to L (inclusive)
    j = 0
    while j <= L:
        acc = 0.0
        # Accumulate over all t in [0, N)
        t = 0
        while t < N:
            # Scalar load of x[b, c, t] (guaranteed in-bounds since t < N)
            x_val = tl.load(x_ptr + x_base + t * stride_t)
            # Compute cos(2*pi*j*t/N), note j ranges 0..L (inclusive)
            angle = 2.0 * 3.141592653589793 * j * t / N
            cos_term = tl.cos(angle)
            acc += x_val * cos_term
            t += 1
        # Normalize and store real part at index j
        re = acc * inv_N
        tl.store(out_real_ptr + out_base + j * out_stride_m, re)
        j += 1


@triton.jit
def _real_to_im_kernel(x_ptr, out_imag_ptr, B, C, N, L,
                        stride_b, stride_c, stride_t,
                        out_stride_b, out_stride_c, out_stride_m):
    # Each program handles one (b, c) slice
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    if (pid_b >= B) or (pid_c >= C):
        return

    x_base = pid_b * stride_b + pid_c * stride_c
    out_base = pid_b * out_stride_b + pid_c * out_stride_c

    inv_N = 1.0 / N

    j = 0
    while j <= L:
        acc = 0.0
        t = 0
        while t < N:
            x_val = tl.load(x_ptr + x_base + t * stride_t)
            angle = 2.0 * 3.141592653589793 * j * t / N
            sin_term = tl.sin(angle)
            acc += x_val * sin_term
            t += 1
        im = acc * inv_N
        tl.store(out_imag_ptr + out_base + j * out_stride_m, im)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen)
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        assert x.dtype == torch.float32, "Input must be float32"
        B, C, L = x.shape
        N = 2 * L  # padding length for rfft
        # Output length (seqlen + 1) equals L + 1
        # We'll allocate (B, C, L+1)

        # Allocate outputs
        out_real = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)

        # Get strides (in elements)
        stride_b, stride_c, stride_t = x.stride()
        out_stride_b, out_stride_c, out_stride_m = out_real.stride()

        # Launch one program per (b, c) slice
        grid = (B, C)

        # Kernel 1: compute real part
        _real_to_re_kernel[grid](
            x, out_real,
            B, C, N, L,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
        )

        # Kernel 2: compute imaginary part
        _real_to_im_kernel[grid](
            x, out_imag,
            B, C, N, L,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
        )

        # The original code divides by N (=2*seqlen); we already multiply by 1/N in the kernels.
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
