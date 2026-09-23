import torch
import triton
import triton.language as tl


@triton.jit
def _rfft_real_kernel(
    x_ptr, out_real_ptr,
    B, C, N, M,
    stride_b, stride_c, stride_t,
    out_stride_b, out_stride_c, out_stride_m,
    BLOCK_T: tl.constexpr,
):
    # Each program handles one (b, c) slice and computes all j in [0, M)
    b = tl.program_id(0)
    c = tl.program_id(1)

    base_x = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c

    # Compute re_j = (1/N) * sum_{t=0}^{N-1} x[t] * cos(2*pi*j*t/N)
    # We will accumulate in fp32 and store fp32 output.
    for j in range(0, M):
        acc = 0.0
        # Vectorize over t in chunks of BLOCK_T
        for t_start in range(0, N, BLOCK_T):
            t_offsets = t_start + tl.arange(0, BLOCK_T)
            mask = t_offsets < N
            x_vals = tl.load(x_ptr + base_x + t_offsets * stride_t, mask=mask, other=0.0)
            # angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * j * t_offsets * (1.0 / N)
            cosv = tl.cos(angle)
            # Multiply and reduce
            acc += tl.sum(x_vals * cosv, axis=0)
        re = acc * (1.0 / N)
        tl.store(out_real_ptr + base_out + j * out_stride_m, re)


@triton.jit
def _rfft_imag_kernel(
    x_ptr, out_imag_ptr,
    B, C, N, M,
    stride_b, stride_c, stride_t,
    out_stride_b, out_stride_c, out_stride_m,
    BLOCK_T: tl.constexpr,
):
    # Each program handles one (b, c) slice and computes all j in [0, M)
    b = tl.program_id(0)
    c = tl.program_id(1)

    base_x = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c

    # Compute im_j = (1/N) * sum_{t=0}^{N-1} x[t] * sin(2*pi*j*t/N)
    for j in range(0, M):
        acc = 0.0
        for t_start in range(0, N, BLOCK_T):
            t_offsets = t_start + tl.arange(0, BLOCK_T)
            mask = t_offsets < N
            x_vals = tl.load(x_ptr + base_x + t_offsets * stride_t, mask=mask, other=0.0)
            angle = 2.0 * 3.141592653589793 * j * t_offsets * (1.0 / N)
            sinv = tl.sin(angle)
            acc += tl.sum(x_vals * sinv, axis=0)
        im = acc * (1.0 / N)
        tl.store(out_imag_ptr + base_out + j * out_stride_m, im)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton implementation of:
            x_freq = torch.fft.rfft(x.to(torch.float32), n=2*L) / (2*L)
            return x_freq.real, x_freq.imag
        where L = x.shape[2], outputs shape (B, C, L+1).
        """
        B, C, L = x.shape
        N = 2 * L  # padded FFT length
        M = L + 1  # output length per (b, c) slice

        # Cast to float32
        x_f32 = x.to(torch.float32)

        # Allocate outputs
        out_real = torch.empty((B, C, M), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, M), device=x.device, dtype=torch.float32)

        # Strides in elements
        stride_b, stride_c, stride_t = x_f32.stride()
        out_stride_b, out_stride_c, out_stride_m = out_real.stride()

        # Launch one program per (b, c) slice
        grid = (B, C)

        # Choose a reasonable chunk size for t-summation
        BLOCK_T = 256

        # Compute real part via DFT
        _rfft_real_kernel[grid](
            x_f32, out_real,
            B, C, N, M,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            BLOCK_T=BLOCK_T,
        )

        # Compute imaginary part via DFT
        _rfft_imag_kernel[grid](
            x_f32, out_imag,
            B, C, N, M,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            BLOCK_T=BLOCK_T,
        )

        # The original code divides by n (= 2*L); we already normalized by 1/N in the kernels.
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
