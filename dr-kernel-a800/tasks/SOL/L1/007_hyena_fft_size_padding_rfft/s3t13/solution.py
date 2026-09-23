import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                  # *f32, input tensor (B, C, L)
    x_padded_ptr,           # *f32, output tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    stride_b_in: tl.int32,  # input stride for batch
    stride_c_in: tl.int32,  # input stride for channel
    stride_l_in: tl.int32,  # input stride for last dim
    stride_b_out: tl.int32, # output stride for batch
    stride_c_out: tl.int32, # output stride for channel
    stride_l_out: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr    # tile size for j
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b_in + c * stride_c_in
    base_out = b * stride_b_out + c * stride_c_out
    twoL = 2 * L

    # Copy original L elements into first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l_in, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * stride_l_out, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining positions with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * stride_l_out, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def read_real_kernel(
    x_complex_ptr,          # *complex64, complex output of torch.fft.rfft(...) with shape (B, C, L+1)
    real_out_ptr,           # *f32, output real part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    Lp1: tl.int32,          # L + 1
    stride_b_in: tl.int32,  # input complex stride for batch
    stride_c_in: tl.int32,  # input complex stride for channel
    stride_l_in: tl.int32,  # input complex stride for last dim
    out_stride_b: tl.int32, # output stride for batch
    out_stride_c: tl.int32, # output stride for channel
    out_stride_l: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr    # tile size for index
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b_in + c * stride_c_in
    base_out = b * out_stride_b + c * out_stride_c

    k = 0
    while k < Lp1:
        # Read real part from complex tensor at [b, c, k]
        # complex64 is two float32s interleaved in memory. torch.view_as_real returns
        # a view with last dim=2 (real, imag). We assume x_complex_ptr is a contiguous tensor
        # representing the complex output, and stride_l_in points to the element stride for k.
        v = tl.load(x_complex_ptr + base_in + k * stride_l_in, mask=True, other=0.0)
        tl.store(real_out_ptr + base_out + k * out_stride_l, v.real)
        k += 1


@triton.jit
def read_imag_kernel(
    x_complex_ptr,          # *complex64, complex output of torch.fft.rfft(...) with shape (B, C, L+1)
    imag_out_ptr,           # *f32, output imag part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    Lp1: tl.int32,          # L + 1
    stride_b_in: tl.int32,  # input complex stride for batch
    stride_c_in: tl.int32,  # input complex stride for channel
    stride_l_in: tl.int32,  # input complex stride for last dim
    out_stride_b: tl.int32, # output stride for batch
    out_stride_c: tl.int32, # output stride for channel
    out_stride_l: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr    # tile size for index
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b_in + c * stride_c_in
    base_out = b * out_stride_b + c * out_stride_c

    k = 0
    while k < Lp1:
        v = tl.load(x_complex_ptr + base_in + k * stride_l_in, mask=True, other=0.0)
        tl.store(imag_out_ptr + base_out + k * out_stride_l, v.imag)
        k += 1


@triton.jit
def scale_divide_real_kernel(
    real_ptr,               # *f32, input real tensor (B, C, L+1)
    scale: tl.float32,      # normalization factor (1/(2*L))
    out_ptr,                # *f32, output tensor (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    Lp1: tl.int32,          # L + 1
    stride_b: tl.int32,     # input stride for batch
    stride_c: tl.int32,     # input stride for channel
    stride_l: tl.int32,     # input stride for last dim
    out_stride_b: tl.int32, # output stride for batch
    out_stride_c: tl.int32, # output stride for channel
    out_stride_l: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr    # tile size for index
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c

    k = 0
    while k < Lp1:
        v = tl.load(real_ptr + base_in + k * stride_l, mask=True, other=0.0)
        v = v * scale
        tl.store(out_ptr + base_out + k * out_stride_l, v)
        k += 1


@triton.jit
def scale_divide_imag_kernel(
    imag_ptr,               # *f32, input imag tensor (B, C, L+1)
    scale: tl.float32,      # normalization factor (1/(2*L))
    out_ptr,                # *f32, output tensor (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    Lp1: tl.int32,          # L + 1
    stride_b: tl.int32,     # input stride for batch
    stride_c: tl.int32,     # input stride for channel
    stride_l: tl.int32,     # input stride for last dim
    out_stride_b: tl.int32, # output stride for batch
    out_stride_c: tl.int32, # output stride for channel
    out_stride_l: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr    # tile size for index
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c

    k = 0
    while k < Lp1:
        v = tl.load(imag_ptr + base_in + k * stride_l, mask=True, other=0.0)
        v = v * scale
        tl.store(out_ptr + base_out + k * out_stride_l, v)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen)
        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L
        Lp1 = L + 1

        # Cast input to float32 for numeric stability
        x_f32 = x.to(torch.float32)

        # Allocate padded input tensor (B, C, 2*L), contiguous
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch pad kernel
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x_f32, x_padded,
            batch, channels, L,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1,
            num_warps=1,
        )

        # Perform rfft on padded tensor using PyTorch (GPU) to leverage fast FFT
        # Output is complex of shape (B, C, L+1)
        x_freq = torch.fft.rfft(x_padded, n=twoL)  # complex

        # Allocate outputs for real and imag parts (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, Lp1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, Lp1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels to read real and imag parts from complex output
        grid_read = (batch * channels,)
        read_real_kernel[grid_read](
            x_freq, real_out,
            batch, channels, Lp1,
            x_freq.stride(0), x_freq.stride(1), x_freq.stride(2),
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            BLOCK_N=1,
            num_warps=1,
        )

        read_imag_kernel[grid_read](
            x_freq, imag_out,
            batch, channels, Lp1,
            x_freq.stride(0), x_freq.stride(1), x_freq.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1,
            num_warps=1,
        )

        # Normalize by 2*L: y = y / (2*L)
        scale = 1.0 / float(twoL)

        # Launch scale for real and imag separately
        grid_scale = (batch * channels,)
        scale_divide_real_kernel[grid_scale](
            real_out, scale, real_out,
            batch, channels, Lp1,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            BLOCK_N=1,
            num_warps=1,
        )

        scale_divide_imag_kernel[grid_scale](
            imag_out, scale, imag_out,
            batch, channels, Lp1,
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1,
            num_warps=1,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
