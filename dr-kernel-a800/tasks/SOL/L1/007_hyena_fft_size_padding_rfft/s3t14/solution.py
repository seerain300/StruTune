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
def rfft_real_direct_kernel(
    x_padded_ptr,           # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,           # *f32, output real part (B, C, L+1), contiguous
    imag_out_ptr,           # *f32, output imag part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    out_stride_b: tl.int32, # stride for batch in output
    out_stride_c: tl.int32, # stride for channel in output
    out_stride_l: tl.int32, # stride for last dim in output (should be 1)
    BLOCK_N: tl.constexpr    # tile size for j
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * x_padded_ptr.stride(0) + c * x_padded_ptr.stride(1)
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Compute y[k] for k = 0..L and store into real_out and imag_out at index k
    k = 0
    while k <= L:
        real_acc = 0.0
        imag_acc = 0.0

        # Iterate over j = 0..2*L-1 in tiles
        j_start = 0
        while j_start < twoL:
            j = j_start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            vals_j = tl.load(x_padded_ptr + base_in + j * x_padded_ptr.stride(2), mask=mask, other=0.0)
            angle = (2.0 * 3.141592653589793 * k * j / twoL)
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)
            real_acc += tl.sum(vals_j * cos_term, axis=0)
            imag_acc += tl.sum(vals_j * sin_term, axis=0)
            j_start += BLOCK_N

        # Store into output tensors at index k
        # real_out_ptr[b, c, k], imag_out_ptr[b, c, k]
        out_base = base_out + k * out_stride_l
        tl.store(real_out_ptr + out_base, real_acc)
        tl.store(imag_out_ptr + out_base, imag_acc)
        k += 1


@triton.jit
def scale_real_kernel(
    real_ptr,               # *f32, real part to scale (B, C, L+1)
    out_real_ptr,           # *f32, output real part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    out_stride_b: tl.int32, # stride for batch in output
    out_stride_c: tl.int32, # stride for channel in output
    out_stride_l: tl.int32, # stride for last dim in output (should be 1)
    scale: tl.float32       # 1.0 / (2*L)
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels
    base_in = b * real_ptr.stride(0) + c * real_ptr.stride(1)
    base_out = b * out_stride_b + c * out_stride_c
    # Loop over k = 0..L
    k = 0
    while k <= L:
        val = tl.load(real_ptr + base_in + k * real_ptr.stride(2))
        val = val * scale
        tl.store(out_real_ptr + base_out + k * out_stride_l, val)
        k += 1


@triton.jit
def scale_imag_kernel(
    imag_ptr,               # *f32, imag part to scale (B, C, L+1)
    out_imag_ptr,           # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    out_stride_b: tl.int32, # stride for batch in output
    out_stride_c: tl.int32, # stride for channel in output
    out_stride_l: tl.int32, # stride for last dim in output (should be 1)
    scale: tl.float32       # 1.0 / (2*L)
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels
    base_in = b * imag_ptr.stride(0) + c * imag_ptr.stride(1)
    base_out = b * out_stride_b + c * out_stride_c
    k = 0
    while k <= L:
        val = tl.load(imag_ptr + base_in + k * imag_ptr.stride(2))
        val = val * scale
        tl.store(out_imag_ptr + base_out + k * out_stride_l, val)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Input: (batch, channels, seqlen)
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        x = x.to(torch.float32).contiguous()  # ensure float32 and contiguous
        batch, channels, L = x.shape
        twoL = 2 * L

        # 1) Construct padded input x_padded: (batch, channels, 2*L) in Triton
        # Allocate padded tensor
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        grid = (batch * channels,)
        pad_to_2L_kernel[grid](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # 2) Compute real and imaginary parts via direct DFT in Triton
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        rfft_real_direct_kernel[grid](
            x_padded, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # 3) Normalize by 2*L using Triton
        scale = 1.0 / (2.0 * L)

        # Triton scale for real
        scale_real_kernel[grid](
            real_out, real_out,  # write back into real_out
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            scale,
            num_warps=1,
        )

        # Triton scale for imag
        scale_imag_kernel[grid](
            imag_out, imag_out,  # write back into imag_out
            batch, channels, L,
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            scale,
            num_warps=1,
        )

        # Return real and imaginary parts (same shape as original output: (batch, channels, seqlen+1))
        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
