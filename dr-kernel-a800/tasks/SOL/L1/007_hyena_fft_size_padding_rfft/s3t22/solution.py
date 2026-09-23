import torch
import triton
import triton.language as tl


@triton.jit
def _pad_to_2L_kernel(
    x_ptr,               # *f32, input tensor (B, C, L)
    x_padded_ptr,        # *f32, output tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,         # seqlen
    in_stride_b: tl.int32, in_stride_c: tl.int32, in_stride_l: tl.int32,
    out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32,
    BLOCK_N: tl.constexpr,
):
    # One program per (batch, channel)
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels
    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Copy first L elements (original x) into padded tensor
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * in_stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Write zeros for the remaining elements (implicit zero-padding)
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def _rfft_real_kernel_v2(
    x_padded_ptr,        # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,        # *f32, output real part (B, C, L+1)
    imag_out_ptr,        # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,         # seqlen
    in_stride_b: tl.int32, in_stride_c: tl.int32, in_stride_l: tl.int32,   # input strides for x_padded (read)
    out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32, # output strides
    BLOCK_N: tl.constexpr,
):
    # One program per (batch, channel)
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels
    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Loop over frequency k = 0..L
    k = 0
    while k <= L:
        real_acc = 0.0
        imag_acc = 0.0

        # Iterate over time positions j = 0..2*L-1 in tiles
        j_start = 0
        while j_start < twoL:
            j = j_start + tl.arange(0, BLOCK_N)
            mask = j < twoL

            # Load x_padded[b, c, j]
            vals = tl.load(x_padded_ptr + base_in + j * in_stride_l, mask=mask, other=0.0)

            # Compute angle for rfft: angle = pi * k * j / (2*L)
            angle = tl.pi * k * j * (1.0 / twoL)
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)

            # Accumulate contributions
            # For vector load, we can do elementwise accumulation into scalars:
            for n in range(BLOCK_N):
                if n < twoL:
                    real_acc += vals[n] * cos_term[n]
                    imag_acc += vals[n] * sin_term[n]

            j_start += BLOCK_N

        # Normalize by 2*L
        real_acc = real_acc / twoL
        imag_acc = imag_acc / twoL

        # Store into output at position k
        tl.store(real_out_ptr + base_out + k * out_stride_l, real_acc)
        tl.store(imag_out_ptr + base_out + k * out_stride_l, imag_acc)

        k += 1


def _cdiv(x, y):
    return (x + y - 1) // y


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure float32 for numerical stability
        x_f32 = x.to(torch.float32)
        batch, channels, L = x_f32.shape

        # Allocate padded input (B, C, 2*L) contiguous
        twoL = 2 * L
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel
        grid = (batch * channels,)
        _pad_to_2L_kernel[grid](
            x_f32, x_padded,
            batch, channels, L,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=256,
            num_warps=4,
        )

        # Allocate outputs (B, C, L+1) contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch Triton rfft real kernel
        _rfft_real_kernel_v2[grid](
            x_padded, real_out, imag_out,
            batch, channels, L,
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),   # input strides for x_padded (read)
            real_out.stride(0), real_out.stride(1), real_out.stride(2),    # output strides
            BLOCK_N=256,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
