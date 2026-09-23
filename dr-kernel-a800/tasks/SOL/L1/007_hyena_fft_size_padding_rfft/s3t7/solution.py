import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                 # *f32, input tensor (B, C, L)
    out_ptr,               # *f32, output padded tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,           # seqlen
    in_stride_b: tl.int32, # stride for batch in input
    in_stride_c: tl.int32, # stride for channel in input
    in_stride_l: tl.int32, # stride for last dim in input (usually 1)
    out_stride_b: tl.int32, # stride for batch in output
    out_stride_c: tl.int32, # stride for channel in output
    out_stride_l: tl.int32, # stride for last dim in output (usually 1)
    BLOCK_N: tl.constexpr,  # tile size for j
):
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    twoL = 2 * L
    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c

    # Copy original L elements to first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * in_stride_l, mask=mask, other=0.0)
        tl.store(out_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Fill the remaining positions with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(out_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
    x_padded_ptr,          # *f32, padded input tensor (B, C, 2*L)
    real_out_ptr,          # *f32, output real part (B, C, L+1)
    imag_out_ptr,          # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,           # seqlen
    out_stride_b: tl.int32,   # stride for batch in real_out
    out_stride_c: tl.int32,   # stride for channel in real_out
    out_stride_l: tl.int32,   # stride for last dim in real_out (usually 1)
    in_stride_b: tl.int32,    # stride for batch in padded input
    in_stride_c: tl.int32,    # stride for channel in padded input
    in_stride_l: tl.int32,    # stride for last dim in padded input (usually 1)
    BLOCK_N: tl.constexpr,    # tile size for j
):
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    twoL = 2 * L
    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c

    # For each output frequency index k in [0, L]
    k = 0
    while k <= L:
        sum_real = 0.0
        sum_imag = 0.0

        # Summation over j = 0..2*L-1 in tiles
        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)  # vector of time indices
            mask = j < twoL
            # Load x_padded[b, c, j]; for j >= L, values are zeros
            vals = tl.load(x_padded_ptr + base_in + j * in_stride_l, mask=mask, other=0.0)

            # Compute angle for DFT
            angle = (2.0 * tl.pi * k * j) / twoL
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)

            # Accumulate contributions
            sum_real += tl.sum(vals * cos_term, axis=0)
            sum_imag += tl.sum(vals * sin_term, axis=0)

            start += BLOCK_N

        # Normalize by 2*L
        sum_real = sum_real * (1.0 / twoL)
        sum_imag = sum_imag * (1.0 / twoL)

        # Store results at frequency index k (k is 0..L, output length is L+1)
        out_offset = b * (channels * (L + 1)) + c * (L + 1) + k
        tl.store(real_out_ptr + out_offset, sum_real)
        tl.store(imag_out_ptr + out_offset, sum_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure float32
        x = x.to(torch.float32)
        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L
        L_plus1 = L + 1

        # Allocate padded input: (batch, channels, 2*L)
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch pad kernel: one program per (batch, channel)
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=128,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L_plus1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L_plus1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel
        grid_rfft = (batch * channels,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=128,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
