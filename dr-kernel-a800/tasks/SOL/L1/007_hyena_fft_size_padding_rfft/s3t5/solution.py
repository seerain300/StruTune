import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,            # *f32, input tensor (B, C, L)
    x_padded_ptr,     # *f32, output padded tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,      # seqlen
    stride_b: tl.int32,  # stride for batch in input
    stride_c: tl.int32,  # stride for channel in input
    stride_l: tl.int32,  # stride for last dim in input
    out_stride_b: tl.int32,  # stride for batch in padded output
    out_stride_c: tl.int32,  # stride for channel in padded output
    out_stride_l: tl.int32,  # stride for last dim in padded output
    BLOCK_N: tl.constexpr
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c

    # Copy original L elements to the first L positions of the padded tensor
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Fill the remaining positions (L to 2*L-1) with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < (2 * L):
        j = start + tl.arange(0, BLOCK_N)
        mask = j < (2 * L)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
    x_padded_ptr,      # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,      # *f32, output real part (B, C, L+1)
    imag_out_ptr,      # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,       # seqlen
    out_stride_b: tl.int32,  # stride for batch in padded output
    out_stride_c: tl.int32,  # stride for channel in padded output
    out_stride_l: tl.int32,  # stride for last dim in padded output
    real_stride_b: tl.int32, # stride for batch in real_out
    real_stride_c: tl.int32, # stride for channel in real_out
    real_stride_l: tl.int32, # stride for last dim in real_out (usually 1)
    imag_stride_b: tl.int32, # stride for batch in imag_out
    imag_stride_c: tl.int32, # stride for channel in imag_out
    imag_stride_l: tl.int32, # stride for last dim in imag_out (usually 1)
    BLOCK_N: tl.constexpr
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    twoL = 2 * L
    base_out = b * out_stride_b + c * out_stride_c
    real_base = b * real_stride_b + c * real_stride_c
    imag_base = b * imag_stride_b + c * imag_stride_c

    # Compute DFT coefficients for k in [0..L]
    k = 0
    while k <= L:
        sum_real = 0.0
        sum_imag = 0.0

        # Iterate over j = 0..2*L-1 in tiles
        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            x_vals = tl.load(x_padded_ptr + base_out + j * out_stride_l, mask=mask, other=0.0)

            # Compute cos and sin for this k and all j in the tile
            angle = (2.0 * tl.pi * k * j) / twoL
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)

            # Accumulate contributions
            prod_real = x_vals * cos_term
            prod_imag = x_vals * sin_term

            sum_real += tl.sum(prod_real, axis=0)
            sum_imag += tl.sum(prod_imag, axis=0)

            start += BLOCK_N

        # Normalize by 2*L (original code divides by fft_size = 2*L)
        inv_twoL = 1.0 / twoL
        sum_real = sum_real * inv_twoL
        sum_imag = sum_imag * inv_twoL

        # Store results at frequency index k
        out_offset = real_base + k * real_stride_l
        tl.store(real_out_ptr + out_offset, sum_real)
        out_offset = imag_base + k * imag_stride_l
        tl.store(imag_out_ptr + out_offset, sum_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Cast to float32 for numerical stability (explicit as in original)
        x = x.to(torch.float32)

        # Ensure input is on CUDA for Triton
        assert x.is_cuda, "Input must be a CUDA tensor for Triton kernels."
        x = x.contiguous()

        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L

        # Allocate padded input tensor (B, C, 2*L)
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)
        x_padded = x_padded.contiguous()

        # Launch pad kernel: build implicit zero-padding to 2*L
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel
        grid_rfft = (batch * channels,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
