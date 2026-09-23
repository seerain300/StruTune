import torch
import triton
import triton.language as tl

@triton.jit
def pad_to_2L_kernel(
    x_ptr,                  # *f32, input tensor (B, C, L)
    x_padded_ptr,           # *f32, output tensor (B, C, 2L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # seqlen
    stride_b: tl.int32,     # input stride for batch
    stride_c: tl.int32,     # input stride for channel
    stride_l: tl.int32,     # input stride for last dim
    out_stride_b: tl.int32, # output stride for batch
    out_stride_c: tl.int32, # output stride for channel
    out_stride_l: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr    # tile size for j
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Copy original L elements into first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining positions with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
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

    # Accumulators for real and imag parts
    real_sum = 0.0
    imag_sum = 0.0

    # Iterate over k = 0 .. L
    k = 0
    while k <= L:
        # Accumulate over j = 0 .. 2L-1 in tiles
        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            vals = tl.load(x_padded_ptr + base_in + j * x_padded_ptr.stride(2), mask=mask, other=0.0)

            # Compute angles: cos(pi*k*j/(2*L)), sin(pi*k*j/(2*L))
            angle = (tl.pi * k * j) / twoL
            cos_t = tl.cos(angle)
            sin_t = tl.sin(angle)

            # Multiply and reduce
            real_sum += tl.sum(vals * cos_t, axis=0)
            imag_sum += tl.sum(vals * sin_t, axis=0)

            start += BLOCK_N

        # Normalize by 2*L
        norm = 1.0 / (2.0 * L)
        real_sum = real_sum * norm
        imag_sum = imag_sum * norm

        # Store results at index k
        out_ptr_real = real_out_ptr + base_out + k * out_stride_l
        out_ptr_imag = imag_out_ptr + base_out + k * out_stride_l
        tl.store(out_ptr_real, real_sum)
        tl.store(out_ptr_imag, imag_sum)

        # Reset accumulators for next k
        real_sum = 0.0
        imag_sum = 0.0
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x shape: (batch, channels, seqlen)
        assert x.dim() == 3, "Input must be 3D (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L

        # Ensure float32 and contiguous for Triton
        x_f32 = x.to(torch.float32).contiguous()

        # Allocate padded input (B, C, 2*L), contiguous
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch pad kernel: construct implicit zero-padding to 2*L per (b, c)
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x_f32, x_padded,
            batch, channels, L,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel (computes for k=0..L)
        grid_rfft = (batch * channels,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
