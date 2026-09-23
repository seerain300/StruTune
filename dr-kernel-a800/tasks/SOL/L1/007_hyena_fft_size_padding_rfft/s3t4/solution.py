import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                  # *f32, input (B, C, L)
    x_padded_ptr,           # *f32, output padded tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # seqlen
    stride_b: tl.int32,     # input stride for batch
    stride_c: tl.int32,     # input stride for channel
    stride_l: tl.int32,     # input stride for last dim
    out_stride_b: tl.int32, # output stride for batch
    out_stride_c: tl.int32, # output stride for channel
    out_stride_l: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr    # tile size
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
def rfft_real_kernel(
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

    twoL = 2 * L
    base_out = b * out_stride_b + c * out_stride_c
    inv_twoL = 1.0 / twoL
    L_plus1 = L + 1

    # For each frequency index k in [0..L], compute sum over j=0..2*L-1
    k = 0
    while k <= L:
        sum_real = 0.0
        sum_imag = 0.0

        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)  # vector of j indices
            mask = j < twoL
            # Load padded input values
            vals = tl.load(x_padded_ptr + base_out + j * out_stride_l, mask=mask, other=0.0)

            # Compute cos and sin for all j in the tile
            angle = (tl.pi * k * j) / twoL
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)

            # Accumulate contributions
            sum_real += tl.sum(vals * cos_term, axis=0)
            sum_imag += tl.sum(vals * sin_term, axis=0)

            start += BLOCK_N

        # Normalize by 2*L
        sum_real = sum_real * inv_twoL
        sum_imag = sum_imag * inv_twoL

        # Store real and imag parts at index k (contiguous (B, C, L+1))
        out_offset = b * (channels * (L_plus1)) + c * (L_plus1) + k
        tl.store(real_out_ptr + out_offset, sum_real)
        tl.store(imag_out_ptr + out_offset, sum_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Cast to float32 to match original behavior
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L

        # Allocate padded input: shape (batch, channels, 2*L)
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel: one program per (batch, channel)
        grid = (batch * channels,)
        pad_to_2L_kernel[grid](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=256,
            num_warps=4, num_stages=2
        )

        # Allocate outputs: real and imag parts (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch real FFT kernel
        rfft_real_kernel[grid](
            x_padded, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=256,
            num_warps=4, num_stages=2
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
