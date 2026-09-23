import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                 # *f32, input tensor (B, C, L)
    x_padded_ptr,          # *f32, output padded tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,           # original seqlen
    in_stride_b: tl.int32, # input stride for batch
    in_stride_c: tl.int32, # input stride for channel
    in_stride_l: tl.int32, # input stride for last dim (should be 1)
    out_stride_b: tl.int32, # output stride for batch
    out_stride_c: tl.int32, # output stride for channel
    out_stride_l: tl.int32, # output stride for last dim (should be 1)
    BLOCK_N: tl.constexpr    # tile size
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Copy original L elements into first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * in_stride_l, mask=mask, other=0.0)
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
def rfft_real_direct_kernel(
    x_padded_ptr,          # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,          # *f32, output real part (B, C, L+1), contiguous
    imag_out_ptr,          # *f32, output imag part (B, C, L+1), contiguous
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

    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Compute DFT coefficients for k = 0..L
    # y_k = (1/(2*L)) * sum_{j=0}^{2*L-1} x_padded[b, c, j] * (cos(2*pi*k*j/(2*L)) + i*sin(2*pi*k*j/(2*L)))
    k = 0
    while k <= L:
        real_sum = 0.0
        imag_sum = 0.0

        # Iterate over j in tiles
        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)
            mask = j < twoL

            # Load x_padded[b, c, j]
            vals = tl.load(x_padded_ptr + base_out + j * out_stride_l, mask=mask, other=0.0)

            # Compute angles: 2*pi*k*j/(2*L) = (k*j)*pi/L
            angle = (3.141592653589793 * k * j) / L
            cos_a = tl.cos(angle)
            sin_a = tl.sin(angle)

            # Accumulate real and imag parts: sum over j dimension
            # vals has shape [BLOCK_N], cos_a/sin_a broadcast accordingly
            # Sum across the vector to a scalar
            real_sum += tl.sum(vals * cos_a, axis=0)
            imag_sum += tl.sum(vals * sin_a, axis=0)

            start += BLOCK_N

        # Normalize by 2*L
        norm = 1.0 / (2.0 * L)
        real_sum = real_sum * norm
        imag_sum = imag_sum * norm

        # Store to outputs at index k (output length is L+1, k in [0..L])
        out_ptr_real = real_out_ptr + base_out + k * out_stride_l
        out_ptr_imag = imag_out_ptr + base_out + k * out_stride_l
        tl.store(out_ptr_real, real_sum)
        tl.store(out_ptr_imag, imag_sum)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Triton-ONLY implementation of the original run() behavior:
        - Input x: (batch, channels, seqlen)
        - Build implicit zero-padding to 2*seqlen per (batch, channel) slice using a Triton kernel
        - Compute real-input DFT with direct method in a Triton kernel
        - Normalize by 2*seqlen and return real and imaginary parts as (batch, channels, seqlen+1)
        """
        # Expect single input tensor x
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single input tensor")
        x = args[0]
        if not isinstance(x, torch.Tensor):
            raise RuntimeError("Input must be a torch.Tensor")

        # Ensure float32 and contiguous input
        x = x.to(torch.float32).contiguous()
        batch, channels, seqlen = x.shape

        # Allocate padded input tensor: shape (batch, channels, 2*seqlen)
        twoL = 2 * seqlen
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel: one program per (batch, channel)
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, seqlen,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel
        grid_rfft = (batch * channels,)
        rfft_real_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, seqlen,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
