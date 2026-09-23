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

    # Copy original L elements into first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * in_stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining positions with zeros (implicit zero-padding)
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def _rfft_direct_kernel(
    x_padded_ptr,        # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,        # *f32, output real part (B, C, L+1)
    imag_out_ptr,        # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,         # original seqlen; output length is L+1
    out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32,
    twoL: tl.int32,      # 2 * L
    BLOCK_N: tl.constexpr,
):
    # One program per (batch, channel)
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels
    base_padded = b * out_stride_b + c * out_stride_c
    base_out = b * out_stride_b + c * out_stride_c  # same layout for outputs

    inv_twoL = 1.0 / (2.0 * L)

    # For each k in [0, L], compute sum over j in [0, 2*L-1]
    k = 0
    while k <= L:
        real_sum = tl.zeros([1], dtype=tl.float32)
        imag_sum = tl.zeros([1], dtype=tl.float32)

        # Tile over j dimension
        start = 0
        while start < twoL:
            j_vec = start + tl.arange(0, BLOCK_N)
            mask_j = j_vec < twoL

            # Load padded input values
            vals = tl.load(x_padded_ptr + base_padded + j_vec * out_stride_l, mask=mask_j, other=0.0)

            # Compute angles for cosine/sine
            angle = tl.pi * (k * j_vec) / twoL

            # Accumulate real and imaginary parts
            # Ensure vals is float32; angle is float32
            real_sum += tl.sum(vals * tl.cos(angle), axis=0)
            imag_sum += tl.sum(vals * tl.sin(angle), axis=0)

            start += BLOCK_N

        # Normalize by 2*L
        real_sum = real_sum * inv_twoL
        imag_sum = imag_sum * inv_twoL

        # Store results into (L+1)-length outputs
        # real_out[b, c, k] and imag_out[b, c, k]
        tl.store(real_out_ptr + base_out + k * out_stride_l, real_sum)
        tl.store(imag_out_ptr + base_out + k * out_stride_l, imag_sum)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only fused padding and real-FFT:
        - Construct implicit zero-padding to 2*seqlen for each (batch, channel) slice.
        - Compute real-input DFT using Triton and write real and imaginary parts.
        Returns:
          real_out: float32 tensor of shape (batch, channels, seqlen+1)
          imag_out: float32 tensor of shape (batch, channels, seqlen+1)
        """
        # Ensure dtype and device
        x = x.to(torch.float32)
        batch, channels, seqlen = x.shape
        twoL = 2 * seqlen
        L_out = seqlen + 1

        # Allocate padded input and outputs
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)
        real_out = torch.empty((batch, channels, L_out), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L_out), dtype=torch.float32, device=x.device)

        # Launch padding kernel: one program per (batch, channel)
        grid_pad = (batch * channels,)
        _pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, seqlen,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Launch direct rfft kernel: one program per (batch, channel)
        grid_dft = (batch * channels,)
        _rfft_direct_kernel[grid_dft](
            x_padded, real_out, imag_out,
            batch, channels, seqlen,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            twoL,
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
