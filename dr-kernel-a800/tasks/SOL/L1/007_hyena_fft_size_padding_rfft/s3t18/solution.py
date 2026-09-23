import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(x_ptr, x_padded_ptr,
                     batch: tl.int32, channels: tl.int32, L: tl.int32,
                     stride_b: tl.int32, stride_c: tl.int32, stride_l: tl.int32,
                     out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32,
                     BLOCK_N: tl.constexpr):
    """
    Zero-pad input x of shape (batch, channels, L) to length twoL = 2*L along last dim,
    producing x_padded of shape (batch, channels, 2*L).
    One program per (batch, channel) slice.
    """
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Copy original L elements to the first half
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
def compute_rfft_real_imag_kernel(x_padded_ptr, real_out_ptr, imag_out_ptr,
                                  batch: tl.int32, channels: tl.int32, L: tl.int32,
                                  in_stride_b: tl.int32, in_stride_c: tl.int32, in_stride_l: tl.int32,
                                  out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32,
                                  BLOCK_N: tl.constexpr):
    """
    Compute real-input DFT over padded time length twoL = 2*L for each (batch, channel) slice,
    and store real and imaginary parts into real_out and imag_out of shape (batch, channels, L+1).
    Normalization: divide by 2*L.
    """
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Compute y_k for k in 0..L
    k = 0
    while k <= L:
        acc_real = tl.zeros((), dtype=tl.float32)
        acc_imag = tl.zeros((), dtype=tl.float32)

        j_start = 0
        while j_start < twoL:
            j = j_start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            xj = tl.load(x_padded_ptr + base_in + j * in_stride_l, mask=mask, other=0.0).to(tl.float32)

            angle = 2.0 * 3.141592653589793 * k * j / twoL  # radians
            cosj = tl.cos(angle)
            sinj = tl.sin(angle)

            # Sum contributions across tile
            for lane in range(BLOCK_N):
                xi = xj[lane]
                cosl = cosj[lane]
                sinl = sinj[lane]
                acc_real += xi * cosl
                acc_imag += xi * sinl

            j_start += BLOCK_N

        # Normalize by 2*L
        scale = 1.0 / twoL
        acc_real *= scale
        acc_imag *= scale

        # Store to output at index k (output length is L+1)
        out_index = k
        tl.store(real_out_ptr + base_out + out_index * out_stride_l, acc_real)
        tl.store(imag_out_ptr + base_out + out_index * out_stride_l, acc_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor x with shape (batch, channels, seqlen)
        x = args[0]
        # Ensure float32 for numerical stability
        x = x.to(torch.float32).contiguous()
        batch, channels, L = x.shape
        twoL = 2 * L

        # Allocate padded input tensor (B, C, 2*L), contiguous
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel: one program per (batch, channel) slice
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=256,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch compute kernel: one program per (batch, channel) slice
        grid_dft = (batch * channels,)
        compute_rfft_real_imag_kernel[grid_dft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=256,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
