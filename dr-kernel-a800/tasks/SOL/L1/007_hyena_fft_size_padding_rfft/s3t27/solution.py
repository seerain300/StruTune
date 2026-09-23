import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                # *f32, input (B, C, L)
    x_padded_ptr,         # *f32, output (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,          # original seqlen
    x_stride_b: tl.int32,
    x_stride_c: tl.int32,
    x_stride_l: tl.int32,
    x2_stride_b: tl.int32,
    x2_stride_c: tl.int32,
    x2_stride_l: tl.int32,
    BLOCK_N: tl.constexpr
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * x_stride_b + c * x_stride_c
    base_out = b * x2_stride_b + c * x2_stride_c
    twoL = 2 * L

    # Copy original L elements into first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * x_stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * x2_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining positions with zeros (second half)
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * x2_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
    x_padded_ptr,         # *f32, input padded (B, C, 2*L)
    real_out_ptr,         # *f32, output real part (B, C, L+1)
    imag_out_ptr,         # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,          # original seqlen
    x2_stride_b: tl.int32,
    x2_stride_c: tl.int32,
    x2_stride_l: tl.int32,
    out_stride_b: tl.int32,
    out_stride_c: tl.int32,
    out_stride_l: tl.int32,
    BLOCK_N: tl.constexpr
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * x2_stride_b + c * x2_stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Compute r_k and i_k for k = 0..L-1
    k = 0
    while k < L:
        r_acc = tl.zeros((), dtype=tl.float32)
        i_acc = tl.zeros((), dtype=tl.float32)

        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            xj = tl.load(x_padded_ptr + base_in + j * x2_stride_l, mask=mask, other=0.0)

            # angle = (2*pi*k*j)/(2*L) = pi*k*j/L
            angle = (3.141592653589793 * k * j) / L
            cosv = tl.cos(angle)
            sinv = tl.sin(angle)

            r_acc += tl.sum(xj * cosv, axis=0)
            i_acc += tl.sum(xj * sinv, axis=0)

            start += BLOCK_N

        # Normalize by 2*L
        inv_twoL = 1.0 / (2.0 * L)
        r_acc *= inv_twoL
        i_acc *= inv_twoL

        tl.store(real_out_ptr + base_out + k * out_stride_l, r_acc)
        tl.store(imag_out_ptr + base_out + k * out_stride_l, i_acc)

        k += 1

    # Nyquist term: k = L
    r_acc = tl.zeros((), dtype=tl.float32)
    i_acc = tl.zeros((), dtype=tl.float32)

    start = 0
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        xj = tl.load(x_padded_ptr + base_in + j * x2_stride_l, mask=mask, other=0.0)

        # cos(pi*j) = (-1)^j; sin(pi*j) = 0
        # Implement (-1)^j using parity: (-1)^j = 1 if j is even, -1 if j is odd
        parity = (j & 1)  # 0 for even, 1 for odd
        # Create vector of -1 and 1 based on parity
        ones = tl.full([BLOCK_N], 1.0, dtype=tl.float32)
        minuses = tl.full([BLOCK_N], -1.0, dtype=tl.float32)
        cosv = tl.where(parity == 0, ones, minuses)  # (-1)^j
        sinv = tl.zeros([BLOCK_N], dtype=tl.float32)  # sin(pi*j)=0

        r_acc += tl.sum(xj * cosv, axis=0)
        i_acc += tl.sum(xj * sinv, axis=0)

        start += BLOCK_N

    inv_twoL = 1.0 / (2.0 * L)
    r_acc *= inv_twoL
    i_acc *= inv_twoL

    tl.store(real_out_ptr + base_out + L * out_stride_l, r_acc)
    tl.store(imag_out_ptr + base_out + L * out_stride_l, i_acc)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect single input tensor x of shape (batch, channels, seqlen)
        x = args[0]
        # Ensure dtype and contiguity
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        batch, channels, L = x.shape
        twoL = 2 * L

        # Allocate padded input
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel: one program per (batch, channel)
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=256,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1)
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel: one program per (batch, channel)
        grid_rfft = (batch * channels,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=256,
            num_warps=4,
        )

        # Return real and imaginary parts (Triton computed)
        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
