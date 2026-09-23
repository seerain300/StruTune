import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                 # *f32, input tensor (B, C, L)
    x_padded_ptr,          # *f32, output tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,           # original seqlen
    stride_b: tl.int32,    # input stride for batch
    stride_c: tl.int32,    # input stride for channel
    stride_l: tl.int32,    # input stride for last dim
    out_stride_b: tl.int32, # output stride for batch
    out_stride_c: tl.int32, # output stride for channel
    out_stride_l: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr,  # tile size for j
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

    # Fill remaining positions with zeros (second half, indices j >= L)
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
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

    # Accumulators for real and imaginary parts
    real_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    imag_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over k = 0..L (Nyquist included)
    k = 0
    while k <= L:
        # Accumulate over j = 0..2*L-1
        start_j = 0
        while start_j < twoL:
            j = start_j + tl.arange(0, BLOCK_N)
            mask = j < twoL
            xj = tl.load(x_padded_ptr + base_out + j * 1, mask=mask, other=0.0)  # x_padded[b, c, j]

            # angle = 2*pi*k*j/(2*L) = pi*k*j/L
            angle = (3.141592653589793 * k * j) / L  # use float64 math, Triton will cast internally

            # cos/sin
            cosv = tl.cos(angle)
            sinv = tl.sin(angle)

            contrib_real = xj * cosv
            contrib_imag = xj * sinv

            real_acc += contrib_real
            imag_acc += contrib_imag

            start_j += BLOCK_N

        # Normalize by 2*L
        inv_twoL = 1.0 / (2.0 * L)
        real_acc = real_acc * inv_twoL
        imag_acc = imag_acc * inv_twoL

        # Store results for this k into outputs at position k
        # real_out[b, c, k] and imag_out[b, c, k]
        out_base_k = base_out  # since output stride_l=1 for contiguous last dim
        # Write a single element; we only need to store one element per k
        # Placeholders: we store vector of length BLOCK_N as scalar
        # Triton will broadcast to scalar pointer; handle one element per iteration
        # For real and imag, store the first element of BLOCK_N vector (others ignored due to mask).
        tl.store(real_out_ptr + base_out + k * out_stride_l, real_acc[0])
        tl.store(imag_out_ptr + base_out + k * out_stride_l, imag_acc[0])

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure dtype float32 for numerical stability, as in original
        x = x.to(torch.float32)

        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L

        # Allocate padded input tensor
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)
        # Strides for input
        stride_b = x.stride(0)
        stride_c = x.stride(1)
        stride_l = x.stride(2)
        # Launch padding kernel
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, L,
            stride_b, stride_c, stride_l,
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Strides for outputs
        out_stride_b = real_out.stride(0)
        out_stride_c = real_out.stride(1)
        out_stride_l = real_out.stride(2)  # should be 1

        # Launch direct rfft kernel
        grid_rfft = (batch * channels,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            out_stride_b, out_stride_c, out_stride_l,
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
