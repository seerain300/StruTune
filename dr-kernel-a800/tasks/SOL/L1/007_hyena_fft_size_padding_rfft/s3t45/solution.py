import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                     # *f32, input tensor (B, C, L) contiguous
    x_padded_ptr,              # *f32, output tensor (B, C, 2*L) contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,               # original seqlen
    stride_b: tl.int32,        # input stride for batch
    stride_c: tl.int32,        # input stride for channel
    stride_l: tl.int32,        # input stride for last dim (seqlen)
    out_stride_b: tl.int32,    # output stride for batch
    out_stride_c: tl.int32,    # output stride for channel
    out_stride_l: tl.int32,    # output stride for last dim (2*seqlen)
    BLOCK_N: tl.constexpr       # tile size
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Copy first L elements
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining L elements with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_real_direct_kernel(
    x_padded_ptr,              # *f32, padded input (B, C, 2*L), contiguous
    real_out_ptr,              # *f32, output real part (B, C, L+1), contiguous
    imag_out_ptr,              # *f32, output imag part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,               # original seqlen
    out_stride_b: tl.int32,    # real_out stride for batch
    out_stride_c: tl.int32,    # real_out stride for channel
    out_stride_l: tl.int32,    # real_out stride for last dim
    x_padded_stride_b: tl.int32,
    x_padded_stride_c: tl.int32,
    x_padded_stride_l: tl.int32,
    BLOCK_N: tl.constexpr       # tile size for j
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base = b * x_padded_stride_b + c * x_padded_stride_c
    base_out = b * out_stride_b + c * out_stride_c

    twoL = 2 * L
    scale = 1.0 / twoL

    k = 0
    while k <= L:
        # Accumulators
        acc_real = 0.0
        acc_imag = 0.0

        # Sum over j = 0..2*L-1 in tiles
        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            xj = tl.load(x_padded_ptr + base + j * x_padded_stride_l, mask=mask, other=0.0)

            # angle = 2*pi*k / (2*L) = pi*k / L
            # Triton expects scalars for tl.cos/tl.sin; compute per tile as scalar
            # Note: k is loop scalar; Triton will broadcast it.
            angle = (2.0 * 3.141592653589793 * k) / twoL

            # cos/sin for this k across j
            cos_t = tl.cos(angle)
            sin_t = tl.sin(angle)

            # Accumulate: real += xj * cos_t; imag += xj * sin_t
            acc_real += tl.sum(xj * cos_t, axis=0)
            acc_imag += tl.sum(xj * sin_t, axis=0)

            start += BLOCK_N

        # Normalize
        acc_real = acc_real * scale
        acc_imag = acc_imag * scale

        # Store to output at position k (since k = 0..L -> L+1)
        # Ensure output is contiguous: (B, C, L+1)
        tl.store(real_out_ptr + base_out + k * out_stride_l, acc_real)
        tl.store(imag_out_ptr + base_out + k * out_stride_l, acc_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen)
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        x = x.to(torch.float32)
        B, C, L = x.shape
        twoL = 2 * L

        # Allocate padded input (B, C, 2*L) contiguous
        x_padded = torch.empty((B, C, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel: one program per (b, c)
        grid_pad = (B * C,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            B, C, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs (real and imag) (B, C, L+1) contiguous
        real_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel: one program per (b, c)
        grid_rfft = (B * C,)
        rfft_real_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            B, C, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
