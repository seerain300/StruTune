import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                 # *f32, input tensor (B, C, L), contiguous last dim
    x_padded_ptr,          # *f32, output tensor (B, C, 2*L), contiguous last dim
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,           # original seqlen
    stride_b_in: tl.int32, # input stride for batch
    stride_c_in: tl.int32, # input stride for channel
    stride_l_in: tl.int32, # input stride for last dim
    out_stride_b: tl.int32, # output stride for batch
    out_stride_c: tl.int32, # output stride for channel
    out_stride_l: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr    # tile size for vectorization
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b_in + c * stride_c_in
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Copy original L elements into first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l_in, mask=mask, other=0.0)
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

    # We need to compute y[k] for k = 0..L. We'll store real and imag parts in outputs.
    # Initialize output accumulators for each k.
    # We'll implement accumulation per k using while loops.
    k = 0
    while k <= L:
        # Accumulate real and imag for this k
        real_acc = 0.0
        imag_acc = 0.0

        # Iterate over j = 0..2*L-1 in tiles
        j_start = 0
        while j_start < twoL:
            j = j_start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            vals_j = tl.load(x_padded_ptr + base_in + j * x_padded_ptr.stride(2), mask=mask, other=0.0)

            # angle = 2*pi*k*j / (2*L) = pi*k*j / L
            angle = (float(k) * float(j)) * (3.141592653589793 / float(L))
            cosv = tl.cos(angle)
            sinv = tl.sin(angle)

            # Multiply-accumulate: real += x_padded[j] * cos; imag += x_padded[j] * sin
            # Note: vals_j, cosv, sinv are vectors; Triton will broadcast/scalar-combine in expression.
            real_acc += tl.sum(vals_j * cosv)
            imag_acc += tl.sum(vals_j * sinv)

            j_start += BLOCK_N

        # Normalize by 2*L
        inv_twoL = 1.0 / float(twoL)
        real_acc = real_acc * inv_twoL
        imag_acc = imag_acc * inv_twoL

        # Store results to output positions [k]
        tl.store(real_out_ptr + base_out + k * out_stride_l, real_acc)
        tl.store(imag_out_ptr + base_out + k * out_stride_l, imag_acc)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure input is on CUDA
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        # Cast to float32 for numerical stability and Triton ops
        x_f32 = x.to(torch.float32)

        batch, channels, seqlen = x_f32.shape
        L = seqlen
        twoL = 2 * L
        L_plus_1 = L + 1

        # Allocate padded input tensor (B, C, 2*L), contiguous along last dim
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x_f32.device)

        # Launch Triton padding kernel: copy first L and zero-pad the rest
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x_f32, x_padded,
            batch, channels, L,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=256,
            num_warps=2,
        )

        # Allocate final outputs (real and imag parts) contiguous
        real_out = torch.empty((batch, channels, L_plus_1), dtype=torch.float32, device=x_f32.device)
        imag_out = torch.empty((batch, channels, L_plus_1), dtype=torch.float32, device=x_f32.device)

        # Launch Triton direct rfft kernel to compute real and imag parts
        # One program per (batch, channel)
        rfft_real_direct_kernel[grid_pad](
            x_padded, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=256,
            num_warps=2,
        )

        # Return real and imaginary parts, matching original function signature
        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
