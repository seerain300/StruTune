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

    # Copy first L elements (original x) into padded tensor
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * in_stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Write zeros for the remaining elements (implicit zero-padding)
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
    L: tl.int32,         # original seqlen
    out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32,
    BLOCK_N: tl.constexpr,
):
    # One program per (batch, channel)
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Precompute inv_twoL for normalization
    inv_twoL = 1.0 / twoL

    # For k in 0..L, compute y_k
    k = 0
    while k <= L:
        # Accumulators for real and imag parts
        acc_real = tl.zeros([1], dtype=tl.float32)
        acc_imag = tl.zeros([1], dtype=tl.float32)

        # Sum over j = 0..2*L-1
        j = 0
        while j < twoL:
            # Load x_padded[b, c, j]
            v = tl.load(x_padded_ptr + base_out + j * out_stride_l, mask=True, other=0.0)  # scalar
            # Compute angle = 2*pi*k*j/(2*L)
            angle = (2.0 * 3.141592653589793 * k * j) / twoL
            cosv = tl.cos(angle)
            sinv = tl.sin(angle)
            acc_real += v * cosv
            acc_imag += v * sinv
            j += 1

        # Store normalized results
        tl.store(real_out_ptr + base_out + k * out_stride_l, acc_real * inv_twoL)
        tl.store(imag_out_ptr + base_out + k * out_stride_l, acc_imag * inv_twoL)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Fused Triton computation of real FFT with implicit zero-padding and output extraction.
        Input x: (batch, channels, seqlen)
        Returns: (batch, channels, seqlen+1) real and imag parts as two float32 tensors.
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Ensure dtype is float32
        x = x.to(torch.float32).contiguous()
        batch, channels, seqlen = x.shape

        # Allocate padded input tensor (B, C, 2*seqlen)
        twoL = 2 * seqlen
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel: one program per (batch, channel)
        grid_pad = (batch * channels,)
        _pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, seqlen,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=256,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel: one program per (batch, channel)
        grid_rfft = (batch * channels,)
        _rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, seqlen,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1,  # inner loop over j is handled by while; BLOCK_N here is not used directly
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
