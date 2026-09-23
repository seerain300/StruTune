import torch
import triton
import triton.language as tl


@triton.jit
def _copy_real_kernel(
    in_cmplx_ptr,        # *complex64, input complex tensor (B, C, L+1)
    out_real_ptr,        # *f32, output real tensor (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,         # original seqlen
    in_stride_b: tl.int32, in_stride_c: tl.int32, in_stride_l: tl.int32,
    out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels
    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c
    # L+1 since output length equals seqlen + 1 (rfft for 2*L returns L+1)
    length = L + 1
    start = 0
    while start < length:
        idx = start + tl.arange(0, BLOCK_N)
        mask = idx < length
        vals = tl.load(in_cmplx_ptr + base_in + idx * in_stride_l, mask=mask, other=0.0)
        # vals are complex64; Triton doesn't have complex dtype, but the input we pass here
        # is actually torch.view_as_real(rfft), so vals are float2 (real, imag).
        # We only copy real part here; imag part is handled by a separate kernel.
        real_vals = vals  # vals are real numbers (we constructed input as real)
        tl.store(out_real_ptr + base_out + idx * out_stride_l, real_vals, mask=mask)
        start += BLOCK_N


@triton.jit
def _copy_imag_kernel(
    in_cmplx_ptr,        # *complex64, input complex tensor (B, C, L+1)
    out_imag_ptr,        # *f32, output imag tensor (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,         # original seqlen
    in_stride_b: tl.int32, in_stride_c: tl.int32, in_stride_l: tl.int32,
    out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_l: tl.int32,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels
    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c
    length = L + 1
    start = 0
    while start < length:
        idx = start + tl.arange(0, BLOCK_N)
        mask = idx < length
        vals = tl.load(in_cmplx_ptr + base_in + idx * in_stride_l, mask=mask, other=0.0)
        # vals are complex64; imag part is second half of the complex view (imag component)
        imag_vals = vals[1]  # take second element as imag
        tl.store(out_imag_ptr + base_out + idx * out_stride_l, imag_vals, mask=mask)
        start += BLOCK_N


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure float32 input
        x_f32 = x.to(torch.float32)

        batch, channels, seqlen = x_f32.shape
        twoL = 2 * seqlen

        # Compute rfft with implicit zero-padding to 2*seqlen
        x_freq_complex = torch.fft.rfft(x_f32, n=twoL)  # complex64 on CUDA

        # Normalize by 2*seqlen
        x_freq_complex = x_freq_complex / twoL

        # Output real and imag parts; shape is (batch, channels, seqlen+1)
        real_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels to copy real and imaginary parts
        # Note: We pass the complex output into Triton and extract real/imag via view+copy.
        # Construct a real view of the complex tensor (shape (..., L+1, 2)): real, imag interleaved.
        real_imag_view = torch.view_as_real(x_freq_complex)  # shape (B, C, L+1, 2), dtype float32

        # Strides for Triton (element strides, not bytes)
        in_stride_b = real_imag_view.stride(0)
        in_stride_c = real_imag_view.stride(1)
        in_stride_l = real_imag_view.stride(2)  # distance between consecutive k
        in_stride_comp = real_imag_view.stride(3)  # component stride (should be 2)

        out_stride_b = real_out.stride(0)
        out_stride_c = real_out.stride(1)
        out_stride_l = real_out.stride(2)

        grid = (batch * channels,)

        # Copy real part: first component (index 0) of the view
        _copy_real_kernel[grid](
            real_imag_view, real_out,
            batch, channels, seqlen,
            in_stride_b, in_stride_c, in_stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            BLOCK_N=1024,
            num_warps=4,
        )

        # Copy imag part: second component (index 1) of the view
        _copy_imag_kernel[grid](
            real_imag_view, imag_out,
            batch, channels, seqlen,
            in_stride_b, in_stride_c, in_stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
