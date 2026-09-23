import torch
import triton
import triton.language as tl


@triton.jit
def rfft_direct_kernel(
    x_ptr,                  # *f32, input tensor (B, C, L)
    real_out_ptr,           # *f32, output real part (B, C, L+1)
    imag_out_ptr,           # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # seqlen
    stride_b: tl.int32,     # stride for batch in input
    stride_c: tl.int32,     # stride for channel in input
    stride_l: tl.int32,     # stride for last dim in input (usually 1)
    out_stride_b: tl.int32, # stride for batch in output real
    out_stride_c: tl.int32, # stride for channel in output real
    out_stride_l: tl.int32, # stride for last dim in output real (usually 1)
    BLOCK_N: tl.constexpr,  # tile size along time dimension
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    twoL = 2 * L
    inv_twoL = 1.0 / twoL
    L_plus1 = L + 1

    # Base offset for input slice (b, c, :)
    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c

    # For each output frequency index k in [0, L], compute DFT
    k = 0
    while k <= L:
        sum_real = 0.0
        sum_imag = 0.0

        # Iterate over time index j in tiles across padded length twoL
        j_start = 0
        while j_start < twoL:
            j = j_start + tl.arange(0, BLOCK_N)  # vector of time indices
            mask = j < twoL

            # Load x[b, c, j]; for j >= L, value is 0 due to implicit zero-padding
            x_vals = tl.load(x_ptr + base_in + j * stride_l, mask=mask & (j < L), other=0.0)  # [BLOCK_N], float32

            # Compute angle and trig terms
            angle = (2.0 * tl.pi * k * j) / twoL
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)

            # Accumulate contributions: masked load already sets x_vals=0 for j >= L
            # Reduce vector to scalars
            sum_real += tl.sum(x_vals * cos_term, axis=0)
            sum_imag += tl.sum(x_vals * sin_term, axis=0)

            j_start += BLOCK_N

        # Normalize by 2*L (same as dividing by fft_size in original code)
        sum_real = sum_real * inv_twoL
        sum_imag = sum_imag * inv_twoL

        # Store results at frequency index k
        out_offset = base_out + k * out_stride_l  # frequency index corresponds to k-th position in output
        tl.store(real_out_ptr + out_offset, sum_real)
        tl.store(imag_out_ptr + out_offset, sum_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure float32 and on CUDA
        x = x.to(torch.float32).contiguous()
        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Get input and output strides (in elements)
        stride_b, stride_c, stride_l = x.stride()
        # Output strides for real and imag are identical (same tensor layout)
        out_stride_b, out_stride_c, out_stride_l = real_out.stride()

        # Launch Triton kernel: one program per (batch, channel) slice
        grid = (batch * channels,)
        rfft_direct_kernel[grid](
            x, real_out, imag_out,
            batch, channels, L,
            stride_b, stride_c, stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            BLOCK_N=128,  # tile size; 128 is a good default, can tune per workload
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
