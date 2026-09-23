import torch
import triton
import triton.language as tl


@triton.jit
def rfft_zero_pad_kernel(
    x_ptr,                  # *f32, input tensor (B, C, L)
    real_out_ptr,           # *f32, output real part (B, C, L+1)
    imag_out_ptr,           # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # seqlen
    stride_b: tl.int32,     # stride for batch in input
    stride_c: tl.int32,     # stride for channel in input
    stride_l: tl.int32,     # stride for last dim in input (usually 1)
    BLOCK_N: tl.constexpr    # tile size along padded time dimension (twoL)
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    twoL = 2 * L
    inv_twoL = 1.0 / twoL

    # Base offset for input slice (b, c, :)
    base_in = b * stride_b + c * stride_c

    # For each output frequency index k in [0, L], compute DFT over padded time length twoL
    k = 0
    while k <= L:
        sum_real = 0.0
        sum_imag = 0.0

        # Iterate over time index j in tiles across the padded length
        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)  # vector of time indices, 0..twoL-1
            mask = j < twoL
            # Load original x[b, c, j] only for j < L; otherwise treat as 0 (implicit zero-padding)
            x_vals = tl.load(x_ptr + base_in + j * stride_l, mask=mask & (j < L), other=0.0)  # real values

            # Compute cos and sin for this k and all j in the tile
            angle = (tl.pi * k * j) / twoL
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)

            # Accumulate contributions: x_vals are 0 for j >= L due to mask in load
            sum_real += tl.sum(x_vals * cos_term, axis=0)
            sum_imag += tl.sum(x_vals * sin_term, axis=0)

            start += BLOCK_N

        # Normalize by 2*L (same as dividing by fft_size in the original code)
        sum_real = sum_real * inv_twoL
        sum_imag = sum_imag * inv_twoL

        # Store results at frequency index k
        # Output is contiguous (B, C, L+1); linear offset for (b, c, k) is b*(channels*(L+1)) + c*(L+1) + k
        out_offset = b * (channels * (L + 1)) + c * (L + 1) + k
        tl.store(real_out_ptr + out_offset, sum_real)
        tl.store(imag_out_ptr + out_offset, sum_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of the original run function:
        - Implicitly zero-pad each (batch, channel) slice to length 2*seqlen, then compute the real-input DFT (rfft).
        - Return real and imaginary parts (batch, channels, seqlen+1).
        """
        assert x.dim() == 3, "Input must be of shape (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L

        # Ensure float32 and CUDA for Triton
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if not x.is_cuda:
            x = x.cuda()

        # Allocate outputs: shape (B,C,L+1) for real and imag
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Get input strides (in elements)
        stride_b, stride_c, stride_l = x.stride()

        # Launch kernel: grid over M = B*C
        M = batch * channels

        # Choose a tile size for j. 256 is a reasonable default; you can tune it based on hardware.
        BLOCK_N = 256

        rfft_zero_pad_kernel[(M,)](
            x, real_out, imag_out,
            batch, channels, L,
            stride_b, stride_c, stride_l,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
