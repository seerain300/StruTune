import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(
    x_ptr,                  # *f32, input tensor of shape (B, C, L)
    real_out_ptr,           # *f32, output real part of shape (B, C, L+1)
    imag_out_ptr,           # *f32, output imag part of shape (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # seqlen
    stride_b: tl.int32,     # stride for batch in input
    stride_c: tl.int32,     # stride for channel in input
    stride_l: tl.int32,     # stride for last dim in input (normally 1)
    BLOCK_N: tl.constexpr    # tile size along time dimension
):
    pid = tl.program_id(axis=0)  # one program per (batch, channel) slice
    b = pid // channels
    c = pid % channels

    twoL = 2 * L
    inv_twoL = 1.0 / twoL
    L_plus1 = L + 1

    # Base offset for this (b, c) slice
    base_in = b * stride_b + c * stride_c

    # For each output frequency index k in [0, L], compute DFT
    for k in range(0, L_plus1):
        # Accumulators for real and imaginary parts
        sum_real = 0.0
        sum_imag = 0.0

        # Iterate over time index j in tiles
        for start in range(0, twoL, BLOCK_N):
            j = start + tl.arange(0, BLOCK_N)  # vector of indices
            mask = j < twoL
            # Load x[b, c, j]
            x_vals = tl.load(x_ptr + base_in + j * stride_l, mask=mask, other=0.0)  # shape: [BLOCK_N]

            # Compute cos and sin for this k and all j in the tile
            if k == 0:
                cos_term = 1.0
                sin_term = 0.0
            else:
                angle = (tl.pi * k * j) / twoL
                cos_term = tl.cos(angle)
                sin_term = tl.sin(angle)

            # Accumulate contributions
            # x_vals is real, cos_term and sin_term are real -> sum_real += x * cos; sum_imag += x * sin
            sum_real += tl.sum(x_vals * cos_term, axis=0)
            sum_imag += tl.sum(x_vals * sin_term, axis=0)

        # Normalize by 2*L
        sum_real = sum_real * inv_twoL
        sum_imag = sum_imag * inv_twoL

        # Store results at frequency index k (note: outputs are (B,C,L+1), so index k directly)
        # We can compute output base for (b,c) similarly
        out_base = b * stride_b + c * stride_c
        # Store to real_out[b, c, k] and imag_out[b, c, k]
        # Since real_out and imag_out are (B,C,L+1), and Triton pointers assume linearized indexing,
        # we need to compute the linearized offset. PyTorch tensors are row-major; for contiguous (B,C,L+1),
        # the linear offset is b*(C*(L+1)) + c*(L+1) + k.
        # But here we pass pointers, so we must ensure real_out_ptr and imag_out_ptr are contiguous.
        # We'll write using linear index computed as:
        # real_out has shape (batch, channels, L+1); contiguous layout implies linear index = b*(channels*(L+1)) + c*(L+1) + k
        # Triton will handle pointer arithmetic; we can compute the offset via b*stride_b + c*stride_c + k*stride_l? No, that's input.
        # Better: create outputs as contiguous tensors and use linear indexing:
        # For contiguous (B,C,L+1), the linear offset for (b,c,k) is b*(channels*(L+1)) + c*(L+1) + k.
        # Triton doesn't need explicit stride for output if we pass contiguous pointers; we can compute offsets directly:
        # We'll assume real_out_ptr and imag_out_ptr are contiguous (default for torch.empty).
        # Therefore, we can compute:
        real_offset = b * (channels * (L + 1)) + c * (L + 1) + k
        imag_offset = b * (channels * (L + 1)) + c * (L + 1) + k

        # Write outputs
        tl.store(real_out_ptr + real_offset, sum_real)
        tl.store(imag_out_ptr + imag_offset, sum_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of the original run function:
        - Computes rfft of each (batch, channel) slice along the last dimension, zero-padded to 2*seqlen.
        - Returns real and imaginary parts (batch, channels, seqlen+1).
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
        # Choose a tile size for j. 256 is a reasonable default; you can tune it.
        BLOCK_N = 256

        rfft_real_kernel[(M,)](
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
