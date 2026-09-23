import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                      # *f32, input tensor (B, C, L)
    x_padded_ptr,               # *f32, output tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,                # original seqlen
    stride_b: tl.int32,         # input stride for batch
    stride_c: tl.int32,         # input stride for channel
    stride_l: tl.int32,         # input stride for last dim
    pad_stride_b: tl.int32,     # padded input stride for batch
    pad_stride_c: tl.int32,     # padded input stride for channel
    pad_stride_l: tl.int32,     # padded input stride for last dim
    BLOCK_N: tl.constexpr        # tile size
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b + c * stride_c
    base_out = b * pad_stride_b + c * pad_stride_c
    twoL = 2 * L

    # Copy original L elements into first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * pad_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining positions with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * pad_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
    x_padded_ptr,               # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,               # *f32, output real part (B, C, L+1), contiguous
    imag_out_ptr,               # *f32, output imag part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,                # original seqlen
    real_stride_b: tl.int32,    # real output stride for batch
    real_stride_c: tl.int32,    # real output stride for channel
    real_stride_l: tl.int32,    # real output stride for last dim (should be 1)
    imag_stride_b: tl.int32,    # imag output stride for batch
    imag_stride_c: tl.int32,    # imag output stride for channel
    imag_stride_l: tl.int32,    # imag output stride for last dim (should be 1)
    BLOCK_N: tl.constexpr        # tile size for j
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_out_real = b * real_stride_b + c * real_stride_c
    base_out_imag = b * imag_stride_b + c * imag_stride_c
    twoL = 2 * L

    # For each k in 0..L, compute y_k and store real/imag parts
    k = 0
    while k <= L:
        # Accumulators as scalars
        acc_real = 0.0
        acc_imag = 0.0

        # Iterate over j = 0..2*L-1 in tiles
        j_start = 0
        while j_start < twoL:
            j = j_start + tl.arange(0, BLOCK_N)
            mask = j < twoL

            vals = tl.load(x_padded_ptr + b * x_padded_ptr.stride(0) + c * x_padded_ptr.stride(1) + j * x_padded_ptr.stride(2),
                           mask=mask, other=0.0)
            # Compute cos and sin
            # angle = pi * k * j / (2*L)
            # Triton supports tl.cos/tl.sin with radians
            angle = tl.pi * k * j / twoL
            cosv = tl.cos(angle)
            sinv = tl.sin(angle)

            # Multiply and reduce: dot with real part and accumulate imag part
            # vals is f32; cosv/sinv are f32; masked elements are zero
            prod_real = vals * cosv
            prod_imag = vals * sinv

            # Reduce across the vector
            # Note: tl.sum reduces along the last axis
            acc_real += tl.sum(prod_real, axis=0)
            acc_imag += tl.sum(prod_imag, axis=0)

            j_start += BLOCK_N

        # Normalize by 2*L
        scale = 1.0 / (2.0 * L)

        # Store outputs at index k
        out_real_index = b * real_stride_b + c * real_stride_c + k * real_stride_l
        out_imag_index = b * imag_stride_b + c * imag_stride_c + k * imag_stride_l
        tl.store(real_out_ptr + out_real_index, acc_real * scale)
        tl.store(imag_out_ptr + out_imag_index, acc_imag * scale)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Fused Triton implementation of:
          x_freq = torch.fft.rfft(x.to(torch.float32), n=2*seqlen) / (2*seqlen)
          return x_freq.real.contiguous(), x_freq.imag.contiguous()
        Output shapes: (batch, channels, seqlen+1) for both real and imaginary parts.
        """
        # Ensure float32 and contiguous along last dim
        x = x.to(torch.float32).contiguous()
        batch, channels, L = x.shape
        twoL = 2 * L

        # Allocate padded input (B, C, 2*L), contiguous
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel
        grid = (batch * channels,)
        pad_to_2L_kernel[grid](
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

        # Launch direct rfft kernel
        grid_rfft = (batch * channels,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=256,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
