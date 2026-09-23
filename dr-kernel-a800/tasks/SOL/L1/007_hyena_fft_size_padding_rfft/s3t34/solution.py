import torch
import triton
import triton.language as tl

@triton.jit
def pad_to_2L_kernel(
    x_ptr,                  # *f32, input tensor (B, C, L)
    x_padded_ptr,           # *f32, output padded tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    stride_b: tl.int32,     # input stride for batch
    stride_c: tl.int32,     # input stride for channel
    stride_l: tl.int32,     # input stride for last dim
    out_stride_b: tl.int32, # padded output stride for batch
    out_stride_c: tl.int32, # padded output stride for channel
    out_stride_p: tl.int32, # padded output stride for last dim (padding index)
    BLOCK_N: tl.constexpr    # tile size for j
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
        tl.store(x_padded_ptr + base_out + j * out_stride_p, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining positions with zeros (j from L to 2*L-1)
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_p, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
    x_padded_ptr,           # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,           # *f32, output real part (B, C, L+1)
    imag_out_ptr,           # *f32, output imag part (B, C, L+1)
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

    # Compute DFT for k = 0..L
    k = 0
    while k <= L:
        acc_real = tl.zeros([1], dtype=tl.float32)
        acc_imag = tl.zeros([1], dtype=tl.float32)
        # Accumulate over j from 0 to 2*L-1
        j = 0
        while j < twoL:
            jj = j + tl.arange(0, BLOCK_N)
            mask = jj < twoL
            vals = tl.load(x_padded_ptr + (b * (x_padded_ptr.stride(0)) + c * (x_padded_ptr.stride(1))) + jj * (x_padded_ptr.stride(2)), mask=mask, other=0.0)
            # Compute angles
            angle = (2.0 * 3.141592653589793) * float(k) * float(jj) / float(twoL)
            cosv = tl.cos(angle)
            sinv = tl.sin(angle)
            # Accumulate real and imag parts: sum over tile
            # We need to sum the vector into scalars
            # Use masked multiply and reduce by summing across the vector
            real_part = vals * cosv
            imag_part = vals * sinv
            acc_real += tl.sum(real_part, axis=0)
            acc_imag += tl.sum(imag_part, axis=0)
            j += BLOCK_N
        # Normalize by 2*L
        norm = 1.0 / float(twoL)
        acc_real = acc_real * norm
        acc_imag = acc_imag * norm
        # Store at index k (which is 0..L)
        tl.store(real_out_ptr + base_out + k * out_stride_l, acc_real)
        tl.store(imag_out_ptr + base_out + k * out_stride_l, acc_imag)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Triton-only implementation; no torch operations here
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        batch, channels, L = x.shape
        twoL = 2 * L

        # Allocate padded input (B, C, 2*L)
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch pad kernel
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1)
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel
        grid_rfft = (batch * channels,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
