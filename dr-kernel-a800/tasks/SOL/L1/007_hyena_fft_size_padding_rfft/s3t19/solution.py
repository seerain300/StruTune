import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                      # *f32, input original tensor (B, C, L)
    x_padded_ptr,               # *f32, output padded tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,                # original seqlen
    stride_b: tl.int32,         # input stride for batch
    stride_c: tl.int32,         # input stride for channel
    stride_l: tl.int32,         # input stride for last dim
    out_stride_b: tl.int32,     # output stride for batch
    out_stride_c: tl.int32,     # output stride for channel
    out_stride_l: tl.int32,     # output stride for last dim (should be 1 for contiguous)
    BLOCK_N: tl.constexpr       # tile size for j
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

    # Fill remaining positions with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def compute_rfft_real_imag_kernel(
    x_padded_ptr,               # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,               # *f32, output real part (B, C, L+1), contiguous
    imag_out_ptr,               # *f32, output imag part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,                # original seqlen
    out_stride_b: tl.int32,     # stride for batch in output
    out_stride_c: tl.int32,     # stride for channel in output
    out_stride_l: tl.int32,     # stride for last dim in output (should be 1)
    BLOCK_N: tl.constexpr       # tile size for j
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Accumulators for real and imaginary parts
    acc_real = tl.zeros([1], dtype=tl.float32)
    acc_imag = tl.zeros([1], dtype=tl.float32)

    # Loop over k = 0..L (seqlen). Triton supports scalar loop variables.
    k = 0
    while k <= L:
        angle = (2.0 * 3.141592653589793 * k) / twoL  # compute angle once per k

        # Accumulate over j = 0..2*L-1
        j = 0
        while j < twoL:
            j_vec = j + tl.arange(0, BLOCK_N)
            mask = j_vec < twoL
            vals = tl.load(x_padded_ptr + (b * 0 + c * 0 + j_vec),  # base_in not used, we can use x_padded_ptr directly
                          mask=mask, other=0.0)

            # Compute cos and sin for each j in the tile
            cos_term = tl.cos(angle * j_vec)
            sin_term = tl.sin(angle * j_vec)

            # Accumulate: vals * (cos + i*sin), but since vals are real, multiply both parts
            acc_real += tl.sum(vals * cos_term, axis=0)
            acc_imag += tl.sum(vals * sin_term, axis=0)

            j += BLOCK_N

        # Store results for this k into the output at index k
        out_idx = k
        # Write into contiguous real and imag outputs
        tl.store(real_out_ptr + base_out + out_idx * out_stride_l, acc_real)
        tl.store(imag_out_ptr + base_out + out_idx * out_stride_l, acc_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Expect input shape (batch, channels, seqlen)
        assert x.ndim == 3, "Input must be 3D (batch, channels, seqlen)"
        batch, channels, L = x.shape
        twoL = 2 * L

        # Ensure contiguous and float32
        x = x.contiguous()
        x = x.to(torch.float32)

        # Allocate padded input
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel: one program per (batch, channel) slice
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch DFT kernel: one program per (batch, channel) slice
        grid_dft = (batch * channels,)
        compute_rfft_real_imag_kernel[grid_dft](
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
