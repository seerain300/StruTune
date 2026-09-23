import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,        # *f32, input tensor (B, C, L)
    x_padded_ptr, # *f32, output tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    in_stride_b: tl.int32,  # strides for input
    in_stride_c: tl.int32,
    in_stride_l: tl.int32,
    out_stride_b: tl.int32, # strides for output
    out_stride_c: tl.int32,
    out_stride_l: tl.int32,
    BLOCK: tl.constexpr,    # tile size for j
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * in_stride_b + c * in_stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Copy original L elements into first L positions of x_padded
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * in_stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK

    # Fill remaining positions with zeros (j >= L)
    zero_vec = tl.zeros([BLOCK], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK


@triton.jit
def rfft_direct_kernel(
    x_padded_ptr,           # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,           # *f32, output real part (B, C, L+1), contiguous
    imag_out_ptr,           # *f32, output imag part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    out_stride_b: tl.int32, # stride for batch in output
    out_stride_c: tl.int32, # stride for channel in output
    out_stride_l: tl.int32, # stride for last dim in output (should be 1)
    BLOCK: tl.constexpr,    # tile size for j
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    k = 0
    while k <= L:
        # Accumulate real and imag parts for frequency index k
        acc_real = 0.0
        acc_imag = 0.0

        # Iterate over j = 0..2*L-1 in tiles
        j = 0
        while j < twoL:
            jj = j + tl.arange(0, BLOCK)
            mask = jj < twoL
            vals = tl.load(x_padded_ptr + base_out + jj * out_stride_l, mask=mask, other=0.0)
            # Compute angle = 2*pi*k*jj/(2*L)
            angle = 2.0 * 3.141592653589793 * float(k) * (jj.to(tl.float32)) / float(twoL)
            cos_t = tl.cos(angle)
            sin_t = tl.sin(angle)
            # Multiply elementwise and reduce
            acc_real += tl.sum(vals * cos_t, axis=0)
            acc_imag += tl.sum(vals * sin_t, axis=0)
            j += BLOCK

        # Scale by 1/(2*L)
        scale = 1.0 / float(twoL)
        acc_real = acc_real * scale
        acc_imag = acc_imag * scale

        # Store into output at index k (k ranges 0..L, output length is L+1)
        out_index = k
        tl.store(real_out_ptr + base_out + out_index * out_stride_l, acc_real)
        tl.store(imag_out_ptr + base_out + out_index * out_stride_l, acc_imag)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure float32 and contiguous input
        x_f32 = x.to(torch.float32).contiguous()
        batch, channels, seqlen = x_f32.shape
        L = seqlen
        twoL = 2 * L

        # Allocate padded input tensor (B, C, 2*L), contiguous
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x_f32, x_padded,
            batch, channels, L,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK=1024,
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
            BLOCK=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
