import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,           # *f32, input tensor (B, C, L)
    x_padded_ptr,    # *f32, output padded tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,          # original seqlen
    stride_x_b: tl.int32, # input strides
    stride_x_c: tl.int32,
    stride_x_l: tl.int32,
    stride_p_b: tl.int32, # padded strides
    stride_p_c: tl.int32,
    stride_p_l: tl.int32,
    BLOCK_N: tl.constexpr  # tile size for j
):
    # One program per (b, c)
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_x = b * stride_x_b + c * stride_x_c
    base_p = b * stride_p_b + c * stride_p_c
    twoL = 2 * L

    # Copy first L elements
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_x + j * stride_x_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_p + j * stride_p_l, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining L elements with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_p + j * stride_p_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
    x_padded_ptr,           # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,           # *f32, output real part (B, C, L+1), contiguous
    imag_out_ptr,           # *f32, output imag part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen, output bins = L+1
    real_stride_b: tl.int32,
    real_stride_c: tl.int32,
    real_stride_l: tl.int32,
    imag_stride_b: tl.int32,
    imag_stride_c: tl.int32,
    imag_stride_l: tl.int32,
    BLOCK_N: tl.constexpr
):
    # One program per (b, c)
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_p = b * x_padded_ptr.stride(0) + c * x_padded_ptr.stride(1)
    base_real = b * real_stride_b + c * real_stride_c
    base_imag = b * imag_stride_b + c * imag_stride_c
    twoL = 2 * L

    # Accumulators for real and imag
    acc_real = tl.zeros((), dtype=tl.float32)
    acc_imag = tl.zeros((), dtype=tl.float32)

    # Loop over k from 0 to L (inclusive), compute real and imag parts and store
    k = 0
    while k <= L:
        # Accumulate sum over j=0..2*L-1
        acc_real = 0.0
        acc_imag = 0.0
        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            vals = tl.load(x_padded_ptr + base_p + j * x_padded_ptr.stride(2), mask=mask, other=0.0)
            # angle = 2*pi*k*j/(2*L) = pi*k*j/L
            angle = (tl.float32(k) * tl.float32(j) * 3.141592653589793) / tl.float32(twoL)
            cosv = tl.cos(angle)
            sinv = tl.sin(angle)
            acc_real += tl.sum(vals * cosv, axis=0)
            acc_imag += tl.sum(vals * sinv, axis=0)
            start += BLOCK_N

        # Normalize by 2*L
        norm = 1.0 / tl.float32(twoL)
        acc_real = acc_real * norm
        acc_imag = acc_imag * norm

        # Store to output at index k (output length is L+1)
        out_idx = k  # since k ranges 0..L
        tl.store(real_out_ptr + base_real + out_idx * real_stride_l, acc_real)
        tl.store(imag_out_ptr + base_imag + out_idx * imag_stride_l, acc_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton implementation that computes:
          y = torch.fft.rfft(x, n=2*L) / (2*L), where L = x.shape[-1]
        Returns real and imaginary parts of shape (B, C, L+1).
        """
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape
        device = x.device
        dtype = torch.float32

        # Ensure input is float32 on device
        x_f32 = x.to(torch.float32)

        # Allocate padded input (B, C, 2*L)
        x_padded = torch.empty((B, C, 2 * L), dtype=dtype, device=device)

        # Launch padding kernel: one program per (b, c)
        grid_pad = (B * C,)
        pad_to_2L_kernel[grid_pad](
            x_f32, x_padded,
            B, C, L,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs: real and imag, shape (B, C, L+1)
        real_out = torch.empty((B, C, L + 1), dtype=dtype, device=device)
        imag_out = torch.empty((B, C, L + 1), dtype=dtype, device=device)

        # Launch direct rfft kernel: one program per (b, c), computes k=0..L
        grid_rfft = (B * C,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            B, C, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
