import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,            # *f32, input x of shape (B, C, L)
    x_padded_ptr,     # *f32, output padded x of shape (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,      # original seqlen
    stride_xb: tl.int32,
    stride_xc: tl.int32,
    stride_xl: tl.int32,
    stride_pdb: tl.int32,
    stride_pdc: tl.int32,
    stride_pdl: tl.int32,
    BLOCK_N: tl.constexpr,
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_x = b * stride_xb + c * stride_xc
    base_pd = b * stride_pdb + c * stride_pdc

    # Copy original L elements to first half
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_x + j * stride_xl, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_pd + j * stride_pdl, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining positions with zeros (second half)
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    twoL = 2 * L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_pd + j * stride_pdl, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_exp_kernel(
    x_padded_ptr,          # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,          # *f32, output real part (B, C, L+1)
    imag_out_ptr,          # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,           # original seqlen
    out_stride_b: tl.int32,
    out_stride_c: tl.int32,
    out_stride_l: tl.int32,
    BLOCK_N: tl.constexpr,
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_pd = b * x_padded_ptr.stride(0) + c * x_padded_ptr.stride(1)
    twoL = 2 * L

    # We will compute y_k for k = 0..L and store into real_out[b,c,k] and imag_out[b,c,k]
    k = 0
    while k <= L:
        acc_real = tl.zeros((), dtype=tl.float32)
        acc_imag = tl.zeros((), dtype=tl.float32)

        j = 0
        while j < twoL:
            j_vec = j + tl.arange(0, BLOCK_N)
            mask_j = j_vec < twoL

            vals = tl.load(x_padded_ptr + base_pd + j_vec * x_padded_ptr.stride(2), mask=mask_j, other=0.0)

            # angle for DFT: 2*pi*k*j / (2*L) = pi*k*j/L
            angle = (tl.float32(k) * tl.float32(twoL)) * tl.pi * j_vec / tl.float32(twoL)
            angle = (tl.float32(k) * tl.pi * j_vec) / tl.float32(L)

            # Use exponential form for cosine/sine: cos(angle) = (e^{i*angle} + e^{-i*angle})/2
            # sin(angle) = (e^{i*angle} - e^{-i*angle})/(2i)
            i = 1.0j  # Triton treats this as a Python complex; expressions will be handled by tl operations
            e_pos = tl.exp(1.0 * 1.0j * angle)
            e_neg = tl.exp(-1.0 * 1.0j * angle)
            cos_term = (e_pos + e_neg).real
            sin_term = (e_pos - e_neg).imag

            acc_real += tl.sum(vals * cos_term, axis=0)
            acc_imag += tl.sum(vals * sin_term, axis=0)

            j += BLOCK_N

        scale = 1.0 / tl.float32(twoL)
        acc_real = acc_real * scale
        acc_imag = acc_imag * scale

        out_base = b * out_stride_b + c * out_stride_c
        tl.store(real_out_ptr + out_base + k * out_stride_l, acc_real)
        tl.store(imag_out_ptr + out_base + k * out_stride_l, acc_imag)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure input is CUDA and contiguous
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        x = x.contiguous()
        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L

        # Cast to float32
        x_f32 = x.to(torch.float32)

        # Allocate padded tensor (B, C, 2*L)
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel: copy first half, zero second half
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x_f32, x_padded,
            batch, channels, L,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel computing y_k for k=0..L
        grid_rfft = (batch * channels,)
        rfft_direct_exp_kernel[grid_rfft](
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
