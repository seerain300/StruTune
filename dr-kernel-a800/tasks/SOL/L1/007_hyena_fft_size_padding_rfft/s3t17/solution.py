import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                       # *f32, input tensor (B, C, L), contiguous
    x_padded_ptr,                # *f32, output tensor (B, C, 2*L), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,                 # original seqlen
    BLOCK_N: tl.constexpr        # tile size for j
):
    """
    Construct zero-padded input of length 2*L per (batch, channel) slice.
    x_ptr points to a flat buffer of size B*C*L; x_padded_ptr points to B*C*2*L.
    One Triton program handles one (b, c) slice.
    """
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    # Base indices for this (b, c) slice
    base_in = (b * channels + c) * L
    base_out = (b * channels + c) * (2 * L)

    twoL = 2 * L

    # Copy first L elements: j in [0, L)
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining elements with zeros: j in [L, 2*L)
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def compute_rfft_real_imag_kernel(
    x_padded_ptr,                # *f32, input padded tensor (B, C, 2*L), contiguous
    real_out_ptr,                # *f32, output real part (B, C, L+1), contiguous
    imag_out_ptr,                # *f32, output imag part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,                 # original seqlen
    BLOCK_N: tl.constexpr        # tile size for j
):
    """
    Compute real-input DFT coefficients for k = 0..L on x_padded of length 2*L.
    One Triton program handles one (b, c) slice and writes real/imag parts at positions 0..L.
    """
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    twoL = 2 * L
    inv_twoL = 1.0 / twoL

    base_out = (b * channels + c) * (L + 1)

    # For each k, accumulate over j from 0 to 2*L-1
    k = 0
    while k <= L:
        acc_real = 0.0
        acc_imag = 0.0

        start = 0
        while start < twoL:
            j = start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            vals = tl.load(x_padded_ptr + (b * channels + c) * twoL + j, mask=mask, other=0.0)
            angle = 2.0 * 3.141592653589793 * k * j / twoL
            contrib_real = tl.sum(vals * tl.cos(angle), axis=0)
            contrib_imag = tl.sum(vals * tl.sin(angle), axis=0)
            acc_real += contrib_real
            acc_imag += contrib_imag
            start += BLOCK_N

        real_val = acc_real * inv_twoL
        imag_val = acc_imag * inv_twoL

        tl.store(real_out_ptr + base_out + k, real_val)
        tl.store(imag_out_ptr + base_out + k, imag_val)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor x with shape (batch, channels, seqlen)
        x = args[0] if len(args) == 1 else args[0]
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L

        # Cast to float32 for numerical stability
        x_f32 = x.to(torch.float32)
        # Ensure contiguous buffers for Triton kernels
        x_f32 = x_f32.contiguous()

        # Allocate padded input tensor
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch pad kernel: one program per (batch, channel) slice
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x_f32, x_padded,
            batch, channels, L,
            BLOCK_N=256,
            num_warps=4,
        )

        # Allocate outputs
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch DFT kernel: one program per (batch, channel) slice
        grid_dft = (batch * channels,)
        compute_rfft_real_imag_kernel[grid_dft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            BLOCK_N=256,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
