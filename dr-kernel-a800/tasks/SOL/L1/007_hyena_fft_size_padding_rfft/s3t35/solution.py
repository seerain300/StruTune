import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                 # *f32, input (B, C, L)
    x_padded_ptr,          # *f32, output (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,           # original seqlen
    stride_b_in: tl.int32, # x stride for batch
    stride_c_in: tl.int32, # x stride for channel
    stride_l_in: tl.int32, # x stride for last dim
    stride_b_out: tl.int32, # x_padded stride for batch
    stride_c_out: tl.int32, # x_padded stride for channel
    stride_l_out: tl.int32, # x_padded stride for last dim
    BLOCK_N: tl.constexpr
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b_in + c * stride_c_in
    base_out = b * stride_b_out + c * stride_c_out
    twoL = 2 * L

    # Copy first L elements
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l_in, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * stride_l_out, vals, mask=mask)
        start += BLOCK_N

    # Zero-fill remaining elements
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * stride_l_out, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
    x_padded_ptr,          # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,          # *f32, output real part (B, C, L+1)
    imag_out_ptr,          # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,           # original seqlen
    stride_b_out_r: tl.int32,  # real output stride for batch
    stride_c_out_r: tl.int32,  # real output stride for channel
    stride_l_out_r: tl.int32,  # real output stride for last dim
    stride_b_out_i: tl.int32,  # imag output stride for batch
    stride_c_out_i: tl.int32,  # imag output stride for channel
    stride_l_out_i: tl.int32,  # imag output stride for last dim
    BLOCK_K: tl.constexpr,   # tile size for k
    BLOCK_J: tl.constexpr    # tile size for j
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_out_r = b * stride_b_out_r + c * stride_c_out_r
    base_out_i = b * stride_b_out_i + c * stride_c_out_i
    twoL = 2 * L

    # Accumulators for real and imag
    acc_real = tl.zeros([1], dtype=tl.float32)
    acc_imag = tl.zeros([1], dtype=tl.float32)

    k = 0
    while k <= L:
        # Accumulate over j from 0 to 2*L - 1
        j = 0
        while j < twoL:
            # Load x_padded[b, c, j]
            xj = tl.load(x_padded_ptr + (b * x_padded_ptr.stride(0) + c * x_padded_ptr.stride(1)) + j * x_padded_ptr.stride(2), mask=j < twoL, other=0.0)
            # Compute angle in radians: 2*pi*k*j/(2*L) = pi*k*j/L
            theta = (tl.float32(k) * tl.float32(j) * 3.141592653589793) / tl.float32(L)
            # cos and sin (scalar)
            c_kj = tl.cos(theta)
            s_kj = tl.sin(theta)
            # Contribution to real and imag
            acc_real += xj * c_kj
            acc_imag += xj * s_kj
            j += BLOCK_J
        # Normalize by (2*L)
        inv_twoL = 1.0 / tl.float32(twoL)
        y_real = acc_real * inv_twoL
        y_imag = acc_imag * inv_twoL
        # Store to outputs at index k
        tl.store(real_out_ptr + base_out_r + k * stride_l_out_r, y_real)
        tl.store(imag_out_ptr + base_out_i + k * stride_l_out_i, y_imag)
        k += BLOCK_K


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure input is float32 on CUDA
        assert x.dim() == 3, "Input must be 3D tensor (batch, channels, seqlen)"
        assert x.is_cuda, "Input must be on CUDA device"
        batch, channels, L = x.shape
        twoL = 2 * L

        # Allocate padded input (B, C, 2*L), float32
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=256,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts)
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel
        grid_rfft = (batch * channels,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_K=1,          # iterate k one-by-one for correctness
            BLOCK_J=256,        # tile size for j
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
