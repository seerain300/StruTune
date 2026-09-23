import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                  # *f32, input tensor (B, C, L)
    x_padded_ptr,           # *f32, output tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    stride_b_in: tl.int32,  # input stride for batch
    stride_c_in: tl.int32,  # input stride for channel
    stride_l_in: tl.int32,  # input stride for last dim
    stride_b_out: tl.int32, # output stride for batch
    stride_c_out: tl.int32, # output stride for channel
    stride_l_out: tl.int32, # output stride for last dim
    BLOCK_N: tl.constexpr,   # tile size
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b_in + c * stride_c_in
    base_out = b * stride_b_out + c * stride_c_out
    twoL = 2 * L

    # Copy original L elements into first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l_in, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * stride_l_out, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining positions with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * stride_l_out, zero_vec, mask=mask)
        start += BLOCK_N


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
    BLOCK_N: tl.constexpr,   # tile size for j
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * x_padded_ptr.stride(0) + c * x_padded_ptr.stride(1)
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Compute y[k] for k = 0..L and store into real_out and imag_out at index k
    k = 0
    while k <= L:
        real_acc = 0.0
        imag_acc = 0.0

        # Iterate over j = 0..2*L-1 in tiles
        j_start = 0
        while j_start < twoL:
            j = j_start + tl.arange(0, BLOCK_N)
            mask = j < twoL
            vals_j = tl.load(x_padded_ptr + base_in + j * x_padded_ptr.stride(2), mask=mask, other=0.0)
            angle = (2.0 * 3.141592653589793 * k * j) / twoL
            cosv = tl.cos(angle)
            sinv = tl.sin(angle)
            real_acc += tl.sum(vals_j * cosv, axis=0)
            imag_acc += tl.sum(vals_j * sinv, axis=0)
            j_start += BLOCK_N

        # Normalize by 2*L
        real_acc = real_acc / (2.0 * L)
        imag_acc = imag_acc / (2.0 * L)

        # Store results into outputs at index k
        # Output is contiguous (B, C, L+1), so linear index is (b*channels + c)*(L+1) + k
        out_index = b * channels * (L + 1) + c * (L + 1) + k
        # real_out_ptr is (B, C, L+1) contiguous; its stride(0) = C*(L+1), stride(1) = L+1, stride(2) = 1
        tl.store(real_out_ptr + out_index, real_acc)
        tl.store(imag_out_ptr + out_index, imag_acc)

        k += 1


@triton.jit
def extract_real_kernel(
    complex_out_ptr,        # *f32 complex, assumed interleaved (real, imag) per (B,C,L+1)
    real_out_ptr,           # *f32, output real part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    comp_stride_b: tl.int32, # stride for batch in complex output
    comp_stride_c: tl.int32, # stride for channel in complex output
    comp_stride_l: tl.int32, # stride for last dim in complex output
    out_stride_b: tl.int32,  # stride for batch in real output
    out_stride_c: tl.int32,  # stride for channel in real output
    out_stride_l: tl.int32,  # stride for last dim in real output
    BLOCK_N: tl.constexpr,   # tile size for k
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_comp = b * comp_stride_b + c * comp_stride_c
    base_out = b * out_stride_b + c * out_stride_c

    k = 0
    while k <= L:
        # complex_out_ptr stores interleaved real/imag: real at index 2*k, imag at 2*k+1
        real_val = tl.load(complex_out_ptr + base_comp + k * comp_stride_l)
        tl.store(real_out_ptr + base_out + k * out_stride_l, real_val)
        k += 1


@triton.jit
def extract_imag_kernel(
    complex_out_ptr,        # *f32 complex, assumed interleaved (real, imag) per (B,C,L+1)
    imag_out_ptr,           # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,            # original seqlen
    comp_stride_b: tl.int32, # stride for batch in complex output
    comp_stride_c: tl.int32, # stride for channel in complex output
    comp_stride_l: tl.int32, # stride for last dim in complex output
    out_stride_b: tl.int32,  # stride for batch in imag output
    out_stride_c: tl.int32,  # stride for channel in imag output
    out_stride_l: tl.int32,  # stride for last dim in imag output
    BLOCK_N: tl.constexpr,   # tile size for k
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_comp = b * comp_stride_b + c * comp_stride_c
    base_out = b * out_stride_b + c * out_stride_c

    k = 0
    while k <= L:
        imag_val = tl.load(complex_out_ptr + base_comp + comp_stride_l + k * comp_stride_l)
        tl.store(imag_out_ptr + base_out + k * out_stride_l, imag_val)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen), dtype float32
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        assert x.dtype == torch.float32, "Input must be float32."
        batch, channels, L = x.shape

        # Construct zero-padded input of length 2*L per (batch, channel) slice
        x_padded = torch.empty((batch, channels, 2 * L), dtype=torch.float32, device=x.device)

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

        # Launch direct rfft kernel to compute real and imaginary parts
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
