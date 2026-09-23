import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,           # *f32, input tensor (B, C, L)
    x_padded_ptr,    # *f32, output tensor (B, C, 2*L)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,         # original seqlen
    stride_b: tl.int32,  # input stride for batch
    stride_c: tl.int32,  # input stride for channel
    stride_l: tl.int32,  # input stride for last dim
    pad_stride_b: tl.int32,
    pad_stride_c: tl.int32,
    pad_stride_l: tl.int32,
    BLOCK_N: tl.constexpr,
):
    # One program per (batch, channel)
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
    x_padded_ptr,     # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,     # *f32, output real part (B, C, L+1)
    imag_out_ptr,     # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,              # original seqlen
    out_stride_b: tl.int32,   # output stride for batch
    out_stride_c: tl.int32,   # output stride for channel
    out_stride_l: tl.int32,   # output stride for last dim (should be 1)
    pad_stride_b: tl.int32,   # input padded stride for batch
    pad_stride_c: tl.int32,   # input padded stride for channel
    pad_stride_l: tl.int32,   # input padded stride for last dim
    BLOCK_N: tl.constexpr,    # tile size for j
):
    # One program per (batch, channel)
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_out = b * out_stride_b + c * out_stride_c
    base_pad = b * pad_stride_b + c * pad_stride_c
    twoL = 2 * L

    # Compute y_k for k = 0..L
    k = 0
    while k <= L:
        r_sum = tl.zeros((), dtype=tl.float32)
        i_sum = tl.zeros((), dtype=tl.float32)

        # Sum over j = 0..2*L-1
        start_j = 0
        while start_j < twoL:
            j = start_j + tl.arange(0, BLOCK_N)
            mask_j = j < twoL

            xj = tl.load(x_padded_ptr + base_pad + j * pad_stride_l, mask=mask_j, other=0.0)

            angle = (2.0 * 3.141592653589793) * k * j / twoL
            cos_t = tl.cos(angle)
            sin_t = tl.sin(angle)

            # Accumulate per lane; tl.sum reduces over the vector to scalar
            r_sum += tl.sum(xj * cos_t, axis=0)
            i_sum += tl.sum(xj * sin_t, axis=0)

            start_j += BLOCK_N

        # Normalize by 2*L
        r_sum = r_sum / twoL
        i_sum = i_sum / twoL

        # Store real and imag parts at index k
        out_index = k  # 0 <= k <= L -> output index 0..L
        tl.store(real_out_ptr + base_out + out_index * out_stride_l, r_sum)
        tl.store(imag_out_ptr + base_out + out_index * out_stride_l, i_sum)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-ONLY implementation:
        - Pads input to 2*seqlen and computes real-input DFT explicitly.
        - Returns real and imaginary parts of normalized output as (B, C, seqlen+1).
        No torch operations in forward; only Triton kernel launches and tensor allocations.
        """
        assert x.dim() == 3, "Input must be 3D (batch, channels, seqlen)"
        B, C, L = x.shape
        device = x.device

        # Ensure float32 and contiguous input
        x = x.contiguous().to(torch.float32)

        # Prepare padded input tensor (B, C, 2*L), zeros in second half
        x_padded = torch.empty((B, C, 2 * L), dtype=torch.float32, device=device)

        # Strides
        stride_b, stride_c, stride_l = x.stride()
        pad_stride_b, pad_stride_c, pad_stride_l = x_padded.stride()

        # Launch padding kernel: one program per (batch, channel)
        grid_pad = (B * C,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            B, C, L,
            stride_b, stride_c, stride_l,
            pad_stride_b, pad_stride_c, pad_stride_l,
            BLOCK_N=256,
            num_warps=4,
        )

        # Allocate outputs: real and imag parts of shape (B, C, L+1), contiguous
        real_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=device)
        imag_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=device)
        out_stride_b, out_stride_c, out_stride_l = real_out.stride()

        # Launch direct DFT kernel: one program per (batch, channel)
        grid_rfft = (B * C,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            B, C, L,
            out_stride_b, out_stride_c, out_stride_l,
            pad_stride_b, pad_stride_c, pad_stride_l,
            BLOCK_N=256,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
