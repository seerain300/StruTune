import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,              # *f32, input tensor (B, C, L), contiguous
    x_padded_ptr,       # *f32, output tensor (B, C, 2L), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,        # seqlen
    stride_b: tl.int32, # stride for batch in input
    stride_c: tl.int32, # stride for channel in input
    stride_l: tl.int32, # stride for last dim in input (usually 1)
    out_stride_b: tl.int32,  # stride for batch in padded output
    out_stride_c: tl.int32,  # stride for channel in padded output
    out_stride_l: tl.int32,  # stride for last dim in padded output (usually 1)
    BLOCK_N: tl.constexpr   # tile size along last dim
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    twoL = 2 * L
    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c

    # Copy original L elements to the first L positions of the padded tensor
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Fill the remaining positions with zeros (use a zero vector to avoid dtype issues)
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Keep the input device; only cast to float32 for numerical stability (as original code does).
        x = x.to(torch.float32)

        batch, channels, seqlen = x.shape
        L = seqlen
        twoL = 2 * L

        # Allocate padded input: (batch, channels, 2*L)
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Get input and output strides (elements)
        stride_b, stride_c, stride_l = x.stride()
        out_stride_b, out_stride_c, out


def run(*args):
    return ModelNew()(*args)
