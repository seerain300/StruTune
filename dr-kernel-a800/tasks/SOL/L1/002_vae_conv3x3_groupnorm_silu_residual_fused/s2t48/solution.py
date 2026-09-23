import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} must be divisible by {divisor}")


# Triton kernel: Conv3x3, NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum over ci and 3x3 neighborhood of x[n, ci, h+dh, w+dw] * w[co, ci, dh+1, dw+1]
# Grid: (B, C_OUT, H). Each program computes one output row for a given (n, co).
@triton.jit
def conv3x3_nchw_row_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, C_OUT, H: tl.constexpr, W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1)  # output channel
    pid_h = tl.program_id(2)   # output row index

    # Vector of W positions for this program
    w_offsets = tl.arange(0, BLOCK)
    mask = w_offsets < W

    # Accumulator for output row
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C):
        # Accumulate over 3x3 neighborhood with padding=1
        for dh in [-1, 0, 1]:
            h_in = pid_h + dh
            for dw in [-1, 0, 1]:
                w_in = w_offsets + dw
                # NCHW contiguous indexing
                nC = pid_n * C + ci
                offset_x = (nC * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + offset_x, mask=mask, other=0.0)
                # weight layout: (C_OUT, C, 3, 3) contiguous
                w_offset = (pid_co * C + ci) * 9 + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    # Store results to out[n, co, h, w_offsets]
    out_offset = (pid_n * C_OUT + pid_co) * (H * W) + pid_h * W + w_offsets
    tl.store(out_ptr + out_offset, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, num_groups: int = 32):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Validate shapes
        assert x.dim() == 4, "x must be NCHW"
        B, C, H, W = x.shape
        _assert_divisible(C, self.num_groups)
        C_PER_GROUP = C // self.num_groups

        # Ensure contiguous and float32 for Triton compute
        x = x.contiguous().to(torch.float32)

        # Prepare output tensors for convs
        out1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        out2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        # Convert weights to float32 and contiguous
        conv1_w = conv1_weight.contiguous().to(torch.float32)
        conv2_w = conv2_weight.contiguous().to(torch.float32)

        # Launch Triton conv1
        grid1 = (B, C, H)
        conv3x3_nchw_row_kernel[grid1](
            x, conv1_w, out1,
            B, C, C, H, W,
            BLOCK=64,  # vectorize along W; mask handles tails
        )

        # Launch Triton conv2
        grid2 = (B, C, H)
        conv3x3_nchw_row_kernel[grid2](
            out1, conv2_w, out2,
            B, C, C, H, W,
            BLOCK=64,
        )

        # Save residual
        residual = x  # we didn't alter x yet; if we wanted residual from conv, we'd store earlier. Here we just use input x as residual.
        # The original code adds original input x as residual. We do that now.
        out2 = out2 + residual

        return out2


def run(*args):
    return ModelNew()(*args)
