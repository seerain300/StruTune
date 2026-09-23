import torch
import triton
import triton.language as tl


# Triton kernel: single-output-per-program 3x3 conv (NCHW, stride=1, padding=1, no bias)
# Assumptions:
# - x is (B, C_in, H, W) float32 contiguous
# - w is (C_out, C_in, 3, 3) float32 contiguous
# - y is (B, C_out, H, W) float32
@triton.jit
def conv3x3_nchw_fp32(x_ptr, w_ptr, y_ptr,
                      B, C_in, H, W, C_out,
                      BLOCK_IN: tl.constexpr):
    pid = tl.program_id(0)  # one program per output element
    total = B * C_out * H * W
    # Map pid -> (n, c_out, h_out, w_out)
    tmp = pid
    w_out = tmp % W
    tmp = tmp // W
    h_out = tmp % H
    tmp = tmp // H
    c_out = tmp % C_out
    n = tmp // C_out

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks and 3x3 neighborhood with masks for padding
    for in_c_start in range(0, C_in, BLOCK_IN):
        for kh in range(0, 3):
            for kw in range(0, 3):
                h_in = h_out + kh - 1  # padding=1
                w_in = w_out + kw - 1
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                for ic in range(0, BLOCK_IN):
                    c_in = in_c_start + ic
                    c_in_valid = c_in < C_in
                    # Compute input offset: ((n * C_in + c_in) * H + h_in) * W + w_in
                    x_offset = (((n * C_in + c_in) * H + h_in) * W + w_in)
                    x_val = tl.load(x_ptr + x_offset, mask=in_bounds & c_in_valid, other=0.0)
                    # Weight offset: ((c_out * C_in + c_in) * 3 + kh) * 3 + kw
                    w_offset = (((c_out * C_in + c_in) * 3 + kh) * 3 + kw)
                    w_val = tl.load(w_ptr + w_offset, mask=c_in_valid, other=0.0)
                    acc += x_val * w_val

    # Store output
    y_offset = (((n * C_out + c_out) * H + h_out) * W + w_out)
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: elementwise SiLU over flattened tensor
@triton.jit
def silu_kernel(x_ptr, y_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    for i in range(0, BLOCK):
        idx = start + i
        valid = idx < total
        x = tl.load(x_ptr + idx, mask=valid, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(y_ptr + idx, y, mask=valid)


# Triton kernel: elementwise add (add original input to final output)
@triton.jit
def add_residual_kernel(x_ptr, y_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    for i in range(0, BLOCK):
        idx = start + i
        valid = idx < total
        a = tl.load(x_ptr + idx, mask=valid, other=0.0)
        b = tl.load(y_ptr + idx, mask=valid, other=0.0)
        c = a + b
        tl.store(out_ptr + idx, c, mask=valid)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all computation is via Triton kernels

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Enforce NCHW contiguous and float32 for Triton kernels
        device = x.device
        dtype = torch.float32

        x_fp32 = x.contiguous().to(dtype)  # (B, C, H, W)
        B, C_in, H, W = x_fp32.shape

        # First conv: y1 = conv(x, conv1_weight)
        C_out = conv1_weight.shape[0]  # output channels of first conv
        y1 = torch.empty((B, C_out, H, W), device=device, dtype=dtype)

        total_out = B * C_out * H * W
        grid_conv = (total_out,)
        conv3x3_nchw_fp32[grid_conv](
            x_fp32, conv1_weight.contiguous().to(dtype), y1,
            B, C_in, H, W, C_out,
            BLOCK_IN=32,
            num_warps=4
        )

        # SiLU on y1
        silu_out1 = torch.empty_like(y1, device=device, dtype=dtype)
        total1 = y1.numel()
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_kernel[grid_silu1](y1, silu_out1, total1, BLOCK=1024)

        # Second conv: y2 = conv(silu_out1, conv2_weight)
        C_in2 = silu_out1.shape[1]  # channels after first conv and SiLU -> C_out
        C_out2 = conv2_weight.shape[0]  # output channels of second conv
        y2 = torch.empty((B, C_out2, H, W), device=device, dtype=dtype)

        total2 = B * C_out2 * H * W
        grid_conv2 = (total2,)
        conv3x3_nchw_fp32[grid_conv2](
            silu_out1, conv2_weight.contiguous().to(dtype), y2,
            B, C_in2, H, W, C_out2,
            BLOCK_IN=32,
            num_warps=4
        )

        # SiLU on y2
        silu_out2 = torch.empty_like(y2, device=device, dtype=dtype)
        total2_silu = y2.numel()
        grid_silu2 = (triton.cdiv(total2_silu, 1024),)
        silu_kernel[grid_silu2](y2, silu_out2, total2_silu, BLOCK=1024)

        # Residual addition: add original input x (cast to fp32) to final output (elementwise)
        x_flat = x_fp32.view(-1)  # (B*C*H*W,)
        silu_out2_flat = silu_out2.view(-1)
        total_add = x_flat.numel()
        out = torch.empty(total_add, device=device, dtype=dtype)
        grid_add = (triton.cdiv(total_add, 1024),)
        add_residual_kernel[grid_add](x_flat, silu_out2_flat, out, total_add, BLOCK=1024)

        final_out = out.view(B, C_in, H, W)
        return final_out


def run(*args):
    return ModelNew()(*args)
