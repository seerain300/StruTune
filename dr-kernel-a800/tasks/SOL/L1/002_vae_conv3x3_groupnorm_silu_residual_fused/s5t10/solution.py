import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# Triton conv3x3: one program per output pixel (n, c_out, h_out, w_out)
# stride=1, padding=1, no bias
@triton.jit
def conv3x3_pixel_kernel(
    x_ptr,            # *f32, input [B, C_in, H, W]
    w_ptr,            # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,            # *f32, output [B, C_out, H, W]
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    # grid = (B, C_out, H*W)
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    hw = tl.program_id(2)
    h_out = hw // W
    w_out = hw % W

    acc = tl.zeros((), dtype=tl.float32)

    # Reduce over input channels and 3x3 neighborhood (padding=1 => indices always in-bounds)
    for c_in in range(C_in):
        for dh in (-1, 0, 1):
            for dw in (-1, 0, 1):
                hi = h_out + dh
                wi = w_out + dw
                # x index: (((n * C_in) + c_in) * H + hi) * W + wi
                x_idx = (((n * C_in) + c_in) * H + hi) * W + wi
                x_val = tl.load(x_ptr + x_idx)
                # weight index: w_ptr[c_out, c_in, 3+dh, 3+dw]
                # weight layout: (C_out, C_in, 3, 3) contiguous => linear index
                w_idx = (c_out * (C_in * 9)) + (c_in * 9) + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    # Store output y[n, c_out, h_out, w_out]
    y_idx = (((n * C_out) + c_out) * H + h_out) * W + w_out
    tl.store(y_ptr + y_idx, acc)


# Triton elementwise residual add: out = a + b
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    # grid = (B, C, H*W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)
    h = hw // W
    w = hw % W

    idx = (((n * C) + c) * H + h) * W + w
    a_val = tl.load(a_ptr + idx)
    b_val = tl.load(b_ptr + idx)
    tl.store(out_ptr + idx, a_val + b_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"

        B, C_in, H, W = x.shape

        # Ensure contiguous and float32 for Triton
        x_f32 = x.contiguous().to(torch.float32)

        # Prepare weights as float32 contiguous
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # Assert channels are divisible by num_groups for GroupNorm
        C_out1 = conv1_w_f32.shape[0]
        C_out2 = conv2_w_f32.shape[0]
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups"
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups"

        # First conv
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)
        grid1 = (B, C_out1, H * W)
        conv3x3_pixel_kernel[grid1](
            x_f32, conv1_w_f32, out1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + SiLU for first block
        out1 = F.group_norm(out1, self.num_groups, weight=norm1_weight_f32, bias=norm1_bias_f32, eps=self.eps)
        out1 = F.silu(out1)

        # Second conv
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        grid2 = (B, C_out2, H * W)
        conv3x3_pixel_kernel[grid2](
            out1, conv2_w_f32, out2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + SiLU for second block
        out2 = F.group_norm(out2, self.num_groups, weight=norm2_weight_f32, bias=norm2_bias_f32, eps=self.eps)
        out2 = F.silu(out2)

        # Residual add (Triton elementwise)
        out = torch.empty_like(out2)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            out2, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # Cast back to original dtype if needed
        if x.dtype != torch.float32:
            out = out.to(x.dtype)

        return out


def run(*args):
    return ModelNew()(*args)
