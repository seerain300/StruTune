import torch
import triton
import triton.language as tl


# Conv3x3: y = x * w, stride=1, padding=1, no bias
# Grid: (B, C_out, H, W) one program per output element
@triton.jit
def conv3x3_kernel_point(
    x_ptr,           # *f32, input [B, C_in, H, W]
    w_ptr,           # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,           # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)   # batch
    c_out = tl.program_id(1)  # output channel
    h_out = tl.program_id(2)  # output row
    w_out = tl.program_id(3)  # output col

    acc = tl.zeros((), dtype=tl.float32)

    # Accumulate over input channels and 3x3 neighborhood, with masks for padding
    for c_in in range(0, C_in):
        for dh in range(-1, 2):
            h_in = h_out + dh
            for dw in range(-1, 2):
                w_in = w_out + dw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # Input offset: (((n * C_in) + c_in) * H + h_in) * W + w_in
                x_off = (((n * C_in) + c_in) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                # Weight offset: (((c_out * C_in) + c_in) * 9) + (dh+1)*3 + (dw+1)
                w_off = (((c_out * C_in) + c_in) * 9) + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    # Output offset: (((n * C_out) + c_out) * H + h_out) * W + w_out
    y_off = (((n * C_out) + c_out) * H + h_out) * W + w_out
    tl.store(y_ptr + y_off, acc)


# Triton elementwise kernel: y = x * gamma + beta (affine), then SiLU: x * sigmoid(x)
# We will use this after PyTorch's group_norm to apply scale/bias and SiLU.
@triton.jit
def affine_silu_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)

    h = hw // W
    w = hw % W

    idx = (((n * C) + c) * H + h) * W + w
    x_val = tl.load(x_ptr + idx)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig * gamma + beta
    tl.store(y_ptr + idx, y_val)


# Elementwise residual add: out = a + b, both (B, C, H, W)
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
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

        # Cast to float32 for stable compute
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # First conv: out1 = conv3x3(x)
        C_out1 = conv1_w_f32.shape[0]
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        grid = (B, C_out1, H, W)
        conv3x3_kernel_point[grid](
            x_f32, conv1_w_f32, out1,
            B=B, C_in=C_in, C_out=C_out1,
            H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm for first block (PyTorch for robustness), then Triton affine + SiLU
        out1_gn = torch.nn.functional.group_norm(out1, self.num_groups, weight=None, bias=None, eps=self.eps)
        # But we need per-channel scale and bias; GroupNorm with provided weight/bias:
        out1_norm = torch.nn.functional.group_norm(out1, self.num_groups, norm1_weight_f32, norm1_bias_f32, eps=self.eps)

        # Apply SiLU + affine in Triton
        out1_affine = torch.empty_like(out1_norm)
        grid_aff = (B, C_out1, H * W)
        affine_silu_kernel[grid_aff](
            out1_norm, norm1_weight_f32, norm1_bias_f32, out1_affine,
            B=B, C=C_out1, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # Second conv: out2 = conv3x3(out1_affine)
        C_in2 = C_out1
        C_out2 = conv2_w_f32.shape[0]
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)

        grid2 = (B, C_in2, H, W)
        # We need to conv over out1_affine, which has shape (B, C_in2, H, W)
        conv3x3_kernel_point[grid2](
            out1_affine, conv2_w_f32, out2,
            B=B, C_in=C_in2, C_out=C_out2,
            H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm for second block (PyTorch), then Triton affine + SiLU
        out2_gn = torch.nn.functional.group_norm(out2, self.num_groups, weight=None, bias=None, eps=self.eps)
        out2_norm = torch.nn.functional.group_norm(out2, self.num_groups, norm2_weight_f32, norm2_bias_f32, eps=self.eps)

        grid_aff2 = (B, C_out2, H * W)
        affine_silu_kernel[grid_aff2](
            out2_norm, norm2_weight_f32, norm2_bias_f32, out2_affine,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # Final residual add: out2_affine + x
        out = torch.empty_like(out2_affine)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            out2_affine, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
