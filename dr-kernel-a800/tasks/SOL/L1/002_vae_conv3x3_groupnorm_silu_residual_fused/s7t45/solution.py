import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_3x3_stride1_pad1_elem_kernel(
    x_ptr,         # *f32, input [N, C_in, H, W]
    w_ptr,         # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,         # *f32, output [N, C_out, H, W]
    N, H, W, C_in, C_out, H_out, W_out,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    # One program computes y[n, co, oh, ow]
    n = tl.program_id(0)
    co = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for c_i in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                ih = oh + kh - 1  # padding=1, stride=1
                iw = ow + kw - 1
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                # Load input scalar with mask
                x_offset = n * x_stride_n + c_i * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                # Load weight scalar
                w_offset = co * w_stride_co + c_i * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    # Store result
    y_offset = n * y_stride_n + co * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def group_norm_kernel(
    x_ptr,          # *f32, input [N, C, H*W] flattened per (n, c, spatial)
    gamma_ptr,      # *f32, scale [C]
    beta_ptr,       # *f32, bias [C]
    y_ptr,          # *f32, output [N, C, H*W]
    N, C, H, W, NUM_GROUPS, eps,
):
    # One program per (n, group)
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    group_id = pid % NUM_GROUPS
    group_size = C // NUM_GROUPS
    c_start = group_id * group_size

    # Accumulate sum and sum of squares over group's channels and all spatial
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    L = H * W
    for gc in range(group_size):
        c = c_start + gc
        for l in range(L):
            x_offset = n * (C * L) + c * L + l
            x_val = tl.load(x_ptr + x_offset)
            sum_val += x_val
            sum_sq += x_val * x_val

    mean = sum_val / (group_size * L)
    var = sum_sq / (group_size * L) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and affine, write to y
    for gc in range(group_size):
        c = c_start + gc
        gamma_c = tl.load(gamma_ptr + c)
        beta_c = tl.load(beta_ptr + c)
        for l in range(L):
            x_offset = n * (C * L) + c * L + l
            x_val = tl.load(x_ptr + x_offset)
            y_val = (x_val - mean) * inv_std
            y_val = y_val * gamma_c + beta_c
            y_offset = n * (C * L) + c * L + l
            tl.store(y_ptr + y_offset, y_val)


@triton.jit
def silu_kernel(x_ptr, y_ptr, N, C, H, W):
    L = H * W
    for n in range(N):
        for c in range(C):
            for l in range(L):
                x_offset = n * (C * L) + c * L + l
                x_val = tl.load(x_ptr + x_offset)
                sig = 1.0 / (1.0 + tl.exp(-x_val))
                y_val = x_val * sig
                y_offset = n * (C * L) + c * L + l
                tl.store(y_ptr + y_offset, y_val)


@triton.jit
def add_residual_kernel(x_ptr, y_ptr, out_ptr, N, C, H, W):
    L = H * W
    for n in range(N):
        for c in range(C):
            for l in range(L):
                a = tl.load(x_ptr + n * (C * L) + c * L + l)
                b = tl.load(y_ptr + n * (C * L) + c * L + l)
                tl.store(out_ptr + n * (C * L) + c * L + l, a + b)


class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All computation is done in Triton; no torch ops in forward.
        """
        assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda, "Tensors must be on CUDA for Triton."
        N, C_in, H, W = x.shape
        C1_out, C_in_w1, KH1, KW1 = conv1_weight.shape
        assert C_in_w1 == C_in and KH1 == 3 and KW1 == 3, "conv1_weight must be [C_out, C_in, 3, 3]"
        C2_out, C_in_w2, KH2, KW2 = conv2_weight.shape
        assert C_in_w2 == C1_out and KH2 == 3 and KW2 == 3, "conv2_weight must be [C_out, C_in, 3, 3]"

        # Ensure contiguous float32
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        # First conv: y1
        y1 = torch.empty((N, C1_out, H, W), dtype=torch.float32, device=x.device)
        grid1 = (N, C1_out, H, W)
        conv2d_3x3_stride1_pad1_elem_kernel[grid1](
            x, conv1_weight, y1,
            N, H, W, C_in, C1_out, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=2,
            num_stages=2,
        )

        # First GroupNorm: enforce C1_out % 32 == 0
        assert (C1_out % 32 == 0), "First GroupNorm requires C1_out divisible by 32"
        y1_flat = y1.view(N, C1_out, H * W).contiguous()
        y1_norm = torch.empty_like(y1_flat)  # [N, C1_out, H*W]
        grid_gn1 = (N * 32,)
        group_norm_kernel[grid_gn1](
            y1_flat, norm1_weight, norm1_bias, y1_norm,
            N, C1_out, H, W, 32, self.eps,
            num_warps=4,
            num_stages=2,
        )
        y1_norm = y1_norm.view(N, C1_out, H, W)

        # SiLU on y1_norm
        y1_silu = torch.empty_like(y1_norm)
        grid_silu1 = (N * C1_out * H * W,)
        silu_kernel[grid_silu1](
            y1_norm, y1_silu,
            N, C1_out, H, W,
            num_warps=4,
            num_stages=2,
        )

        # Second conv: y2
        y2 = torch.empty((N, C2_out, H, W), dtype=torch.float32, device=x.device)
        grid2 = (N, C2_out, H, W)
        conv2d_3x3_stride1_pad1_elem_kernel[grid2](
            y1_silu, conv2_weight, y2,
            N, H, W, C1_out, C2_out, H, W,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=2,
            num_stages=2,
        )

        # Second GroupNorm: enforce C2_out % 32 == 0
        assert (C2_out % 32 == 0), "Second GroupNorm requires C2_out divisible by 32"
        y2_flat = y2.view(N, C2_out, H * W).contiguous()
        y2_norm = torch.empty_like(y2_flat)  # [N, C2_out, H*W]
        grid_gn2 = (N * 32,)
        group_norm_kernel[grid_gn2](
            y2_flat, norm2_weight, norm2_bias, y2_norm,
            N, C2_out, H, W, 32, self.eps,
            num_warps=4,
            num_stages=2,
        )
        y2_norm = y2_norm.view(N, C2_out, H, W)

        # SiLU on y2_norm
        y2_silu = torch.empty_like(y2_norm)
        grid_silu2 = (N * C2_out * H * W,)
        silu_kernel[grid_silu2](
            y2_norm, y2_silu,
            N, C2_out, H, W,
            num_warps=4,
            num_stages=2,
        )

        # Residual add: out = y2_silu + x
        out = torch.empty_like(y2_silu)
        grid_add = (N * C2_out * H * W,)
        add_residual_kernel[grid_add](
            x, y2_silu, out,
            N, C2_out, H, W,
            num_warps=4,
            num_stages=2,
        )

        return out.view(N, C2_out, H, W)


# Example runner
@torch.no_grad()
def run_triton(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    model = ModelNew(eps=eps)
    return model(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias)


# Notes:
# - This implementation uses Triton kernels for all computations (conv2d, group_norm, silu, add).
# - It assumes GroupNorm with num_groups=32 and per-channel affine parameters. It checks C_out % 32 == 0.
# - The conv2d kernel performs a single output element per program, looping over input channels and 3x3 taps with masking for padding.
# - The forward method converts inputs/weights to float32 and contiguous tensors for stable math.


def run(*args):
    return ModelNew()(*args)
