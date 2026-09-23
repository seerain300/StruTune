import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid: (B, C_out, H_out, W_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = 0.0
    # Loop over input channels and 3x3 neighborhood
    # Note: use vectorized loads with masks to emulate padding
    for ci in range(0, C_in):
        for dh in range(0, 3):
            for dw in range(0, 3):
                h_in = pid_h + dh - 1
                w_in = pid_w + dw - 1
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                x_offset = pid_n * x_stride_n + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def groupnorm_triton_channels(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                               B, C, H_out, W_out,
                               y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                               y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                               num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid: (B, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    N = H_out * W_out  # spatial elements per channel
    # Vector of spatial indices (linearized)
    idx = tl.arange(0, num_warps)
    total = 0.0
    total_sq = 0.0
    for h in range(0, H_out):
        for w in range(0, W_out):
            lin = h * W_out + w
            ptr = pid_n * y_in_stride_n + pid_c * y_in_stride_c + lin
            x = tl.load(y_in_ptr + ptr)
            total += x
            total_sq += x * x

    mean = total / N
    var = total_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(weight_ptr + pid_c)
    beta = tl.load(bias_ptr + pid_c)
    for h in range(0, H_out):
        for w in range(0, W_out):
            lin = h * W_out + w
            ptr_in = pid_n * y_in_stride_n + pid_c * y_in_stride_c + lin
            x = tl.load(y_in_ptr + ptr_in)
            norm = (x - mean) * rstd
            y = norm * gamma + beta
            ptr_out = pid_n * y_out_stride_n + pid_c * y_out_stride_c + lin
            tl.store(y_out_ptr + ptr_out, y)


@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                 num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + idx, y)


@triton.jit
def add_residual_triton(x_ptr, y_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    a = tl.load(x_ptr + idx, mask=mask, other=0.0)
    b = tl.load(y_ptr + idx, mask=mask, other=0.0)
    c = a + b
    tl.store(y_ptr + idx, c)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float = 1e-5):
        # Ensure inputs are contiguous and float32
        x = x.contiguous().to(torch.float32)
        B, C, H, W = x.shape
        C_in = C
        C_out = C  # conv weights are (C, C, 3, 3), so output channels = input channels

        # Prepare tensors
        device = x.device

        # First path: conv1
        H1 = H - 2  # padding=1, 3x3 kernel
        W1 = W - 2
        y1 = torch.empty((B, C_out, H1, W1), device=device, dtype=torch.float32)
        # Strides
        x_stride_n, x_stride_c, x_stride_h, x_stride_w = x.stride()
        y1_stride_n, y1_stride_c, y1_stride_h, y1_stride_w = y1.stride()
        w1_stride_co, w1_stride_ci, w1_stride_dh, w1_stride_dw = conv1_weight.stride()
        grid1 = (B, C_out, H1, W1)
        conv3x3_nchw_nobias[grid1](
            x, conv1_weight, y1,
            B, C_in, C_out, H, W, H1, W1,
            x_stride_n, x_stride_c, x_stride_h, x_stride_w,
            w1_stride_co, w1_stride_ci, w1_stride_dh, w1_stride_dw,
            y1_stride_n, y1_stride_c, y1_stride_h, y1_stride_w,
            num_warps=4, num_stages=2,
        )

        # GroupNorm1: per-channel stats across spatial H1*W1
        y2_before_gn1 = y1  # alias, no new allocation
        y_after_gn1 = torch.empty_like(y1)
        y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w = y2_before_gn1.stride()
        y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w = y_after_gn1.stride()
        grid_gn1 = (B, C)
        groupnorm_triton_channels[grid_gn1](
            y2_before_gn1, norm1_weight, norm1_bias, y_after_gn1,
            B, C, H1, W1,
            y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
            y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
            num_warps=4, num_stages=2,
        )

        # SiLU1
        y_silu1 = torch.empty_like(y_after_gn1)
        N1 = y_silu1.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_triton[grid_silu1](
            y_after_gn1, y_silu1, N1,
            num_warps=4, num_stages=2,
        )

        # conv2
        H2 = H - 4
        W2 = W - 4
        y2 = torch.empty((B, C_out, H2, W2), device=device, dtype=torch.float32)
        x2_stride_n, x2_stride_c, x2_stride_h, x2_stride_w = y_silu1.stride()
        y2_stride_n, y2_stride_c, y2_stride_h, y2_stride_w = y2.stride()
        w2_stride_co, w2_stride_ci, w2_stride_dh, w2_stride_dw = conv2_weight.stride()
        grid2 = (B, C_out, H2, W2)
        conv3x3_nchw_nobias[grid2](
            y_silu1, conv2_weight, y2,
            B, C, C_out, H1, W1, H2, W2,  # note: using H1, W1 for input to conv2
            x2_stride_n, x2_stride_c, x2_stride_h, x2_stride_w,
            w2_stride_co, w2_stride_ci, w2_stride_dh, w2_stride_dw,
            y2_stride_n, y2_stride_c, y2_stride_h, y2_stride_w,
            num_warps=4, num_stages=2,
        )

        # GroupNorm2: per-channel stats across spatial H2*W2
        y2_before_gn2 = y2
        y_after_gn2 = torch.empty_like(y2)
        y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w = y2_before_gn2.stride()
        y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w = y_after_gn2.stride()
        grid_gn2 = (B, C)
        groupnorm_triton_channels[grid_gn2](
            y2_before_gn2, norm2_weight, norm2_bias, y_after_gn2,
            B, C, H2, W2,
            y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
            y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
            num_warps=4, num_stages=2,
        )

        # SiLU2
        y_silu2 = torch.empty_like(y_after_gn2)
        N2 = y_silu2.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_triton[grid_silu2](
            y_after_gn2, y_silu2, N2,
            num_warps=4, num_stages=2,
        )

        # Residual addition: y = y_silu2 + x
        # x shape (B, C, H, W), y_silu2 shape (B, C, H-4, W-4); residual addition applies to the final output position
        # Since H_out2 depends on input H, we add the original x to the final output at the appropriate spatial offset.
        # However, ModelNew should return final y_silu2; if residual is intended to be added, it should be added in-place to y_silu2.
        # The original PyTorch implementation adds residual = x to the final output. Here, we launch Triton kernel to add.
        out = torch.empty_like(y_silu2)
        N_out = out.numel()
        grid_add = (triton.cdiv(N_out, 1024),)
        add_residual_triton[grid_add](
            x, y_silu2, N_out,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
