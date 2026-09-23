import torch
import triton
import triton.language as tl


@triton.jit
def groupnorm_reduce_sums(
    x_ptr, sums_ptr,
    N, C, H, W, num_groups,
    X_sN, X_sC, X_sH, X_sW,
    BLOCK_HW: tl.constexpr,
):
    # Grid: (N, num_groups)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    C_per_group = C // num_groups
    start_chan = pid_n * C_per_group + pid_g

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    num_hw = H * W
    for i in range(0, C_per_group):
        chan = start_chan + i
        for off in range(0, num_hw, BLOCK_HW):
            hw_off = off + tl.arange(0, BLOCK_HW)
            mask_hw = hw_off < num_hw
            h = hw_off // W
            w = hw_off % W
            x_off = pid_n * X_sN + chan * X_sC + h * X_sH + w * X_sW
            x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
            x_f32 = x_val.to(tl.float32)
            sum_val += tl.sum(x_f32, axis=0)
            sumsq_val += tl.sum(x_f32 * x_f32, axis=0)

    M = C_per_group * H * W
    base = pid_n * (num_groups * 2) + pid_g * 2  # store [sum, sumsq] at indices base and base+1
    tl.store(sums_ptr + base + 0, sum_val)
    tl.store(sums_ptr + base + 1, sumsq_val)


@triton.jit
def groupnorm_apply_affine_silu_nc(
    x_ptr, y_ptr, weight_ptr, bias_ptr, sums_ptr,
    N, C, H, W, num_groups, eps,
    X_sN, X_sC, X_sH, X_sW,
    Y_sN, Y_sC, Y_sH, Y_sW,
):
    # Grid: (N*C,)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    C_per_group = C // num_groups
    g = c // C_per_group

    base = n * (num_groups * 2) + g * 2
    sum_val = tl.load(sums_ptr + base + 0)
    sumsq_val = tl.load(sums_ptr + base + 1)

    M = C_per_group * H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Iterate over H*W positions for this (n, c)
    num_hw = H * W
    for off in range(0, num_hw):
        h = off // W
        w = off % W
        x_val = tl.load(x_ptr + n * X_sN + c * X_sC + h * X_sH + w * X_sW)
        x_f32 = x_val.to(tl.float32)
        norm = (x_f32 - mean) * rstd
        w_c = tl.load(weight_ptr + c)
        b_c = tl.load(bias_ptr + c)
        y_affine = norm * w_c + b_c

        # SiLU
        sig = 1.0 / (1.0 + tl.exp(-y_affine))
        y = y_affine * sig

        y_off = n * Y_sN + c * Y_sC + h * Y_sH + w * Y_sW
        tl.store(y_ptr + y_off, y)


@triton.jit
def add_inplace(x1_ptr, x2_ptr, out_ptr, N, C, H, W):
    # Grid: (N*C,)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    num_hw = H * W
    for off in range(0, num_hw):
        h = off // W
        w = off % W
        a = tl.load(x1_ptr + n * C * H * W + c * H * W + h * W + w)
        b = tl.load(x2_ptr + n * C * H * W + c * H * W + h * W + w)
        tl.store(out_ptr + n * C * H * W + c * H * W + h * W + w, a + b)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32):
        super().__init__()
        self.num_groups = num_groups
        if self.num_groups == 0:
            raise ValueError("num_groups must be > 0")

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        """
        Triton-based GroupNorm + SiLU on top of PyTorch convs.
        Keeps convs in PyTorch (cuDNN) for robustness; applies GroupNorm and SiLU via Triton.
        Final residual add is Triton elementwise.
        """
        N, C, H, W = x.shape

        # First conv: y1 = conv3x3(x)
        y1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Buffers for GroupNorm stats
        sums1 = torch.empty(N * self.num_groups * 2, device=x.device, dtype=torch.float32)

        # GroupNorm and SiLU for y1: y2
        groupnorm_reduce_sums[(N, self.num_groups)](
            y1, sums1,
            N, y1.shape[1], H, W, self.num_groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_HW=1024,
            num_warps=1
        )
        y2 = torch.empty_like(y1)
        groupnorm_apply_affine_silu_nc[(N * y1.shape[1])](
            y1, y2, norm1_weight, norm1_bias, sums1,
            N, y1.shape[1], H, W, self.num_groups, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )

        # Second conv: y3 = conv3x3(y2)
        y3 = torch.nn.functional.conv2d(y2, conv2_weight, bias=None, stride=1, padding=1)

        # Buffers for GroupNorm stats (second)
        sums2 = torch.empty(N * self.num_groups * 2, device=x.device, dtype=torch.float32)

        # GroupNorm and SiLU for y3: y4
        groupnorm_reduce_sums[(N, self.num_groups)](
            y3, sums2,
            N, y3.shape[1], H, W, self.num_groups,
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_HW=1024,
            num_warps=1
        )
        y4 = torch.empty_like(y3)
        groupnorm_apply_affine_silu_nc[(N * y3.shape[1])](
            y3, y4, norm2_weight, norm2_bias, sums2,
            N, y3.shape[1], H, W, self.num_groups, eps,
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            y4.stride(0), y4.stride(1), y4.stride(2), y4.stride(3),
            num_warps=1
        )

        # Final residual add: out = y4 + x
        out = torch.empty_like(y4)
        add_inplace[(N * y4.shape[1])](
            y4, x, out,
            N, y4.shape[1], y4.shape[2], y4.shape[3],
            num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
