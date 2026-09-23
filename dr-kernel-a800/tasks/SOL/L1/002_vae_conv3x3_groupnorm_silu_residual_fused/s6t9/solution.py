import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, C_out, H, W, H_out, W_out,
    X_sN, X_sC, X_sH, X_sW,
    W_sCO, W_sCI, W_sKH, W_sKW,
    Y_sN, Y_sC, Y_sH, Y_sW,
    KH: tl.constexpr, KW: tl.constexpr
):
    # program ids
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        # kh, kw loop
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = pid_h * KH + kh
                iw = pid_w * KW + kw
                # mask for padding
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_off = pid_n * X_sN + ci * X_sC + ih * X_sH + iw * X_sW
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                x_f32 = x_val.to(tl.float32)

                # weight scalar
                w_off = pid_co * W_sCO + ci * W_sCI + kh * W_sKH + kw * W_sKW
                w_val = tl.load(w_ptr + w_off)
                w_f32 = w_val.to(tl.float32)

                acc += x_f32 * w_f32

    # store result
    y_off = pid_n * Y_sN + pid_co * Y_sC + pid_h * Y_sH + pid_w * Y_sW
    tl.store(y_ptr + y_off, acc)


@triton.jit
def groupnorm_reduce_sums(
    x_ptr, sums_ptr,
    N, C, H, W, num_groups,
    X_sN, X_sC, X_sH, X_sW,
    BLOCK_HW: tl.constexpr
):
    # grid over (N, num_groups)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    C_per_group = C // num_groups
    start_chan = pid_n * C_per_group + pid_g

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    for i in range(0, C_per_group):
        chan = start_chan + i
        num_hw = H * W
        s = tl.zeros((), dtype=tl.float32)
        ss = tl.zeros((), dtype=tl.float32)
        for off in range(0, num_hw, BLOCK_HW):
            hw_idx = off + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < num_hw
            h = (hw_idx // W).to(tl.int32)
            w = (hw_idx % W).to(tl.int32)
            x_off = pid_n * X_sN + chan * X_sC + h * X_sH + w * X_sW
            x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
            x_f32 = x_val.to(tl.float32)
            s += tl.sum(x_f32, axis=0)
            ss += tl.sum(x_f32 * x_f32, axis=0)
        sum_val += s
        sumsq_val += ss

    M = C_per_group * H * W
    base = pid_n * (num_groups * 2) + pid_g * 2  # indices [sum, sumsq]
    tl.store(sums_ptr + base + 0, sum_val)       # sum
    tl.store(sums_ptr + base + 1, sumsq_val)     # sumsq


@triton.jit
def groupnorm_apply_affine_silu(
    x_ptr, y_ptr, weight_ptr, bias_ptr, sums_ptr,
    N, C, H, W, num_groups, eps,
    X_sN, X_sC, X_sH, X_sW,
    Y_sN, Y_sC, Y_sH, Y_sW
):
    # grid over (N*C,)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    C_per_group = C // num_groups
    g = c // C_per_group

    # per-(n, g) stats
    base = n * (num_groups * 2) + g * 2
    sum_val = tl.load(sums_ptr + base + 0)
    sumsq_val = tl.load(sums_ptr + base + 1)
    M = C_per_group * H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    num_hw = H * W
    for off in range(0, num_hw):
        h = off // W
        w = off % W
        x_off = n * X_sN + c * X_sC + h * X_sH + w * X_sW
        x_val = tl.load(x_ptr + x_off)
        x_f32 = x_val.to(tl.float32)

        # normalize
        y_norm = (x_f32 - mean) * rstd

        # affine
        w_c = tl.load(weight_ptr + c)
        b_c = tl.load(bias_ptr + c)
        w_f32 = w_c.to(tl.float32)
        b_f32 = b_c.to(tl.float32)
        y_affine = y_norm * w_f32 + b_f32

        # SiLU
        sig = 1.0 / (1.0 + tl.exp(-y_affine))
        y_silu = y_affine * sig

        # store
        y_off = n * Y_sN + c * Y_sC + h * Y_sH + w * Y_sW
        tl.store(y_ptr + y_off, y_silu)


@triton.jit
def add_inplace(y_ptr, x_ptr, out_ptr, total):
    # simple elementwise add over total elements
    pid = tl.program_id(0)
    if pid < total:
        a = tl.load(y_ptr + pid)
        b = tl.load(x_ptr + pid)
        tl.store(out_ptr + pid, a + b)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Triton-only implementation:
        - Performs two 3x3 convolutions in Triton (stride=1, padding=1).
        - Applies GroupNorm(num_groups=32) and SiLU in Triton.
        - Adds residual x at the end in Triton.
        """

        # Allocate outputs for convs and final output
        N, C, H, W = x.shape
        # Ensure channel divisible by num_groups (original code assumes this)
        assert C % self.num_groups == 0, "C must be divisible by num_groups (32)"

        # 1) First conv: y1 = conv3x3(x)
        y1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)

        # Launch Triton conv1
        # Grid: (N, C, H, W)
        grid_conv1 = (N, C, H, W)
        conv3x3_kernel[grid_conv1](
            x, conv1_weight, y1,
            N, C, C, H, W, H, W,
            x.stride()[0], x.stride()[1], x.stride()[2], x.stride()[3],
            conv1_weight.stride()[0], conv1_weight.stride()[1], conv1_weight.stride()[2], conv1_weight.stride()[3],
            y1.stride()[0], y1.stride()[1], y1.stride()[2], y1.stride()[3],
            KH=3, KW=3,
            num_warps=1
        )

        # 2) GroupNorm + SiLU for y1: y2
        # Compute per-(N,group) stats
        sums1 = torch.empty(N * self.num_groups * 2, device=x.device, dtype=torch.float32)
        groupnorm_reduce_sums[(N, self.num_groups)](
            y1, sums1,
            N, C, H, W, self.num_groups,
            y1.stride()[0], y1.stride()[1], y1.stride()[2], y1.stride()[3],
            BLOCK_HW=1024,
            num_warps=1
        )
        # Apply GroupNorm + affine + SiLU
        y2 = torch.empty_like(y1)
        groupnorm_apply_affine_silu[(N * C)](
            y1, y2, norm1_weight, norm1_bias, sums1,
            N, C, H, W, self.num_groups, self.eps,
            y1.stride()[0], y1.stride()[1], y1.stride()[2], y1.stride()[3],
            y2.stride()[0], y2.stride()[1], y2.stride()[2], y2.stride()[3],
            num_warps=1
        )

        # 3) Second conv: y3 = conv3x3(y2)
        y3 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (N, C, H, W)
        conv3x3_kernel[grid_conv2](
            y2, conv2_weight, y3,
            N, C, C, H, W, H, W,
            y2.stride()[0], y2.stride()[1], y2.stride()[2], y2.stride()[3],
            conv2_weight.stride()[0], conv2_weight.stride()[1], conv2_weight.stride()[2], conv2_weight.stride()[3],
            y3.stride()[0], y3.stride()[1], y3.stride()[2], y3.stride()[3],
            KH=3, KW=3,
            num_warps=1
        )

        # 4) GroupNorm + SiLU for y3: y4
        sums2 = torch.empty(N * self.num_groups * 2, device=x.device, dtype=torch.float32)
        groupnorm_reduce_sums[(N, self.num_groups)](
            y3, sums2,
            N, C, H, W, self.num_groups,
            y3.stride()[0], y3.stride()[1], y3.stride()[2], y3.stride()[3],
            BLOCK_HW=1024,
            num_warps=1
        )
        y4 = torch.empty_like(y3)
        groupnorm_apply_affine_silu[(N * C)](
            y3, y4, norm2_weight, norm2_bias, sums2,
            N, C, H, W, self.num_groups, self.eps,
            y3.stride()[0], y3.stride()[1], y3.stride()[2], y3.stride()[3],
            y4.stride()[0], y4.stride()[1], y4.stride()[2], y4.stride()[3],
            num_warps=1
        )

        # 5) Residual addition: out = y4 + x
        total = N * C * H * W
        out = torch.empty_like(y4)
        add_inplace[(total,)](
            y4, x, out,
            total,
            num_warps=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
