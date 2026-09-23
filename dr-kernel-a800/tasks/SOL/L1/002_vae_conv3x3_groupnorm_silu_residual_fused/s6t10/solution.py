import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_str1_pad1_nchw(
    X_ptr,       # *float32, input tensor (N, C_in, H, W)
    W_ptr,       # *float32, weights tensor (C_out, C_in, 3, 3)
    Y_ptr,       # *float32, output tensor (N, C_out, H, W)
    N, C_out, C_in, H, W,
    X_sN, X_sC, X_sH, X_sW,
    W_sCo, W_sCi, W_sKh, W_sKw,
    Y_sN, Y_sC, Y_sH, Y_sW,
    num_warps: tl.constexpr
):
    # Grid: (N, C_out, H, W)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = 0.0

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                h_in = pid_h + kh - 1
                w_in = pid_w + kw - 1
                h_in_ok = (h_in >= 0) & (h_in < H)
                w_in_ok = (w_in >= 0) & (w_in < W)
                if h_in_ok and w_in_ok:
                    x_off = pid_n * X_sN + ci * X_sC + h_in * X_sH + w_in * X_sW
                    x_val = tl.load(X_ptr + x_off)
                else:
                    x_val = 0.0

                w_off = pid_co * W_sCo + ci * W_sCi + kh * W_sKh + kw * W_sKw
                w_val = tl.load(W_ptr + w_off)

                acc += x_val * w_val

    y_off = pid_n * Y_sN + pid_co * Y_sC + pid_h * Y_sH + pid_w * Y_sW
    tl.store(Y_ptr + y_off, acc)


@triton.jit
def groupnorm_reduce_sums(
    X_ptr,        # *float32, input tensor (N, C, H, W)
    sums_ptr,     # *float32, buffer (N, num_groups, 2) where [n, g, 0]=sum, [n, g, 1]=sumsq
    N, C, H, W, num_groups,
    X_sN, X_sC, X_sH, X_sW,
    sums_sN, sums_sG, sums_sT,  # sums_sT is the last dimension (2)
    num_warps: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # group_size = C // num_groups (since C % num_groups == 0 in this model)
    group_size = C // num_groups
    c_start = pid_g * group_size
    c_end = c_start + group_size

    sum_val = 0.0
    sumsq_val = 0.0

    for c in range(c_start, c_end):
        for h in range(0, H):
            for w in range(0, W):
                off = pid_n * X_sN + c * X_sC + h * X_sH + w * X_sW
                x = tl.load(X_ptr + off)
                sum_val += x
                sumsq_val += x * x

    base = pid_n * sums_sN + pid_g * sums_sG
    tl.store(sums_ptr + base + 0 * sums_sT, sum_val)
    tl.store(sums_ptr + base + 1 * sums_sT, sumsq_val)


@triton.jit
def groupnorm_apply_affine_silu(
    X_ptr,        # *float32, input tensor (N, C, H, W)
    W_ptr,        # *float32, scale tensor (C,)
    B_ptr,        # *float32, bias tensor (C,)
    Y_ptr,        # *float32, output tensor (N, C, H, W)
    sums_ptr,     # *float32, buffer (N, num_groups, 2)
    N, C, H, W, num_groups,
    X_sN, X_sC, X_sH, X_sW,
    Y_sN, Y_sC, Y_sH, Y_sW,
    W_sC, B_sC,
    sums_sN, sums_sG, sums_sT,
    eps,
    num_warps: tl.constexpr
):
    pid_nc = tl.program_id(0)  # iterate over N*C
    n = pid_nc // C
    c = pid_nc % C

    group_size = C // num_groups
    g = c // group_size

    base = n * sums_sN + g * sums_sG
    sum_val = tl.load(sums_ptr + base + 0 * sums_sT)
    sumsq_val = tl.load(sums_ptr + base + 1 * sums_sT)

    m = H * W * group_size
    mean = sum_val / m
    var = sumsq_val / m - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for h in range(0, H):
        for w in range(0, W):
            off_x = n * X_sN + c * X_sC + h * X_sH + w * X_sW
            x = tl.load(X_ptr + off_x)
            norm = (x - mean) * rstd
            scale = tl.load(W_ptr + c * W_sC)
            bias = tl.load(B_ptr + c * B_sC)
            y = norm * scale + bias
            # SiLU: y * sigmoid(y)
            sig = 1.0 / (1.0 + tl.exp(-y))
            y = y * sig

            off_y = n * Y_sN + c * Y_sC + h * Y_sH + w * Y_sW
            tl.store(Y_ptr + off_y, y)


@triton.jit
def add_residual_kernel(
    out_ptr,      # *float32, destination tensor (N, C, H, W)
    x_ptr,        # *float32, input tensor (N, C, H, W) to add
    N, C, H, W,
    out_sN, out_sC, out_sH, out_sW,
    x_sN, x_sC, x_sH, x_sW,
    num_warps: tl.constexpr
):
    total = N * C * H * W
    pid = tl.program_id(0)
    if pid < total:
        n = pid // (C * H * W)
        rem = pid % (C * H * W)
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W

        off_out = n * out_sN + c * out_sC + h * out_sH + w * out_sW
        off_x = n * x_sN + c * x_sC + h * x_sH + w * x_sW
        a = tl.load(out_ptr + off_out)
        b = tl.load(x_ptr + off_x)
        tl.store(out_ptr + off_out, a + b)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,  # (C, C, 3, 3)
        norm1_weight: torch.Tensor,  # (C,)
        norm1_bias: torch.Tensor,    # (C,)
        conv2_weight: torch.Tensor,  # (C, C, 3, 3)
        norm2_weight: torch.Tensor,  # (C,)
        norm2_bias: torch.Tensor,    # (C,)
        eps: float,
    ):
        # Ensure CUDA and float32 for simplicity and robustness
        assert x.is_cuda, "Input tensor must be on CUDA device"
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weight tensors must be on CUDA device"
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm tensors must be on CUDA device"

        # Cast to float32 for computation
        x32 = x.contiguous().to(torch.float32)
        conv1_w32 = conv1_weight.contiguous().to(torch.float32)
        conv2_w32 = conv2_weight.contiguous().to(torch.float32)
        norm1_w32 = norm1_weight.contiguous().to(torch.float32)
        norm1_b32 = norm1_bias.contiguous().to(torch.float32)
        norm2_w32 = norm2_weight.contiguous().to(torch.float32)
        norm2_b32 = norm2_bias.contiguous().to(torch.float32)

        N, C_in, H, W = x32.shape
        C = C_in  # each conv uses same C_in=C_out

        # Conv1
        y1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        grid1 = (N, C, H, W)
        conv3x3_str1_pad1_nchw[grid1](
            x32, conv1_w32, y1,
            N, C, C_in, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_w32.stride(0), conv1_w32.stride(1), conv1_w32.stride(2), conv1_w32.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )

        # GroupNorm1 reduction
        groups = 32
        assert (C % groups) == 0, "C must be divisible by num_groups=32"
        sums1 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)  # [sum, sumsq] per (n, group)

        grid_reduce1 = (N, groups)
        group


def run(*args):
    return ModelNew()(*args)
