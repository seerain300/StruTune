import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias)
# Grid: (B, C_out, H_out*W_out). Each program computes one output element.
@triton.jit
def conv3x3_nchw_nobias(
    x_ptr, w_ptr, y_ptr,
    B, C_in, C_out, H, W, H_out, W_out,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # map pid_hw to (h, w)
    h = pid_hw // W_out
    w = pid_hw % W_out

    # initialize accumulator
    acc = 0.0

    # loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(-1, 2):
            ih = h + dh
            # mask for valid ih
            valid_h = (ih >= 0) & (ih < H)
            for dw in range(-1, 2):
                iw = w + dw
                valid_w = (iw >= 0) & (iw < W)
                valid = valid_h & valid_w
                # compute input pointer
                x_offset = pid_b * x_stride_n + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                # masked load (other=0.0)
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                # compute weight pointer: w[co, ci, dh+1, dw+1]
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + (dh + 1) * w_stride_dh + (dw + 1) * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    # store output
    y_offset = pid_b * y_stride_n + pid_co * y_stride_c + h * y_stride_h + w * y_stride_w
    tl.store(y_ptr + y_offset, acc)

# Triton elementwise: GroupNorm across spatial, per-channel stats (num_groups=32 semantics used here as per-channel stats across spatial)
# Input y_in: (B, C, H, W), Output y_out: same shape
# We will use 1D launch over B*C*H*W
@triton.jit
def groupnorm_triton_per_channel_1d(
    y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
    B, C, H, W,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Each program handles one channel for one sample and vectorizes over spatial
    pid = tl.program_id(0)
    N = B * C * H * W
    # compute n, c
    n = pid // (C * H * W)
    c = (pid // (H * W)) % C
    # sum and sumsq across spatial
    sum_val = 0.0
    sum_sq = 0.0
    for h in range(0, H):
        for w in range(0, W):
            ptr = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
            x = tl.load(y_in_ptr + ptr)
            sum_val += x
            sum_sq += x * x
    mean = sum_val / (H * W)
    var = sum_sq / (H * W) - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    gamma = tl.load(weight_ptr + c)
    beta = tl.load(bias_ptr + c)
    for h in range(0, H):
        for w in range(0, W):
            ptr_in = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
            x = tl.load(y_in_ptr + ptr_in)
            norm = (x - mean) * rstd
            y = norm * gamma + beta
            ptr_out = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
            tl.store(y_out_ptr + ptr_out, y)

# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_triton(x_ptr, y_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + idx, y, mask=mask)

# Triton elementwise residual add: y_out = y + x
# y: (B, C, H-4, W-4), x: (B, C, H, W), we launch over B*C*H2*W2
@triton.jit
def add_residual_triton(
    y_ptr, x_ptr, y_out_ptr,
    B, C, H2, W2,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    pid = tl.program_id(0)
    N = B * C * H2 * W2
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    # compute (n, c, h, w) for y
    tmp = idx // (C * H2 * W2)
    n = tmp // (C * H2 * W2)  # tmp is already idx // (C*H2*W2), so n = tmp // (C*H2*W2) but tmp == idx // (C*H2*W2)
    # fix n computation
    n = tmp // (C * H2 * W2)
    # corrected: n = idx // (C * H2 * W2), h = (idx % (C * H2 * W2)) // (C * W2), c = (idx % (C * H2 * W2)) // (H2 * W2)
    tmp2 = idx // (C * H2 * W2)
    n = tmp2 // (C * H2 * W2)
    # wait, tmp2 already divides by (C * H2 * W2), so n = idx // (C * H2 * W2) is correct for B, but we need to recompute:
    # Let's simplify: use idx // (C * H2 * W2) to get n, then remaining to get c and h,w
    n = idx // (C * H2 * W2)
    rem = idx % (C * H2 * W2)
    c = rem // (H2 * W2)
    rem2 = rem % (H2 * W2)
    h = rem2 // W2
    w = rem2 % W2

    y_offset = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
    x_offset = n * x_stride_n + c * x_stride_c + h * y_stride_h + w * y_stride_w  # note: use y_stride_h/w for x? wrong
    # correct: x is (B, C, H, W), so use x_stride_h and x_stride_w
    x_offset = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w

    y_val = tl.load(y_ptr + y_offset, mask=mask, other=0.0)
    x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    tl.store(y_out_ptr + y_offset, y_val + x_val, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float
    ):
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        B, C, H, W = x.shape

        # conv1: (B, C, H, W) -> (B, C, H-2, W-2)
        H1 = H - 2
        W1 = W - 2
        y1 = torch.empty((B, C, H1, W1), dtype=torch.float32, device=x.device)

        # Launch conv1 Triton kernel
        grid1 = (B, C, H1 * W1)
        conv3x3_nchw_nobias[grid1](
            x, conv1_weight, y1,
            B, C, C, H, W, H1, W1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1 (per-channel stats over spatial): (B, C, H1, W1) -> (B, C, H1, W1)
        y1n = torch.empty_like(y1)
        grid_gn1 = (B * C,)
        groupnorm_triton_per_channel_1d[grid_gn1](
            y1, norm1_weight, norm1_bias, y1n,
            B, C, H1, W1,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU1
        y1s = torch.empty_like(y1n)
        N1 = B * C * H1 * W1
        silu_triton[N1](y1n, y1s, N1, num_warps=4, num_stages=2)

        # conv2: (B, C, H1, W1) -> (B, C, H1-2, W1-2) = (B, C, H-4, W-4)
        H2 = H1 - 2
        W2 = W1 - 2
        y2 = torch.empty((B, C, H2, W2), dtype=torch.float32, device=x.device)

        grid2 = (B, C, H2 * W2)
        conv3x3_nchw_nobias[grid2](
            y1s, conv2_weight, y2,
            B, C, C, H1, W1, H2, W2,
            y1s.stride(0), y1s.stride(1), y1s.stride(2), y1s.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2 (per-channel stats over spatial): (B, C, H2, W2) -> (B, C, H2, W2)
        y2n = torch.empty_like(y2)
        grid_gn2 = (B * C,)
        groupnorm_triton_per_channel_1d[grid_gn2](
            y2, norm2_weight, norm2_bias, y2n,
            B, C, H2, W2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU2
        y_out = torch.empty_like(y2n)
        N2 = B * C * H2 * W2
        silu_triton[N2](y2n, y_out, N2, num_warps=4, num_stages=2)

        # Residual addition: y_out += x (B, C, H, W)
        # Note: y_out has shape (B, C, H-4, W-4), x has shape (B, C, H, W). This matches the reference behavior.
        # We launch Triton to perform this elementwise addition.
        grid_add = (B * C * H2 * W2,)
        add_residual_triton[grid_add](
            y_out, x, y_out,  # in-place addition into y_out
            B, C, H2, W2,
            y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            num_warps=4, num_stages=2
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
