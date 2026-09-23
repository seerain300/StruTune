import torch
import triton
import triton.language as tl


# Triton kernel: conv2d 3x3, stride=1, padding=1, bias=None
# Each program computes a single output element y[n, c_out, h, w]
@triton.jit
def conv3x3_stride1_pad1_elem_kernel(
    x_ptr,           # *float32 input tensor: (B, C_in, H, W)
    w_ptr,           # *float32 weights tensor: (C_out, C_in, 3, 3)
    y_ptr,           # *float32 output tensor: (B, C_out, H, W)
    N,               # int: batch size
    C_in,            # int: input channels
    H,               # int: input height
    W,               # int: input width
    C_out,           # int: output channels
    H_out,           # int: output height (== H)
    W_out,           # int: output width (== W)
):
    # Grid: (N, C_out, H_out, W_out)
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for cin in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                ih = h + kh - 1  # stride=1, pad=1
                iw = w + kw - 1
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_index = (((n * C_in + cin) * H + ih) * W + iw)
                # Load input with mask; use 0.0 if out of bounds
                x_val = tl.load(x_ptr + x_index, mask=in_bounds, other=0.0)

                # Weight index: w_ptr layout (C_out, C_in, 3, 3)
                w_index = (c_out * (C_in * 9)) + (cin * 9) + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_index)

                acc += x_val * w_val

    # Store result
    y_index = (((n * C_out + c_out) * H_out + h) * W_out + w)
    tl.store(y_ptr + y_index, acc)


# Triton kernel: GroupNorm with per-channel affine, per-sample, per-group
# Assumes num_groups divides C; grid is (N, num_groups), each program handles one (n, g)
@triton.jit
def group_norm_affine_kernel(
    x_ptr,          # *float32 input tensor: (N, C, H, W)
    weight_ptr,     # *float32 scale tensor: (C,)
    bias_ptr,       # *float32 bias tensor: (C,)
    y_ptr,          # *float32 output tensor: (N, C, H, W)
    N,              # int
    C,              # int
    H,              # int
    W,              # int
    num_groups,     # int (32)
    eps,            # float
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    sum_all = tl.zeros((), dtype=tl.float32)
    sumsq_all = tl.zeros((), dtype=tl.float32)

    # First pass: compute sum and sum of squares for the group
    for c in range(0, channels_per_group):
        c_idx = group_start + c
        for h in range(0, H):
            for w in range(0, W):
                x_index = (((n * C + c_idx) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                sum_all += x_val
                sumsq_all += x_val * x_val

    # Compute mean and inv_std
    M = channels_per_group * H * W
    mean = sum_all / M
    var = sumsq_all / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, write to y
    for c in range(0, channels_per_group):
        c_idx = group_start + c
        w_scale = tl.load(weight_ptr + c_idx)
        b_bias = tl.load(bias_ptr + c_idx)
        for h in range(0, H):
            for w in range(0, W):
                x_index = (((n * C + c_idx) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                y_val = (x_val - mean) * inv_std
                y_val = y_val * w_scale + b_bias
                y_index = (((n * C + c_idx) * H + h) * W + w)
                tl.store(y_ptr + y_index, y_val)


# Triton kernel: SiLU activation elementwise
@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
):
    # Flatten iteration over N*C*H*W
    total = N * C * H * W
    pid = tl.program_id(0)
    for i in range(0, total):
        # Compute indices (n, c, h, w) from i
        n = i // (C * H * W)
        tmp = i % (C * H * W)
        c = tmp // (H * W)
        tmp = tmp % (H * W)
        h = tmp // W
        w = tmp % W

        x_index = (((n * C + c) * H + h) * W + w)
        x_val = tl.load(x_ptr + x_index)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-x_val))
        y_val = x_val * sig
        y_index = (((n * C + c) * H + h) * W + w)
        tl.store(y_ptr + y_index, y_val)


# Triton kernel: elementwise residual add: y = y + x
@triton.jit
def add_residual_kernel(
    y_ptr, x_ptr,
    N, C, H, W,
):
    total = N * C * H * W
    pid = tl.program_id(0)
    for i in range(0, total):
        n = i // (C * H * W)
        tmp = i % (C * H * W)
        c = tmp // (H * W)
        tmp = tmp % (H * W)
        h = tmp // W
        w = tmp % W

        y_index = (((n * C + c) * H + h) * W + w)
        y_val = tl.load(y_ptr + y_index)
        x_val = tl.load(x_ptr + y_index)  # same indexing
        y_val = y_val + x_val
        tl.store(y_ptr + y_index, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
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
        All computation is done via Triton kernels; no torch ops in forward.
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        # Ensure contiguous tensors
        x = x.contiguous()
        # conv1: F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        B, C_in, H, W = x.shape
        C1_out, C_in_w, KH, KW = conv1_weight.shape
        assert C_in_w == C_in and KH == 3 and KW == 3, "conv1_weight must be (C_out, C_in, 3, 3)"
        y1 = torch.empty((B, C1_out, H, W), dtype=torch.float32, device=x.device)

        # Launch conv1 kernel: grid (B, C1_out, H, W)
        grid_conv1 = (B, C1_out, H, W)
        conv3x3_stride1_pad1_elem_kernel[grid_conv1](
            x, conv1_weight, y1,
            B, C_in, H, W, C1_out, H, W,
            num_warps=4, num_stages=2,
        )

        # GroupNorm1: y1 with num_groups=32
        N, C1, H1, W1 = y1.shape
        # Check GroupNorm divisibility
        assert C1 % 32 == 0, "num_groups=32 requires C % 32 == 0 for GroupNorm"
        y1_norm = torch.empty_like(y1, dtype=torch.float32)

        grid_gn1 = (N, 32)
        group_norm_affine_kernel[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_norm,
            N, C1, H1, W1, 32, self.eps,
            num_warps=4, num_stages=2,
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm, dtype=torch.float32)
        total = N * C1 * H1 * W1
        grid_silu1 = (total,)
        silu_kernel[grid_silu1](y1_norm, y1_silu, N, C1, H1, W1, num_warps=4, num_stages=2)

        # conv2: F.conv2d(y1_silu, conv2_weight, bias=None, stride=1, padding=1)
        C2_out, C2_in_w, KH2, KW2 = conv2_weight.shape
        assert C2_in_w == C1 and KH2 == 3 and KW2 == 3, "conv2_weight must be (C_out, C_in, 3, 3)"
        y2 = torch.empty((N, C2_out, H1, W1), dtype=torch.float32, device=y1_silu.device)

        grid_conv2 = (N, C2_out, H1, W1)
        conv3x3_stride1_pad1_elem_kernel[grid_conv2](
            y1_silu, conv2_weight, y2,
            N, C1, H1, W1, C2_out, H1, W1,
            num_warps=4, num_stages=2,
        )

        # GroupNorm2
        N2, C2, H2, W2 = y2.shape
        assert C2 % 32 == 0, "num_groups=32 requires C % 32 == 0 for GroupNorm"
        y2_norm = torch.empty_like(y2, dtype=torch.float32)

        grid_gn2 = (N2, 32)
        group_norm_affine_kernel[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_norm,
            N2, C2, H2, W2, 32, self.eps,
            num_warps=4, num_stages=2,
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm, dtype=torch.float32)
        total2 = N2 * C2 * H2 * W2
        grid_silu2 = (total2,)
        silu_kernel[grid_silu2](y2_norm, y2_silu, N2, C2, H2, W2, num_warps=4, num_stages=2)

        # Residual add: y_out = y2_silu + x
        y_out = torch.empty_like(y2_silu, dtype=torch.float32)
        grid_add = (total2,)
        add_residual_kernel[grid_add](y2_silu, x, N2, C2, H2, W2, num_warps=4, num_stages=2)

        return y_out


# The original run function expects tensors with specific names; we provide a wrapper that uses ModelNew.
@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    model = ModelNew(eps)
    # Ensure inputs are on CUDA
    assert x.is_cuda, "x must be on CUDA"
    assert conv1_weight.is_cuda and conv2_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
           and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be on CUDA"
    return model(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias)


def run(*args):
    return ModelNew()(*args)
