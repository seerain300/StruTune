import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias)
@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    # program ids
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # bounds check (not strictly necessary if grid matches)
    # accumulate in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(0, 3):
            in_h = pid_h + dh
            valid_h = (in_h >= 0) & (in_h < H)
            for dw in range(0, 3):
                in_w = pid_w + dw
                valid_w = (in_w >= 0) & (in_w < W)
                valid = valid_h & valid_w

                # load weight scalar w[co, ci, dh, dw]
                w_off = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_off)

                # load input scalar x[b, ci, in_h, in_w] with mask
                x_off = pid_b * x_stride_n + ci * x_stride_c + in_h * x_stride_h + in_w * x_stride_w
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)

                acc += x_val * w_val

    # store output
    y_off = pid_b * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_off, acc)

# Triton kernel: GroupNorm (num_groups=32, per-channel affine) per (n, c) across spatial H*W
@triton.jit
def group_norm_triton(x_ptr, scale_ptr, bias_ptr, y_ptr,
                       B, C, H_in, W_in,
                       x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                       y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                       eps: tl.constexpr,
                       num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid: (B, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Compute sum and sum of squares across all spatial positions for channel pid_c in sample pid_n
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    total_elems = H_in * W_in
    for h in range(0, H_in):
        for w in range(0, W_in):
            x_off = pid_n * x_stride_n + pid_c * x_stride_c + h * x_stride_h + w * x_stride_w
            x_val = tl.load(x_ptr + x_off)
            sum_val += x_val
            sum_sq += x_val * x_val

    mean = sum_val / total_elems
    var = sum_sq / total_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Write normalized output with per-channel affine
    gamma = tl.load(scale_ptr + pid_c)
    beta = tl.load(bias_ptr + pid_c)

    for h in range(0, H_in):
        for w in range(0, W_in):
            x_off = pid_n * x_stride_n + pid_c * x_stride_c + h * x_stride_h + w * x_stride_w
            x_val = tl.load(x_ptr + x_off)
            norm = (x_val - mean) * rstd
            y_val = norm * gamma + beta

            y_off = pid_n * y_stride_n + pid_c * y_stride_c + h * y_stride_h + w * y_stride_w
            tl.store(y_ptr + y_off, y_val)

# Triton kernel: SiLU activation elementwise
@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                 num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y, mask=mask)

# Triton kernel: elementwise residual add y = x1 + x2 (adds previous output to input)
@triton.jit
def add_residual_triton(x1_ptr, x2_ptr, y_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    a = tl.load(x1_ptr + idx, mask=mask, other=0.0)
    b = tl.load(x2_ptr + idx, mask=mask, other=0.0)
    y = a + b
    tl.store(y_ptr + idx, y, mask=mask)

# Helper to launch conv3x3 for a given (B, C_in, C_out, H, W) -> (B, C_out, H_out, W_out)
def conv3x3_launch(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    B, C_in = x.shape[0], x.shape[1]
    C_out = w.shape[0]
    H, W = x.shape[2], x.shape[3]
    H_out, W_out = H - 2, W - 2

    grid = (B, C_out, H_out, W_out)
    conv3x3_nchw_nobias[grid](
        x, w, out,
        B, C_in, C_out, H, W, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_warps=4, num_stages=2,
    )

# Forward: Triton-only implementation
class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                 conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        super().__init__()
        self.conv1_weight = conv1_weight
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.conv2_weight = conv2_weight
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.eps = eps
        self.num_groups = 32  # consistent with original Model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure NCHW, contiguous, float32
        x = x.contiguous().to(torch.float32)
        B, C, H, W = x.shape

        # conv1: (B, C, H, W) -> (B, C, H-2, W-2)
        y1 = torch.empty((B, C, H - 2, W - 2), device=x.device, dtype=torch.float32)
        conv3x3_launch(x, self.conv1_weight, y1)

        # GroupNorm1 (num_groups=32), per-channel affine
        y1_norm = torch.empty_like(y1)
        grid_groups1 = (B, C)
        group_norm_triton[grid_groups1](
            y1, self.norm1_weight, self.norm1_bias, y1_norm,
            B, C, y1.shape[2], y1.shape[3],
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            eps=self.eps,
            num_warps=4, num_stages=2,
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        N1 = y1_norm.numel()
        silu_triton[(triton.cdiv(N1, 1024),)](
            y1_norm, y1_silu, N1,
            num_warps=4, num_stages=2,
        )

        # conv2: (B, C, H-2, W-2) -> (B, C, H-4, W-4)
        y2 = torch.empty((B, C, H - 4, W - 4), device=x.device, dtype=torch.float32)
        conv3x3_launch(y1_silu, self.conv2_weight, y2)

        # GroupNorm2
        y2_norm = torch.empty_like(y2)
        grid_groups2 = (B, C)
        group_norm_triton[grid_groups2](
            y2, self.norm2_weight, self.norm2_bias, y2_norm,
            B, C, y2.shape[2], y2.shape[3],
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            eps=self.eps,
            num_warps=4, num_stages=2,
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        N2 = y2_norm.numel()
        silu_triton[(triton.cdiv(N2, 1024),)](
            y2_norm, y2_silu, N2,
            num_warps=4, num_stages=2,
        )

        # Residual addition: y = y2_silu + x (broadcasted over spatial dims)
        y_out = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        N = y_out.numel()
        add_residual_triton[(triton.cdiv(N, 1024),)](
            x, y2_silu, y_out, N,
            num_warps=4, num_stages=2,
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
