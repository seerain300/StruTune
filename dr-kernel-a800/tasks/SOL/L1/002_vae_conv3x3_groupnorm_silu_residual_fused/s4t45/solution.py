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
    # Grid: (B, C_out, H_out, W_out)
    n = tl.program_id(0)
    co = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    acc = 0.0  # scalar accumulation

    # Loop over input channels and 3x3 neighborhood with masks for padding
    for ci in range(0, C_in):
        for dh in range(-1, 2):
            hi = h + dh
            valid_h = (hi >= 0) & (hi < H)
            for dw in range(-1, 2):
                wi = w + dw
                valid_w = (wi >= 0) & (wi < W)
                valid = valid_h & valid_w
                x_off = n * x_stride_n + ci * x_stride_c + hi * x_stride_h + wi * x_stride_w
                x_val = tl.load(x_ptr + x_off) if valid else 0.0
                w_off = co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    # Store accumulated result
    y_off = n * y_stride_n + co * y_stride_c + h * y_stride_h + w * y_stride_w
    tl.store(y_ptr + y_off, acc)

# Triton kernel: GroupNorm over channels (per-channel stats across spatial), num_groups fixed
# We assume num_groups=32 and GroupNorm operates per channel across the entire feature map.
# grid: (B, C)
@triton.jit
def groupnorm_triton_channels(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                               B, C, H_out, W_out,
                               y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                               y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                               eps,  # numerical stability epsilon
                               num_warps: tl.constexpr, num_stages: tl.constexpr):
    n = tl.program_id(0)
    c = tl.program_id(1)

    N = H_out * W_out  # number of spatial elements per channel

    # Compute mean and variance across spatial plane
    sum_val = 0.0
    sum_sq = 0.0
    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr = n * y_in_stride_n + c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr)
            sum_val += x
            sum_sq += x * x

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Apply normalization and affine per channel
    for h in range(0, H_out):
        for w in range(0, W_out):
            in_ptr = n * y_in_stride_n + c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + in_ptr)
            norm = (x - mean) * rstd
            gamma = tl.load(weight_ptr + c)
            beta = tl.load(bias_ptr + c)
            y = norm * gamma + beta
            out_ptr = n * y_out_stride_n + c * y_out_stride_c + h * y_out_stride_h + w * y_out_stride_w
            tl.store(y_out_ptr + out_ptr, y)

# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                 num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    BLOCK = 256
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)

# Triton elementwise residual add: y = y + x
@triton.jit
def add_residual_triton(y_ptr, x_ptr, out_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    BLOCK = 256
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y + x, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                 conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        super().__init__()
        # Store parameters
        self.conv1_weight = conv1_weight
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.conv2_weight = conv2_weight
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.eps = eps  # GroupNorm epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dtype and contiguity; Triton requires CUDA tensors
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous()
        dtype = x.dtype
        assert dtype == torch.float32, "This Triton implementation currently supports float32."

        B, C, H, W = x.shape
        device = x.device

        # First conv: output (B, C, H-2, W-2)
        H1 = H - 2
        W1 = W - 2
        y1 = torch.empty((B, C, H1, W1), dtype=torch.float32, device=device)

        conv3x3_nchw_nobias[(B, C, H1, W1)](
            x, self.conv1_weight, y1,
            B, C, C, H, W, H1, W1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv1_weight.stride(0), self.conv1_weight.stride(1), self.conv1_weight.stride(2), self.conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1 (num_groups=32), per-channel across spatial
        y1_norm = torch.empty_like(y1)
        groupnorm_triton_channels[(B, C)](
            y1, self.norm1_weight, self.norm1_bias, y1_norm,
            B, C, H1, W1,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            self.eps,
            num_warps=4, num_stages=2
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        N1 = B * C * H1 * W1
        silu_triton[( (N1 + 256 - 1) // 256, )](
            y1_norm, y1_silu, N1,
            num_warps=2, num_stages=2
        )

        # Second conv: output (B, C, H-4, W-4)
        H2 = H1 - 2  # H - 4
        W2 = W1 - 2  # W - 4
        y2 = torch.empty((B, C, H2, W2), dtype=torch.float32, device=device)

        conv3x3_nchw_nobias[(B, C, H2, W2)](
            y1_silu, self.conv2_weight, y2,
            B, C, C, H1, W1, H2, W2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            self.conv2_weight.stride(0), self.conv2_weight.stride(1), self.conv2_weight.stride(2), self.conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2
        y2_norm = torch.empty_like(y2)
        groupnorm_triton_channels[(B, C)](
            y2, self.norm2_weight, self.norm2_bias, y2_norm,
            B, C, H2, W2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            self.eps,
            num_warps=4, num_stages=2
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        N2 = B * C * H2 * W2
        silu_triton[( (N2 + 256 - 1) // 256, )](
            y2_norm, y2_silu, N2,
            num_warps=2, num_stages=2
        )

        # Residual add: out = y2_silu + x
        # x has shape (B, C, H, W), y2_silu has shape (B, C, H-4, W-4). We add using broadcasting by cropping x:
        # Create x_cropped with shape (B, C, H-4, W-4) by slicing x[:, :, 2:H2+2, 2:W2+2]
        x_cropped = x[:, :, 2:2 + H2, 2:2 + W2].contiguous()
        out = torch.empty((B, C, H2, W2), dtype=torch.float32, device=device)
        N_add = B * C * H2 * W2
        add_residual_triton[( (N_add + 256 - 1) // 256, )](
            y2_silu, x_cropped, out, N_add,
            num_warps=2, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
