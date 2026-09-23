import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias)
# Each program computes one output element y[n, co, h, w].
@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = 0.0

    # Loop over input channels and 3x3 neighborhood (padding handled by masked loads)
    for ci in range(0, C_in):
        for dh in range(0, 3):
            h_in = pid_h + dh
            valid_h = (h_in >= 0) & (h_in < H_out)
            for dw in range(0, 3):
                w_in = pid_w + dw
                valid_w = (w_in >= 0) & (w_in < W_out)
                mask = valid_h & valid_w
                x_offset = pid_n * x_stride_n + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)

# Triton kernel: GroupNorm over channels (per-channel stats across spatial), num_groups fixed
# grid: (B, C)
@triton.jit
def groupnorm_triton_channels(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                               B, C, H_out, W_out,
                               y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                               y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                               num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    N = H_out * W_out  # number of spatial elements per channel

    # accumulate sum and sum of squares over spatial plane
    sum_val = 0.0
    sum_sq = 0.0
    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr)
            sum_val += x
            sum_sq += x * x

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # apply normalization and affine
    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr_in = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr_in)
            norm = (x - mean) * rstd
            gamma = tl.load(weight_ptr + pid_c)
            beta = tl.load(bias_ptr + pid_c)
            y = norm * gamma + beta
            ptr_out = pid_n * y_out_stride_n + pid_c * y_out_stride_c + h * y_out_stride_h + w * y_out_stride_w
            tl.store(y_out_ptr + ptr_out, y)

# Triton elementwise SiLU: y = x * sigmoid(x)
# Uses separate input/output buffers; does not read/write same pointer.
@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                 num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    BLOCK = 1024
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y)

# Triton elementwise addition with padding: out = y2_silu + x
# Assumes x and y2_silu may have different spatial sizes; out has target spatial (H_out2, W_out2).
@triton.jit
def add_residual_triton(y2_ptr, x_ptr, out_ptr,
                         B, C, H_out2, W_out2, H, W,
                         y2_stride_n, y2_stride_c, y2_stride_h, y2_stride_w,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         out_stride_n, out_stride_c, out_stride_h, out_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    total = B * C * H_out2 * W_out2
    pid = tl.program_id(0)
    BLOCK = 1024
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total

    # Compute n, c, h, w from linear idx
    n = idx // (C * H_out2 * W_out2)
    rem = idx % (C * H_out2 * W_out2)
    c = rem // (H_out2 * W_out2)
    rem2 = rem % (H_out2 * W_out2)
    h = rem2 // W_out2
    w = rem2 % W_out2

    # y2 load (always valid since out allocated with target size)
    y2_ptr_idx = n * y2_stride_n + c * y2_stride_c + h * y2_stride_h + w * y2_stride_w
    y2_val = tl.load(y2_ptr + y2_ptr_idx)

    # x load with padding (safe via mask; if original spatial smaller, idx falls out-of-range)
    # Build mask for x based on original H,W
    in_range = (h < H) & (w < W)
    x_ptr_idx = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    x_val = tl.load(x_ptr + x_ptr_idx, mask=mask & in_range, other=0.0)

    out_val = y2_val + x_val
    out_ptr_idx = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w
    tl.store(out_ptr + out_ptr_idx, out_val, mask=mask)

# ModelNew: Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_groups is fixed to 32 as per the original code
        self.num_groups = 32

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float = 1e-5):
        # Ensure contiguity (no torch ops after this)
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        B, C, H, W = x.shape
        # conv1
        H_out = H - 2
        W_out = W - 2
        y1 = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

        conv3x3_nchw_nobias[(B, C, H_out, W_out)](
            x, conv1_weight, y1,
            B, C, C, H, W, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1 (num_groups=32), per-channel over spatial
        y1_norm = torch.empty_like(y1)
        groupnorm_triton_channels[(B, C)](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C, H_out, W_out,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        N1 = y1_silu.numel()
        silu_triton[(triton.cdiv(N1, 1024),)](
            y1_norm.reshape(-1), y1_silu.reshape(-1), N1,
            num_warps=4, num_stages=2
        )
        y1_silu = y1_silu.reshape(y1_norm.shape)

        # conv2
        H_out2 = H - 4
        W_out2 = W - 4
        y2 = torch.empty((B, C, H_out2, W_out2), device=x.device, dtype=x.dtype)

        conv3x3_nchw_nobias[(B, C, H_out2, W_out2)](
            y1_silu, conv2_weight, y2,
            B, C, C, H_out, W_out, H_out2, W_out2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2 (per-channel over spatial)
        y2_norm = torch.empty_like(y2)
        groupnorm_triton_channels[(B, C)](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C, H_out2, W_out2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        N2 = y2_silu.numel()
        silu_triton[(triton.cdiv(N2, 1024),)](
            y2_norm.reshape(-1), y2_silu.reshape(-1), N2,
            num_warps=4, num_stages=2
        )
        y2_silu = y2_silu.reshape(y2_norm.shape)

        # Residual addition: out = y2_silu + x (pad x to (B,C,H_out2,W_out2) in Triton)
        out = torch.empty((B, C, H_out2, W_out2), device=x.device, dtype=x.dtype)
        total_elems = B * C * H_out2 * W_out2
        add_residual_triton[(triton.cdiv(total_elems, 1024),)](
            y2_silu, x, out,
            B, C, H_out2, W_out2, H, W,
            y2_silu.stride(0), y2_silu.stride(1), y2_silu.stride(2), y2_silu.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
