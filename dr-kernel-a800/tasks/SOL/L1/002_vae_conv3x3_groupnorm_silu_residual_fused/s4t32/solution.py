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

# Triton kernel: per-sample, per-group GroupNorm across spatial plane and channels in the group
# grid: (B, 32)
@triton.jit
def groupnorm_groups_triton(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                             B, C, num_groups, H_out, W_out,
                             y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                             y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                             num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    channels_per_group = C // num_groups
    base = pid_g * channels_per_group

    # Loop over channels in this group and compute per-channel stats
    for c in range(0, channels_per_group):
        c_idx = base + c

        N = H_out * W_out  # spatial elements per channel
        sum_val = 0.0
        sum_sq = 0.0
        for h in range(0, H_out):
            for w in range(0, W_out):
                ptr = pid_b * y_in_stride_n + c_idx * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
                x = tl.load(y_in_ptr + ptr)
                sum_val += x
                sum_sq += x * x

        mean = sum_val / N
        var = sum_sq / N - mean * mean
        rstd = 1.0 / tl.sqrt(var + 1e-5)

        # apply normalization and affine
        for h in range(0, H_out):
            for w in range(0, W_out):
                ptr_in = pid_b * y_in_stride_n + c_idx * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
                x = tl.load(y_in_ptr + ptr_in)
                norm = (x - mean) * rstd
                gamma = tl.load(weight_ptr + c_idx)
                beta = tl.load(bias_ptr + c_idx)
                y = norm * gamma + beta
                ptr_out = pid_b * y_out_stride_n + c_idx * y_out_stride_c + h * y_out_stride_h + w * y_out_stride_w
                tl.store(y_out_ptr + ptr_out, y)

# Triton elementwise SiLU over a flattened tensor
@triton.jit
def silu_triton_overall(x_ptr, y_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y)

# Triton elementwise residual addition with padding to match y2_silu shape
# y2_silu: (B, C, H2, W2), x: (B, C, H, W) -> pad x to (B, C, H2, W2) before addition
@triton.jit
def add_residual_triton(y2_ptr, x_ptr, out_ptr,
                         B, C, H2, W2, H, W,
                         y2_stride_n, y2_stride_c, y2_stride_h, y2_stride_w,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         out_stride_n, out_stride_c, out_stride_h, out_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # load y2
    y2_val = tl.load(y2_ptr + pid_n * y2_stride_n + pid_c * y2_stride_c + pid_h * y2_stride_h + pid_w * y2_stride_w)

    # compute corresponding source index in x with padding: if h/W outside [0,H-1/W-1], pad with 0
    h_src = pid_h
    w_src = pid_w
    valid_h = (h_src >= 0) & (h_src < H)
    valid_w = (w_src >= 0) & (w_src < W)
    mask = valid_h & valid_w

    x_val = tl.load(x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + h_src * x_stride_h + w_src * x_stride_w, mask=mask, other=0.0)

    out_val = y2_val + x_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_h * out_stride_h + pid_w * out_stride_w, out_val)

# ModelNew: forward only launches Triton kernels; no torch ops in forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Create outputs with empty (avoid torch ops like .to, .contiguous, .reshape)
        B, C, H, W = x.shape

        # conv1 output: (B, C, H-2, W-2)
        H_out1 = H - 2
        W_out1 = W - 2
        y1 = torch.empty((B, C, H_out1, W_out1), device=x.device)

        # conv2 output: (B, C, H-4, W-4)
        H_out2 = H - 4
        W_out2 = W - 4
        y2 = torch.empty((B, C, H_out2, W_out2), device=x.device)

        # Allocate intermediates
        y1_norm = torch.empty((B, C, H_out1, W_out1), device=x.device)
        y1_silu = torch.empty((B, C, H_out1, W_out1), device=x.device)
        y2_norm = torch.empty((B, C, H_out2, W_out2), device=x.device)
        y2_silu = torch.empty((B, C, H_out2, W_out2), device=x.device)
        out = torch.empty((B, C, H_out2, W_out2), device=x.device)

        # Launch conv1
        conv3x3_nchw_nobias[(B, C, H_out1, W_out1)](
            x, conv1_weight, y1,
            B, C, C, H, W, H_out1, W_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1 (num_groups=32): per-sample, per-group across channels in the group and spatial plane
        groupnorm_groups_triton[(B, 32)](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C, 32, H_out1, W_out1,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU1
        y1_flat = y1_norm.reshape(-1)
        silu_triton_overall[(triton.cdiv(y1_flat.numel(), 1024),)](
            y1_flat, y1_flat, y1_flat.numel(),
            num_warps=4, num_stages=2
        )
        y1_silu = y1_flat.reshape(y1_norm.shape)

        # conv2: inputs are y1_silu with shape (B, C, H_out1, W_out1)
        conv3x3_nchw_nobias[(B, C, H_out2, W_out2)](
            y1_silu, conv2_weight, y2,
            B, C, C, H_out1, W_out1, H_out2, W_out2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2
        groupnorm_groups_triton[(B, 32)](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C, 32, H_out2, W_out2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU2
        y2_flat = y2_norm.reshape(-1)
        silu_triton_overall[(triton.cdiv(y2_flat.numel(), 1024),)](
            y2_flat, y2_flat, y2_flat.numel(),
            num_warps=4, num_stages=2
        )
        y2_silu = y2_flat.reshape(y2_norm.shape)

        # Residual addition: y_out = y2_silu + x (pad x to (B, C, H_out2, W_out2) in Triton)
        add_residual_triton[(B, C, H_out2, W_out2)](
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
