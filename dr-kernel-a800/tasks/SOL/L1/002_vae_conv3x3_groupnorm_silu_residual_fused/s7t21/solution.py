import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,                # *float32 input tensor (B, C_in, H, W)
    w_ptr,                # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,                # *float32 output tensor (B, C_out, H, W)
    N, C_in, C_out, H, W, # sizes
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC), ceil_div(H*W, BLOCK_HW))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    hw_block_id = tl.program_id(2)

    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    hw_start = hw_block_id * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < (H * W)
    h = hw_offsets // W
    w = hw_offsets % W

    # Accumulator per output channel in tile
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels
    for cin in range(C_in):
        # Loop over 3x3 kernel taps
        for kh in range(3):
            for kw in range(3):
                ih = h + kh - 1
                iw = w + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & hw_mask

                # Input linear index: n*C_in*H*W + cin*H*W + ih*W + iw
                in_ptrs = n * (C_in * H * W) + cin * (H * W) + ih * W + iw
                x_vals = tl.load(x_ptr + in_ptrs, mask=valid, other=0.0)  # [BLOCK_HW]

                # Load weights for all oc in tile: w[oc, cin, kh, kw]
                # Weight linear index: oc*C_in*9 + cin*9 + kh*3 + kw
                for j in range(BLOCK_OC):
                    if oc_mask[j]:
                        w_index = oc_offsets[j] * (C_in * 9) + cin * 9 + kh * 3 + kw
                        w_val = tl.load(w_ptr + w_index)  # scalar
                        acc[j] += tl.sum(x_vals * w_val, axis=0)  # sum over BLOCK_HW to scalar

    # Store results y[n, oc, h, w] for all h,w in this block
    for j in range(BLOCK_OC):
        if oc_mask[j]:
            # y linear index: n*C_out*H*W + oc*H*W + h*W + w
            y_ptrs = n * (C_out * H * W) + oc_offsets[j] * (H * W) + h * W + w
            tl.store(y_ptr + y_ptrs, acc[j], mask=hw_mask)


@triton.jit
def group_norm_affine_kernel(
    x_ptr,                # input [N, C, H, W] with strides
    weight_ptr,           # per-channel scale [C] with strides
    bias_ptr,             # per-channel bias [C] with strides
    y_ptr,                # output [N, C, H, W] with strides
    N, C, H, W,           # sizes
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    weight_stride_c, bias_stride_c,
    num_groups: tl.constexpr,        # number of groups (32)
    eps,                          # epsilon for numerical stability
):
    # Each program handles one (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Pass 1: compute sum and sumsq over the group
    sum_g = 0.0
    sumsq_g = 0.0

    for cin in range(channels_per_group):
        c = group_start + cin
        for oh in range(H):
            for ow in range(W):
                x_index = n * (C * H * W) + c * (H * W) + oh * W + ow
                x_val = tl.load(x_ptr + x_index)
                sum_g += x_val
                sumsq_g += x_val * x_val

    mean = sum_g / (channels_per_group * H * W)
    var = sumsq_g / (channels_per_group * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and affine
    for cin in range(channels_per_group):
        c = group_start + cin
        w_scale = tl.load(weight_ptr + c * weight_stride_c)
        b_bias = tl.load(bias_ptr + c * bias_stride_c)
        for oh in range(H):
            for ow in range(W):
                x_index = n * (C * H * W) + c * (H * W) + oh * W + ow
                x_val = tl.load(x_ptr + x_index)
                y_val = (x_val - mean) * inv_std * w_scale + b_bias
                y_index = n * (C * H * W) + c * (H * W) + oh * W + ow
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr,                # *float32 input [N, C, H, W]
    y_ptr,                # *float32 output [N, C, H, W]
    N, C, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    # Simple elementwise kernel
    # Flatten over N, C, H, W for straightforward indexing
    total = N * C * H * W
    pid = tl.program_id(0)
    idx = pid
    while idx < total:
        n = idx // (C * H * W)
        rem1 = idx % (C * H * W)
        c = rem1 // (H * W)
        rem2 = rem1 % (H * W)
        h = rem2 // W
        w = rem2 % W

        x_index = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
        x_val = tl.load(x_ptr + x_index)
        # SiLU: x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-x_val))
        y_val = x_val * sig
        y_index = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptr + y_index, y_val)
        idx += 1


@triton.jit
def add_residual_kernel(
    y_ptr,                # *float32 input [N, C, H, W] (post-SiLU result)
    x_ptr,                # *float32 input [N, C, H, W] (residual x)
    out_ptr,              # *float32 output [N, C, H, W]
    N, C, H, W,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    total = N * C * H * W
    pid = tl.program_id(0)
    idx = pid
    while idx < total:
        n = idx // (C * H * W)
        rem1 = idx % (C * H * W)
        c = rem1 // (H * W)
        rem2 = rem1 % (H * W)
        h = rem2 // W
        w = rem2 % W

        y_index = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
        x_index = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
        out_index = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w

        y_val = tl.load(y_ptr + y_index)
        x_val = tl.load(x_ptr + x_index)
        out_val = y_val + x_val
        tl.store(out_ptr + out_index, out_val)
        idx += 1


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        # For the given workload, num_groups=32 is hardcoded; enforce divisibility.
        # If needed, you can adapt num_groups dynamically based on C.

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Assumptions and assertions for correctness
        assert x.ndim == 4, "Input must be (B, C, H, W)"
        N, C, H, W = x.shape
        assert conv1_weight.shape[0] == C, "conv1_weight in_channels must match x.channels"
        assert conv2_weight.shape[0] == C, "conv2_weight in_channels must match x.channels"
        assert conv1_weight.shape[1] == C, "conv1_weight out_channels must match x.channels"
        assert conv2_weight.shape[1] == C, "conv2_weight out_channels must match x.channels"
        assert conv1_weight.shape[2] == 3 and conv1_weight.shape[3] == 3, "First conv must be 3x3"
        assert conv2_weight.shape[2] == 3 and conv2_weight.shape[3] == 3, "Second conv must be 3x3"
        assert C % self.num_groups == 0, "Channels must be divisible by num_groups for GroupNorm"

        # Prepare dtypes and devices; use float32 for stability
        device = x.device
        dtype = x.dtype  # assume float32
        x_contig = x.contiguous().to(torch.float32)

        # 1) Conv1 (Triton)
        y1 = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
        conv3x3_stride1_pad1_kernel[(N, triton.cdiv(C, 16), triton.cdiv(H * W, 1024))](
            x_contig, conv1_weight.to(torch.float32).contiguous(), y1,
            N, C, C, H, W,
            x_contig.stride(0), x_contig.stride(1), x_contig.stride(2), x_contig.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=16, BLOCK_HW=1024, num_warps=4, num_stages=2
        )

        # 2) GroupNorm1 (Triton)
        y1_norm = torch.empty_like(y1)
        group_norm_affine_kernel[(N, self.num_groups)](
            y1, norm1_weight.to(torch.float32).contiguous(), norm1_bias.to(torch.float32).contiguous(), y1_norm,
            N, C, H, W,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            norm1_weight.stride(0), norm1_bias.stride(0),
            num_groups=self.num_groups, eps=eps, num_warps=4, num_stages=2
        )

        # 3) SiLU1 (Triton)
        y1_silu = torch.empty_like(y1_norm)
        total_elems = N * C * H * W
        silu_kernel[(triton.cdiv(total_elems, 1024),)](
            y1_norm, y1_silu,
            N, C, H, W,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            num_warps=4, num_stages=2
        )

        # Save residual x for addition
        residual = x_contig

        # 4) Conv2 (Triton)
        y2 = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
        conv3x3_stride1_pad1_kernel[(N, triton.cdiv(C, 16), triton.cdiv(H * W, 1024))](
            y1_silu, conv2_weight.to(torch.float32).contiguous(), y2,
            N, C, C, H, W,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=16, BLOCK_HW=1024, num_warps=4, num_stages=2
        )

        # 5) GroupNorm2 (Triton)
        y2_norm = torch.empty_like(y2)
        group_norm_affine_kernel[(N, self.num_groups)](
            y2, norm2_weight.to(torch.float32).contiguous(), norm2_bias.to(torch.float32).contiguous(), y2_norm,
            N, C, H, W,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            norm2_weight.stride(0), norm2_bias.stride(0),
            num_groups=self.num_groups, eps=eps, num_warps=4, num_stages=2
        )

        # 6) SiLU2 (Triton)
        y2_silu = torch.empty_like(y2_norm)
        silu_kernel[(triton.cdiv(total_elems, 1024),)](
            y2_norm, y2_silu,
            N, C, H, W,
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            y2_silu.stride(0), y2_silu.stride(1), y2_silu.stride(2), y2_silu.stride(3),
            num_warps=4, num_stages=2
        )

        # 7) Add residual (Triton)
        out = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
        add_residual_kernel[(triton.cdiv(total_elems, 1024),)](
            y2_silu, residual, out,
            N, C, H, W,
            y2_silu.stride(0), y2_silu.stride(1), y2_silu.stride(2), y2_silu.stride(3),
            residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            num_warps=4, num_stages=2
        )

        # Cast back to original dtype if needed
        if dtype != torch.float32:
            out = out.to(dtype)

        return out


def run(*args):
    return ModelNew()(*args)
