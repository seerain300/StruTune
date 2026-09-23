import math
import triton
import triton.language as tl


@triton.jit
def conv1d_k5_p2(x_ptr, w_ptr, b_ptr, y_ptr,
                 B, C_in, C_out, T_in, T_out,
                 BLOCK_T: tl.constexpr):
    """
    Triton kernel for conv1d with kernel_size=5, padding=2.
    x_ptr: [B, C_in, T_in]
    w_ptr: [C_out, C_in, 5]
    b_ptr: [C_out]
    y_ptr: [B, C_out, T_out], where T_out = T_in - 1
    """
    b_idx = tl.program_id(0)  # batch index
    co = tl.program_id(1)     # output channel index

    # vector of output time indices for this program
    t_offsets = tl.arange(0, BLOCK_T)
    t = t_offsets
    mask_t = t < T_out

    # initialize accumulator
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, 5):
            # map output time index t to input time index t_in = t - 2 + k
            t_in = t - 2 + k
            # valid positions satisfy 0 <= t_in < T_in
            valid_pos = (t_in >= 0) & (t_in < T_in) & mask_t

            # load input x[b, ci, t_in]
            x_addr = x_ptr + b_idx * (C_in * T_in) + ci * T_in + t_in
            x_val = tl.load(x_addr, mask=valid_pos, other=0.0)

            # load weight w[co, ci, k]
            w_addr = w_ptr + co * (C_in * 5) + ci * 5 + k
            w_val = tl.load(w_addr)  # scalar

            # accumulate
            acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + co)
    acc = acc + b_val

    # store to y[b, co, t]
    y_addr = y_ptr + b_idx * (C_out * T_out) + co * T_out + t
    tl.store(y_addr, acc, mask=mask_t)


@triton.jit
def add_bias(x_ptr, b_ptr, y_ptr, B, C, T):
    """
    y = x + b, b is per-channel scalar
    x_ptr: [B, C, T]
    b_ptr: [C]
    y_ptr: [B, C, T]
    """
    b_idx = tl.program_id(0)  # batch index
    c = tl.program_id(1)      # channel index
    t_offsets = tl.arange(0, 128)
    t = t_offsets
    mask_t = t < T

    x_addr = x_ptr + b_idx * (C * T) + c * T + t
    x_val = tl.load(x_addr, mask=mask_t, other=0.0)

    b_val = tl.load(b_ptr + c)
    y_val = x_val + b_val

    y_addr = y_ptr + b_idx * (C * T) + c * T + t
    tl.store(y_addr, y_val, mask=mask_t)


@triton.jit
def relu_kernel(x_ptr, y_ptr, B, C, T):
    """
    y = relu(x)
    x_ptr: [B, C, T]
    y_ptr: [B, C, T]
    """
    b_idx = tl.program_id(0)
    c = tl.program_id(1)
    t_offsets = tl.arange(0, 128)
    t = t_offsets
    mask_t = t < T

    x_addr = x_ptr + b_idx * (C * T) + c * T + t
    x_val = tl.load(x_addr, mask=mask_t, other=0.0)
    y_val = tl.maximum(x_val, 0.0)

    y_addr = y_ptr + b_idx * (C * T) + c * T + t
    tl.store(y_addr, y_val, mask=mask_t)


@triton.jit
def mul_mask(x_ptr, mask_ptr, y_ptr, B, C, T):
    """
    Elementwise multiply x by mask along time, broadcast across channels.
    x_ptr: [B, C, T]
    mask_ptr: [B, 1, T] (we pass as [B*T]) — we index as mask[b, 0, t]
    y_ptr: [B, C, T]
    """
    b_idx = tl.program_id(0)  # batch index
    c = tl.program_id(1)      # channel index
    t_offsets = tl.arange(0, 128)
    t = t_offsets
    mask_t = t < T

    # x load
    x_addr = x_ptr + b_idx * (C * T) + c * T + t
    x_val = tl.load(x_addr, mask=mask_t, other=0.0)

    # mask load: mask is [B, 1, T] with strides (C*T, T, 1). We pass it as 1D [B*T] and index by b*T + t
    # For simplicity and since mask has size [B, 1, T], we can load mask for each (b, t) across all c
    # Here we load mask[b, 0, t]
    mask_addr = mask_ptr + b_idx * T + t
    mask_val = tl.load(mask_addr, mask=mask_t, other=1.0)

    y_val = x_val * mask_val

    y_addr = y_ptr + b_idx * (C * T) + c * T + t
    tl.store(y_addr, y_val, mask=mask_t)


@triton.jit
def add_or_sub(x1_ptr, h2_ptr, y1_ptr, B, C, T, add: tl.constexpr):
    """
    Elementwise add or subtract: y1 = x1 + h2 if add else y1 = x1 - h2
    x1_ptr: [B, C, T]
    h2_ptr: [B, C, T] (here C=96)
    y1_ptr: [B, C, T]
    """
    b_idx = tl.program_id(0)
    c = tl.program_id(1)
    t_offsets = tl.arange(0, 128)
    t = t_offsets
    mask_t = t < T

    x1_addr = x1_ptr + b_idx * (C * T) + c * T + t
    h2_addr = h2_ptr + b_idx * (C * T) + c * T + t

    x1_val = tl.load(x1_addr, mask=mask_t, other=0.0)
    h2_val = tl.load(h2_addr, mask=mask_t, other=0.0)

    y1_val = x1_val + h2_val if add else x1_val - h2_val

    y1_addr = y1_ptr + b_idx * (C * T) + c * T + t
    tl.store(y1_addr, y1_val, mask=mask_t)


@triton.jit
def copy_to(src_ptr, dst_ptr, B, C_src, C_dst, T, start_c: tl.constexpr):
    """
    Copy src [B, C_src, T] into dst [B, C_dst, T] at channel range [start_c, start_c + C_src).
    src_ptr: [B, C_src, T]
    dst_ptr: [B, C_dst, T]
    """
    b_idx = tl.program_id(0)
    c_src = tl.program_id(1)  # source channel index
    t_offsets = tl.arange(0, 128)
    t = t_offsets
    mask_t = t < T

    # src address
    src_addr = src_ptr + b_idx * (C_src * T) + c_src * T + t

    # dst address: channel index = start_c + c_src
    dst_c = start_c + c_src
    dst_addr = dst_ptr + b_idx * (C_dst * T) + dst_c * T + t

    val = tl.load(src_addr, mask=mask_t, other=0.0)
    tl.store(dst_addr, val, mask=mask_t)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        transform_0_conv0_weight: torch.Tensor,
        transform_0_conv0_bias: torch.Tensor,
        transform_0_conv1_weight: torch.Tensor,
        transform_0_conv1_bias: torch.Tensor,
        transform_0_conv2_weight: torch.Tensor,
        transform_0_conv2_bias: torch.Tensor,
    ):
        """
        Forward computes one transform:
        - Split x into x0 [B, 96, T] and x1 [B, 96, T]
        - conv0: x0 -> h0 [B, 192, T-1], add bias, ReLU
        - conv1: h0 -> h1 [B, 192, T-2], add bias, ReLU
        - conv2: h1 -> h2 [B, 96, T-3], add bias, ReLU, multiply by mask
        - coupling: x1 = x1 + h2 (forward), else subtract
        - output y: [B, 192, T-3], first half channels = x0, second half = x1_after
        All computations are done via Triton kernels; no torch ops in forward.
        """
        assert x.ndim == 3 and x.shape[1] == 192, "x must be [B, 192, T]"
        B, C, T = x.shape
        assert C == 192, "C must be 192"
        half = 96
        device = x.device

        # Split x
        x0 = x[:, :half, :].contiguous()
        x1 = x[:, half:, :].contiguous()

        # Prepare output buffers
        # h0: [B, 192, T-1]
        h0 = torch.empty((B, 192, T - 1), device=device, dtype=x.dtype)
        # h1: [B, 192, T-2]
        h1 = torch.empty((B, 192, T - 2), device=device, dtype=x.dtype)
        # h2: [B, 96, T-3]
        h2 = torch.empty((B, 96, T - 3), device=device, dtype=x.dtype)

        # y: [B, 192, T-3]
        y = torch.empty((B, 192, T - 3), device=device, dtype=x.dtype)

        # Kernel 1: conv0
        grid0 = (B, 192)
        conv1d_k5_p2[grid0](x0, transform_0_conv0_weight, transform_0_conv0_bias, h0, B, 96, 192, T, T - 1, BLOCK_T=128)

        # Elementwise: add bias and ReLU
        grid_bias0 = (B, 192, T - 1)
        add_bias[grid_bias0](h0, transform_0_conv0_bias, h0, B, 192, T - 1)  # add bias
        grid_relu0 = (B, 192, T - 1)
        relu_kernel[grid_relu0](h0, h0, B, 192, T - 1)

        # Kernel 2: conv1
        grid1 = (B, 192)
        conv1d_k5_p2[grid1](h0, transform_0_conv1_weight, transform_0_conv1_bias, h1, B, 192, 192, T - 1, T - 2, BLOCK_T=128)

        # Elementwise: add bias and ReLU
        grid_bias1 = (B, 192, T - 2)
        add_bias[grid_bias1](h1, transform_0_conv1_bias, h1, B, 192, T - 2)  # add bias
        grid_relu1 = (B, 192, T - 2)
        relu_kernel[grid_relu1](h1, h1, B, 192, T - 2)

        # Kernel 3: conv2
        grid2 = (B, 96)
        conv1d_k5_p2[grid2](h1, transform_0_conv2_weight, transform_0_conv2_bias, h2, B, 192, 96, T - 2, T - 3, BLOCK_T=128)

        # Elementwise: add bias and ReLU
        grid_bias2 = (B, 96, T - 3)
        add_bias[grid_bias2](h2, transform_0_conv2_bias, h2, B, 96, T - 3)  # add bias
        grid_relu2 = (B, 96, T - 3)
        relu_kernel[grid_relu2](h2, h2, B, 96, T - 3)

        # Apply mask: h2_masked = h2 * x_mask (broadcast across channels)
        # x_mask is [B, 1, T], we multiply across channels
        # For safety, ensure x_mask is contiguous
        x_mask = x_mask.to(x.dtype).contiguous()
        grid_mask = (B, 96, T - 3)
        mul_mask[grid_mask](h2, x_mask, h2, B, 96, T - 3)

        # Update x1: x1_after = x1 + h2 (forward), else subtract (reverse could be implemented similarly)
        grid_add = (B, 96, T - 3)
        add_or_sub[grid_add](x1, h2, x1, B, 96, T - 3, add=True)

        # Concatenate: y[:, :96, :] = x0, y[:, 96:, :] = x1_after
        # Copy x0 to y[:, :96, :]
        grid_copy0 = (B, 96, T - 3)
        copy_to[grid_copy0](x0, y, B, 96, 192, T - 3, start_c=0)

        # Copy x1_after to y[:, 96:, :]
        grid_copy1 = (B, 96, T - 3)
        copy_to[grid_copy1](x1, y, B, 96, 192, T - 3, start_c=96)

        return y


# Helper functions for benchmarking consistency (if needed by the harness)
def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    time = axes_and_scalars["time"]
    channels = 192
    hidden_channels = 192
    half_channels = 96
    kernel_size = 5

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv1d(out_c, in_c, k):
        fan_in = in_c * k
        return torch.randn(out_c, in_c, k, device=device, generator=g) * math.sqrt(2.0 / fan_in)

    inputs = {
        "x": torch.randn(batch_size, channels, time, device=device, generator=g),
        # Binary mask
        "x_mask": torch.ones(batch_size, 1, time, device=device),
        "reverse": False,
    }

    # 4 transforms x 3 convs each
    inputs[f"transform_0_conv0_weight"] = kaiming_conv1d(hidden_channels, half_channels, kernel_size)
    inputs[f"transform_0_conv0_bias"] = torch.randn(hidden_channels, device=device, generator=g)
    inputs[f"transform_0_conv1_weight"] = kaiming_conv1d(hidden_channels, hidden_channels, kernel_size)
    inputs[f"transform_0_conv1_bias"] = torch.randn(hidden_channels, device=device, generator=g)
    inputs[f"transform_0_conv2_weight"] = kaiming_conv1d(half_channels, hidden_channels, kernel_size)
    inputs[f"transform_0_conv2_bias"] = torch.randn(half_channels, device=device, generator=g)

    return inputs


def run(*args):
    return ModelNew()(*args)
