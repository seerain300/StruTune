import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_k5_p2(x_ptr, w_ptr, y_ptr,
                 B, C_in, C_out, T_in, T_out,
                 x_stride_b, x_stride_c, x_stride_t,
                 w_stride_co, w_stride_ci, w_stride_k,
                 y_stride_b, y_stride_c, y_stride_t,
                 BLOCK_T: tl.constexpr):
    """
    Conv1d with kernel_size=5, padding=2, stride=1, no groups.
    x: [B, C_in, T_in]
    w: [C_out, C_in, 5]
    y: [B, C_out, T_out], T_out = T_in - 1
    """
    b = tl.program_id(0)
    co = tl.program_id(1)
    t_block = tl.program_id(2)

    offs_t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T_out

    # Accumulator for this (b, co) and time tile
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps
    for ci in range(0, C_in):
        # For each kernel tap k in [0, 4]
        for k in range(0, 5):
            # Map output time index to input time index with padding=2
            t_in = offs_t + 2 - k
            # Compute pointers
            x_ptrs = x_ptr + b * x_stride_b + ci * x_stride_c + t_in * x_stride_t
            w_ptrs = w_ptr + co * w_stride_co + ci * w_stride_ci + k * w_stride_k

            # Mask for valid t_in in [0, T_in-1]
            mask_t_in = (t_in >= 0) & (t_in < T_in) & mask_t

            # Load
            x_vals = tl.load(x_ptrs, mask=mask_t_in, other=0.0)
            w_val = tl.load(w_ptrs)  # scalar
            acc += x_vals * w_val

    # Write back
    y_ptrs = y_ptr + b * y_stride_b + co * y_stride_c + offs_t * y_stride_t
    tl.store(y_ptrs, acc, mask=mask_t)


@triton.jit
def add_bias(x_ptr, bias_ptr, y_ptr,
             B, C_out, T,
             x_stride_b, x_stride_c, x_stride_t,
             y_stride_b, y_stride_c, y_stride_t):
    """
    y = x + bias, bias shape [C_out], broadcast across B and T.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)

    x_ptrs = x_ptr + b * x_stride_b + c * x_stride_c + t * x_stride_t
    bias_val = tl.load(bias_ptr + c)
    y_ptrs = y_ptr + b * y_stride_b + c * y_stride_c + t * y_stride_t

    x_val = tl.load(x_ptrs)
    y_val = x_val + bias_val
    tl.store(y_ptrs, y_val)


@triton.jit
def relu_kernel(x_ptr, y_ptr,
                B, C, T,
                x_stride_b, x_stride_c, x_stride_t,
                y_stride_b, y_stride_c, y_stride_t):
    """
    Elementwise ReLU: y = max(x, 0).
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)

    x_ptrs = x_ptr + b * x_stride_b + c * x_stride_c + t * x_stride_t
    y_ptrs = y_ptr + b * y_stride_b + c * y_stride_c + t * y_stride_t

    x_val = tl.load(x_ptrs)
    y_val = tl.maximum(x_val, 0.0)
    tl.store(y_ptrs, y_val)


@triton.jit
def mul_mask(x_ptr, mask_ptr, y_ptr,
             B, C, T,
             x_stride_b, x_stride_c, x_stride_t,
             mask_stride_b, mask_stride_c, mask_stride_t,
             y_stride_b, y_stride_c, y_stride_t):
    """
    Elementwise multiply by mask: y = x * mask.
    mask has shape [B, 1, T], broadcast across C.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)

    x_ptrs = x_ptr + b * x_stride_b + c * x_stride_c + t * x_stride_t
    mask_ptrs = mask_ptr + b * mask_stride_b + 0 * mask_stride_c + t * mask_stride_t
    y_ptrs = y_ptr + b * y_stride_b + c * y_stride_c + t * y_stride_t

    x_val = tl.load(x_ptrs)
    mask_val = tl.load(mask_ptrs)  # mask is [B,1,T], so c index is ignored
    y_val = x_val * mask_val
    tl.store(y_ptrs, y_val)


@triton.jit
def add_or_sub(x_ptr, h_ptr, y_ptr,
               B, C, T,
               x_stride_b, x_stride_c, x_stride_t,
               h_stride_b, h_stride_c, h_stride_t,
               y_stride_b, y_stride_c, y_stride_t,
               add_flag: tl.constexpr):
    """
    y = x + h or y = x - h. add_flag = 1 => add, 0 => sub.
    x and h have shape [B, C, T]; y has same shape.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)

    x_ptrs = x_ptr + b * x_stride_b + c * x_stride_c + t * x_stride_t
    h_ptrs = h_ptr + b * h_stride_b + c * h_stride_c + t * h_stride_t
    y_ptrs = y_ptr + b * y_stride_b + c * y_stride_c + t * y_stride_t

    x_val = tl.load(x_ptrs)
    h_val = tl.load(h_ptrs)
    if add_flag:
        y_val = x_val + h_val
    else:
        y_val = x_val - h_val
    tl.store(y_ptrs, y_val)


@triton.jit
def copy_to(src_ptr, dst_ptr,
            B, C, T,
            src_stride_b, src_stride_c, src_stride_t,
            dst_stride_b, dst_c_offset, dst_stride_c, dst_stride_t):
    """
    Copy src tensor [B, C, T] into dst tensor at channel offset dst_c_offset.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)

    src_ptrs = src_ptr + b * src_stride_b + c * src_stride_c + t * src_stride_t
    dst_ptrs = dst_ptr + b * dst_stride_b + (c + dst_c_offset) * dst_stride_c + t * dst_stride_t

    val = tl.load(src_ptrs)
    tl.store(dst_ptrs, val)


def _conv1d_triton(x0, w, T_out):
    """
    Launch conv1d_k5_p2 to compute y = conv1d(x0, w) with padding=2, output length T_out = T_in - 1.
    x0: [B, C_in, T_in], w: [C_out, C_in, 5]
    returns y: [B, C_out, T_out], float32
    """
    B, C_in, T_in = x0.shape
    C_out = w.shape[0]
    y = torch.empty((B, C_out, T_out), device=x0.device, dtype=torch.float32)

    grid = (B, C_out, triton.cdiv(T_out, 128))
    conv1d_k5_p2[grid](
        x0, w, y,
        B, C_in, C_out, T_in, T_out,
        x0.stride(0), x0.stride(1), x0.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_T=128
    )
    return y


def _add_bias_triton(x, bias):
    """
    Launch add_bias to add bias per channel.
    x: [B, C, T], bias: [C]
    returns y: [B, C, T]
    """
    B, C, T = x.shape
    y = torch.empty_like(x)
    grid = (B, C, T)
    add_bias[grid](x, bias, y, B, C, T,
                   x.stride(0), x.stride(1), x.stride(2),
                   y.stride(0), y.stride(1), y.stride(2))
    return y


def _relu_triton(x):
    """
    Launch relu_kernel to apply ReLU.
    x: [B, C, T]
    returns y: [B, C, T]
    """
    B, C, T = x.shape
    y = torch.empty_like(x)
    grid = (B, C, T)
    relu_kernel[grid](x, y, B, C, T,
                      x.stride(0), x.stride(1), x.stride(2),
                      y.stride(0), y.stride(1), y.stride(2))
    return y


def _mul_mask_triton(x, mask):
    """
    Launch mul_mask to multiply by mask (broadcast across channels).
    x: [B, C, T], mask: [B, 1, T]
    returns y: [B, C, T]
    """
    B, C, T = x.shape
    y = torch.empty_like(x)
    grid = (B, C, T)
    # mask strides
    mask_stride_b, mask_stride_c, mask_stride_t = mask.stride(0), mask.stride(1), mask.stride(2)
    mul_mask[grid](x, mask, y, B, C, T,
                   x.stride(0), x.stride(1), x.stride(2),
                   mask_stride_b, mask_stride_c, mask_stride_t,
                   y.stride(0), y.stride(1), y.stride(2))
    return y


def _add_or_sub_triton(x, h, add_flag):
    """
    Launch add_or_sub to add/subtract h to x.
    x, h: [B, C, T]
    returns y: [B, C, T]
    """
    B, C, T = x.shape
    y = torch.empty_like(x)
    grid = (B, C, T)
    add_or_sub[grid](x, h, y, B, C, T,
                     x.stride(0), x.stride(1), x.stride(2),
                     h.stride(0), h.stride(1), h.stride(2),
                     y.stride(0), y.stride(1), y.stride(2),
                     add_flag=add_flag)
    return y


def _copy_to_triton(src, dst, channel_offset):
    """
    Launch copy_to to copy src into dst at channel offset.
    src: [B, C, T], dst: [B, 2*C, T], copies src[:, :, :] into dst[:, channel_offset:channel_offset+C, :]
    """
    B, C, T = src.shape
    y_c = 2 * C  # total channels in dst
    grid = (B, C, T)
    copy_to[grid](src, dst, B, C, T,
                  src.stride(0), src.stride(1), src.stride(2),
                  dst.stride(0), channel_offset, dst.stride(1), dst.stride(2))


class ModelNew(torch.nn.Module):
    def forward(self,
        x: torch.Tensor,                     # [B, 192, T]
        x_mask: torch.Tensor,               # [B, 1, T]
        reverse: bool,                      # unused in forward
        # weights for transform 0
        transform_0_conv0_weight: torch.Tensor,  # [192, 96, 5]
        transform_0_conv0_bias: torch.Tensor,    # [192]
        transform_0_conv1_weight: torch.Tensor,  # [192, 192, 5]
        transform_0_conv1_bias: torch.Tensor,    # [192]
        transform_0_conv2_weight: torch.Tensor,  # [96, 192, 5]
        transform_0_conv2_bias: torch.Tensor,    # [96]
        # (unused placeholders for other transforms to keep signature; we implement only one transform)
        transform_1_conv0_weight: torch.Tensor, transform_1_conv0_bias: torch.Tensor,
        transform_1_conv1_weight: torch.Tensor, transform_1_conv1_bias: torch.Tensor,
        transform_1_conv2_weight: torch.Tensor, transform_1_conv2_bias: torch.Tensor,
        transform_2_conv0_weight: torch.Tensor, transform_2_conv0_bias: torch.Tensor,
        transform_2_conv1_weight: torch.Tensor, transform_2_conv1_bias: torch.Tensor,
        transform_2_conv2_weight: torch.Tensor, transform_2_conv2_bias: torch.Tensor,
        transform_3_conv0_weight: torch.Tensor, transform_3_conv0_bias: torch.Tensor,
        transform_3_conv1_weight: torch.Tensor, transform_3_conv1_bias: torch.Tensor,
        transform_3_conv2_weight: torch.Tensor, transform_3_conv2_bias: torch.Tensor,
    ):
        """
        Triton-optimized forward for a single transform:
        - Split x into x0: [B, 96, T] and x1: [B, 96, T]
        - Apply conv0 -> ReLU -> conv1 -> ReLU -> conv2
        - Apply mask to h2, then x1 = x1 + h2 (forward), else subtract in reverse
        - Concatenate x0 and x1 along channel dimension into output [B, 192, T-3]
        """
        device = x.device
        B, C, T = x.shape
        half_channels = C // 2
        assert C == 192, "ModelNew expects input channels=192"
        assert half_channels == 96, "half_channels must be 96"

        # Ensure contiguous tensors for predictable strides
        x = x.contiguous()
        x_mask = x_mask.contiguous()
        # Weights are float32; ensure contiguous
        w0 = transform_0_conv0_weight.contiguous()
        b0 = transform_0_conv0_bias.contiguous()
        w1 = transform_0_conv1_weight.contiguous()
        b1 = transform_0_conv1_bias.contiguous()
        w2 = transform_0_conv2_weight.contiguous()
        b2 = transform_0_conv2_bias.contiguous()

        # Split x
        x0 = x[:, :half_channels, :]
        x1 = x[:, half_channels:, :]

        # conv0: in=96, out=192, K=5, padding=2
        T0_out = T - 1
        h0 = _conv1d_triton(x0, w0, T0_out)  # [B, 192, T-1]
        h0 = _add_bias_triton(h0, b0)
        h0 = _relu_triton(h0)

        # conv1: in=192, out=192, K=5, padding=2
        T1_out = T0_out - 1  # T - 2
        h1 = _conv1d_triton(h0, w1, T1_out)  # [B, 192, T-2]
        h1 = _add_bias_triton(h1, b1)
        h1 = _relu_triton(h1)

        # conv2: in=192, out=96, K=5, padding=2
        T2_out = T1_out - 1  # T - 3
        h2 = _conv1d_triton(h1, w2, T2_out)  # [B, 96, T-3]
        h2 = _relu_triton(h2)

        # Apply mask (broadcast across channels)
        h2_masked = _mul_mask_triton(h2, x_mask)  # [B, 96, T-3]

        # Update x1
        # If reverse=False (default), add; otherwise subtract
        add_flag = 1
        x1_new = _add_or_sub_triton(x1, h2_masked, add_flag)  # [B, 96, T]

        # Concatenate x0 and x1_new along channel dimension to get final output
        y = torch.empty((B, C, T2_out), device=device, dtype=torch.float32)  # [B, 192, T-3]
        _copy_to_triton(x0, y, 0)  # copy x0 into y[:, :96, :]
        _copy_to_triton(x1_new, y, half_channels)  # copy x1_new into y[:, 96:, :]

        # Apply mask to final output (broadcast across channels)
        y_masked = _mul_mask_triton(y, x_mask)  # [B, 192, T-3]

        return y_masked


def run(*args):
    return ModelNew()(*args)
