import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_k5_p2(x_ptr, w_ptr, y_ptr,
                 B, Ci, Co, T_in, T_out,
                 x_stride_b, x_stride_c, x_stride_t,
                 w_stride_co, w_stride_ci, w_stride_k,
                 y_stride_b, y_stride_c, y_stride_t,
                 BLOCK_T: tl.constexpr):
    # Each program handles one (b, co) and a tile of T_out
    pid0 = tl.program_id(0)  # over B*Co
    pid1 = tl.program_id(1)  # over tiles of T_out
    co = pid0 % Co
    b = pid0 // Co

    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T_out

    # Accumulator for this (b, co, tile)
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps (static bounds)
    for ci in tl.static_range(0, Ci):
        for k in tl.static_range(0, 5):
            # valid conv: t_in = t_offsets + 2 - k
            t_in = t_offsets + (2 - k)
            in_mask = (t_in >= 0) & (t_in < T_in)
            # Load x[b, ci, t_in] for this tile
            x_offset = b * x_stride_b + ci * x_stride_c + t_in * x_stride_t
            x_val = tl.load(x_ptr + x_offset, mask=in_mask & t_mask, other=0.0)
            # Load w[co, ci, k]
            w_offset = co * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_val = tl.load(w_ptr + w_offset)
            # Accumulate
            acc += x_val * w_val

    # Store y[b, co, t_offsets]
    y_offset = b * y_stride_b + co * y_stride_c + t_offsets * y_stride_t
    tl.store(y_ptr + y_offset, acc, mask=t_mask)


@triton.jit
def add_bias(y_ptr, bias_ptr, B, C, T, stride_b, stride_c, stride_t, BLOCK_T: tl.constexpr):
    # Add bias per output channel to y of shape [B, C, T]
    pid0 = tl.program_id(0)  # over B*C
    pid1 = tl.program_id(1)  # over tiles of T
    co = pid0 % C
    b = pid0 // C
    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T
    y_offset = b * stride_b + co * stride_c + t_offsets * stride_t
    y_val = tl.load(y_ptr + y_offset, mask=t_mask, other=0.0)
    bias_val = tl.load(bias_ptr + co)  # scalar bias
    y_val = y_val + bias_val
    tl.store(y_ptr + y_offset, y_val, mask=t_mask)


@triton.jit
def relu(y_ptr, B, C, T, stride_b, stride_c, stride_t, BLOCK_T: tl.constexpr):
    # Elementwise ReLU on y of shape [B, C, T]
    pid0 = tl.program_id(0)  # over B*C
    pid1 = tl.program_id(1)  # over tiles of T
    co = pid0 % C
    b = pid0 // C
    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T
    y_offset = b * stride_b + co * stride_c + t_offsets * stride_t
    y_val = tl.load(y_ptr + y_offset, mask=t_mask, other=0.0)
    y_val = tl.maximum(y_val, 0.0)
    tl.store(y_ptr + y_offset, y_val, mask=t_mask)


@triton.jit
def mul_mask(y_ptr, mask_ptr, B, C, T, stride_b, stride_c, stride_t,
             mask_stride_b, mask_stride_c, mask_stride_t, BLOCK_T: tl.constexpr):
    # Multiply y by mask; mask has shape [B, 1, T], broadcast across C
    pid0 = tl.program_id(0)  # over B*C
    pid1 = tl.program_id(1)  # over tiles of T
    co = pid0 % C
    b = pid0 // C
    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T
    y_offset = b * stride_b + co * stride_c + t_offsets * stride_t
    y_val = tl.load(y_ptr + y_offset, mask=t_mask, other=0.0)
    # mask[b, 0, t]
    mask_offset = b * mask_stride_b + 0 * mask_stride_c + t_offsets * mask_stride_t
    mask_val = tl.load(mask_ptr + mask_offset, mask=t_mask, other=1.0)
    y_val = y_val * mask_val
    tl.store(y_ptr + y_offset, y_val, mask=t_mask)


@triton.jit
def add_or_sub(y_ptr, h_ptr, B, C, T, stride_b, stride_c, stride_t, add_flag: tl.constexpr, BLOCK_T: tl.constexpr):
    # y_ptr: input x1 (second half), h_ptr: transform output h (shape [B, C, T])
    pid0 = tl.program_id(0)  # over B*C
    pid1 = tl.program_id(1)  # over tiles of T
    co = pid0 % C
    b = pid0 // C
    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T
    y_offset = b * stride_b + co * stride_c + t_offsets * stride_t
    h_offset = b * stride_b + co * stride_c + t_offsets * stride_t  # same time T
    y_val = tl.load(y_ptr + y_offset, mask=t_mask, other=0.0)
    h_val = tl.load(h_ptr + h_offset, mask=t_mask, other=0.0)
    if add_flag:
        y_val = y_val + h_val
    else:
        y_val = y_val - h_val
    tl.store(y_ptr + y_offset, y_val, mask=t_mask)


@triton.jit
def copy_to(src_ptr, dst_ptr, B, C, T, src_stride_b, src_stride_c, src_stride_t, dst_stride_b, dst_stride_c, dst_stride_t, BLOCK_T: tl.constexpr):
    # copy src (shape [B, C, T]) into dst (same shape) along a tile of T
    pid0 = tl.program_id(0)  # over B*C
    pid1 = tl.program_id(1)  # over tiles of T
    co = pid0 % C
    b = pid0 // C
    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T
    src_offset = b * src_stride_b + co * src_stride_c + t_offsets * src_stride_t
    dst_offset = b * dst_stride_b + co * dst_stride_c + t_offsets * dst_stride_t
    val = tl.load(src_ptr + src_offset, mask=t_mask, other=0.0)
    tl.store(dst_ptr + dst_offset, val, mask=t_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                x_mask: torch.Tensor,
                reverse: bool,
                transform_0_conv0_weight: torch.Tensor,
                transform_0_conv0_bias: torch.Tensor,
                transform_0_conv1_weight: torch.Tensor,
                transform_0_conv1_bias: torch.Tensor,
                transform_0_conv2_weight: torch.Tensor,
                transform_0_conv2_bias: torch.Tensor,
                transform_1_conv0_weight: torch.Tensor,
                transform_1_conv0_bias: torch.Tensor,
                transform_1_conv1_weight: torch.Tensor,
                transform_1_conv1_bias: torch.Tensor,
                transform_1_conv2_weight: torch.Tensor,
                transform_1_conv2_bias: torch.Tensor,
                transform_2_conv0_weight: torch.Tensor,
                transform_2_conv0_bias: torch.Tensor,
                transform_2_conv1_weight: torch.Tensor,
                transform_2_conv1_bias: torch.Tensor,
                transform_2_conv2_weight: torch.Tensor,
                transform_2_conv2_bias: torch.Tensor,
                transform_3_conv0_weight: torch.Tensor,
                transform_3_conv0_bias: torch.Tensor,
                transform_3_conv1_weight: torch.Tensor,
                transform_3_conv1_bias: torch.Tensor,
                transform_3_conv2_weight: torch.Tensor,
                transform_3_conv2_bias: torch.Tensor):
        """
        Triton-only forward. Implements the original logic with Triton kernels.
        All tensors are assumed to be on CUDA.
        """
        assert x.is_cuda and x.dtype == torch.float32, "x must be CUDA float32"
        assert x_mask.is_cuda and x_mask.dtype == torch.float32, "x_mask must be CUDA float32"

        B = x.shape[0]
        C_total = x.shape[1]
        T = x.shape[2]
        half = C_total // 2  # 96
        hidden = 192
        K = 5
        pad = 2

        # Prepare output for final concatenation: y_out [B, 192, T]
        # We will fill y_out[:, :half, :] with x0 and y_out[:, half:, :] with x1_after all transforms.
        y_out = torch.empty((B, C_total, T), device=x.device, dtype=x.dtype)

        # Helper to launch Triton conv for a given (x0/x1, weight, bias), return h (channels half or hidden)
        def triton_conv(x_half, weights, bias, out_channels, expected_time_out):
            Bx = x_half.shape[0]
            Ci = x_half.shape[1]
            T_in = x_half.shape[2]
            T_out = expected_time_out
            # Allocate y
            y = torch.empty((Bx, out_channels, T_out), device=x.device, dtype=torch.float32)
            # Launch grid
            grid = (Bx * out_channels, triton.cdiv(T_out, 128))
            conv1d_k5_p2[grid](
                x_half, weights, y,
                Bx, Ci, out_channels, T_in, T_out,
                x_half.stride(0), x_half.stride(1), x_half.stride(2),
                weights.stride(0), weights.stride(1), weights.stride(2),
                y.stride(0), y.stride(1), y.stride(2),
                BLOCK_T=128,
            )
            # Add bias (per output channel)
            bias_vec = bias.to(torch.float32).contiguous()
            grid_bias = (Bx * out_channels, triton.cdiv(T_out, 128))
            add_bias[grid_bias](y, bias_vec, Bx, out_channels, T_out, y.stride(0), y.stride(1), y.stride(2), BLOCK_T=128)
            # ReLU
            grid_relu = (Bx * out_channels, triton.cdiv(T_out, 128))
            relu[grid_relu](y, Bx, out_channels, T_out, y.stride(0), y.stride(1), y.stride(2), BLOCK_T=128)
            return y

        # Transform pipeline: apply sequentially
        for (
            conv0_w, conv0_b,
            conv1_w, conv1_b,
            conv2_w, conv2_b
        ) in [
            (transform_0_conv0_weight, transform_0_conv0_bias,
             transform_0_conv1_weight, transform_0_conv1_bias,
             transform_0_conv2_weight, transform_0_conv2_bias),
            (transform_1_conv0_weight, transform_1_conv0_bias,
             transform_1_conv1_weight, transform_1_conv1_bias,
             transform_1_conv2_weight, transform_1_conv2_bias),
            (transform_2_conv0_weight, transform_2_conv0_bias,
             transform_2_conv1_weight, transform_2_conv1_bias,
             transform_2_conv2_weight, transform_2_conv2_bias),
            (transform_3_conv0_weight, transform_3_conv0_bias,
             transform_3_conv1_weight, transform_3_conv1_bias,
             transform_3_conv2_weight, transform_3_conv2_bias),
        ]:
            # x0, x1 before transform
            x0 = x[:, :half, :]
            x1 = x[:, half:, :]

            # conv0: in Ci=half, out Co=hidden, T0_out = T - 1
            h0 = triton_conv(x0, conv0_w, conv0_b, hidden, T - 1)
            # conv1: in Ci=hidden, out Co=hidden, T1_out = T - 2
            h1 = triton_conv(h0, conv1_w, conv1_b, hidden, T - 2)
            # conv2: in Ci=hidden, out Co=half, T2_out = T - 3
            h2 = triton_conv(h1, conv2_w, conv2_b, half, T - 3)

            # Apply mask to h2 (broadcast mask across channels)
            mask3d = x_mask.unsqueeze(1).expand(B, half, T)  # [B, half, T]
            mask3d = mask3d.to(torch.float32).contiguous()
            grid_mask = (B * half, triton.cdiv(T - 3, 128))
            h2_masked = torch.empty_like(h2)
            mul_mask[grid_mask](h2, mask3d, B, half, T - 3, h2.stride(0), h2.stride(1), h2.stride(2),
                                mask3d.stride(0), mask3d.stride(1), mask3d.stride(2), BLOCK_T=128)

            # Affine coupling: update x1
            # We need to launch add_or_sub on x1 and h2_masked
            grid_add = (B * half, triton.cdiv(T - 3, 128))
            x1 = x1.contiguous()
            h2_masked = h2_masked.contiguous()
            add_or_sub[grid_add](x1, h2_masked, B, half, T - 3, x1.stride(0), x1.stride(1), x1.stride(2),
                                 add_flag=True, BLOCK_T=128)

            # Concatenate into y_out: first half channels are x0, second half are x1
            grid_copy0 = (B * half, triton.cdiv(T, 128))
            grid_copy1 = (B * half, triton.cdiv(T, 128))
            # Copy x0 into y_out[:, :half, :]
            copy_to[grid_copy0](x0, y_out, B, half, T, x0.stride(0), x0.stride(1), x0.stride(2),
                                y_out.stride(0), y_out.stride(1), y_out.stride(2), BLOCK_T=128)
            # Copy x1 into y_out[:, half:, :]
            copy_to[grid_copy1](x1, y_out, B, half, T, x1.stride(0), x1.stride(1), x1.stride(2),
                                y_out.stride(0), y_out.stride(1), y_out.stride(2) + half * y_out.stride(1), BLOCK_T=128)
            # Update x for next transform
            x = y_out

        return y_out


def run(*args):
    return ModelNew()(*args)
