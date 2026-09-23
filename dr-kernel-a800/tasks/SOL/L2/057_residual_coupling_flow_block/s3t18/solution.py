import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_pad0_stride1_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Cout, L_in, L_out,
    x_stride_n, x_stride_c, x_stride_l,
    w_stride_co, w_stride_ci, w_stride_k,
    y_stride_n, y_stride_co, y_stride_l,
    num_warps: tl.constexpr
):
    """
    Conv1d: y[n, co, t_out] = sum_{ci=0..Cin-1} sum_{k=0..4} w[co, ci, k] * x[n, ci, t_out - k]
    padding=0, stride=1, kernel_size=5, bias=True
    Output length L_out = L_in - 4.
    """
    pid_nco = tl.program_id(0)  # over N*Cout
    pid_ltile = tl.program_id(1)  # tiles over L_out
    co = pid_nco % Cout
    n = pid_nco // Cout
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask_l = offs_l < L_out

    # accumulator in float32
    acc = tl.zeros([128], dtype=tl.float32)

    # loop over input channels and kernel taps
    for ci in range(0, Cin):
        for k in range(0, 5):
            t_in = offs_l - k  # valid when 0 <= t_in < L_in
            valid = (t_in >= 0) & (t_in < L_in) & mask_l
            # load x[n, ci, t_in]
            x_idx = n * x_stride_n + ci * x_stride_c + t_in * x_stride_l
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            # load w[co, ci, k] scalar
            w_idx = co * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_val = tl.load(w_ptr + w_idx)  # scalar, default dtype of w
            # multiply and accumulate
            acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + co)
    acc = acc + b_val

    # store y[n, co, offs_l]
    y_idx = n * y_stride_n + co * y_stride_co + offs_l * y_stride_l
    tl.store(y_ptr + y_idx, acc, mask=mask_l)


@triton.jit
def relu_triton(x_ptr, y_ptr, N, C, L, x_stride_n, x_stride_c, x_stride_l, num_warps: tl.constexpr):
    """
    Elementwise ReLU: y = max(x, 0)
    """
    pid_nc = tl.program_id(0)  # N*C
    pid_ltile = tl.program_id(1)
    c = pid_nc % C
    n = pid_nc // C
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L
    x_idx = n * x_stride_n + c * x_stride_c + offs_l * x_stride_l
    y_idx = n * y_stride_n + c * y_stride_c + offs_l * y_stride_l  # y has same shape/strides as x
    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    y_val = tl.maximum(x_val, 0.0)
    tl.store(y_ptr + y_idx, y_val, mask=mask)


@triton.jit
def multiply_mask_triton(x_ptr, mask_ptr, y_ptr, N, C, L,
                          x_stride_n, x_stride_c, x_stride_l,
                          mask_stride_n, mask_stride_l,  # mask has shape [N, 1, L]
                          y_stride_n, y_stride_c, y_stride_l,
                          num_warps: tl.constexpr):
    """
    Elementwise: y = x * mask, mask shape [N, 1, L], broadcast over channels.
    """
    pid_nc = tl.program_id(0)  # N*C
    pid_ltile = tl.program_id(1)
    c = pid_nc % C
    n = pid_nc // C
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L
    x_idx = n * x_stride_n + c * x_stride_c + offs_l * x_stride_l
    # mask index ignores channel (size 1); only use n and l
    m_idx = n * mask_stride_n + offs_l * mask_stride_l
    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    m_val = tl.load(mask_ptr + m_idx, mask=mask, other=1.0)  # mask is 0/1, but ensure 1.0 default
    y_idx = n * y_stride_n + c * y_stride_c + offs_l * y_stride_l
    y_val = x_val * m_val
    tl.store(y_ptr + y_idx, y_val, mask=mask)


@triton.jit
def add_masked_triton(x_ptr, h_ptr, y_ptr, N, C, L,
                      x_stride_n, x_stride_c, x_stride_l,
                      h_stride_n, h_stride_c, h_stride_l,
                      y_stride_n, y_stride_c, y_stride_l,
                      add_positive: tl.constexpr,  # True => y=x+h, False => y=x-h
                      num_warps: tl.constexpr):
    """
    Elementwise add/subtract: y = x + h if add_positive else y = x - h
    """
    pid_nc = tl.program_id(0)  # N*C
    pid_ltile = tl.program_id(1)
    c = pid_nc % C
    n = pid_nc // C
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L
    x_idx = n * x_stride_n + c * x_stride_c + offs_l * x_stride_l
    h_idx = n * h_stride_n + c * h_stride_c + offs_l * h_stride_l
    y_idx = n * y_stride_n + c * y_stride_c + offs_l * y_stride_l
    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    h_val = tl.load(h_ptr + h_idx, mask=mask, other=0.0)
    y_val = x_val + h_val if add_positive else x_val - h_val
    tl.store(y_ptr + y_idx, y_val, mask=mask)


@triton.jit
def concatenate_channels_triton(x0_ptr, x1_ptr, y_ptr, N, C_half, L,
                                x0_stride_n, x0_stride_c, x0_stride_l,
                                x1_stride_n, x1_stride_c, x1_stride_l,
                                y_stride_n, y_stride_c, y_stride_l,
                                num_warps: tl.constexpr):
    """
    Concatenate along channel: y[n, c, l] = x0[n, c, l] for c < C_half; else y[n, c, l] = x1[n, c-C_half, l]
    """
    pid_nc = tl.program_id(0)  # N * (2*C_half)
    pid_ltile = tl.program_id(1)  # tiles of L
    c = pid_nc % (2 * C_half)
    n = pid_nc // (2 * C_half)
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L

    if c < C_half:
        src_ptr = x0_ptr
        src_c = c
        y_c = c
    else:
        src_ptr = x1_ptr
        src_c = c - C_half
        y_c = c

    src_idx = n * src_ptr.stride(0) + src_c * src_ptr.stride(1) + offs_l * src_ptr.stride(2)
    y_idx = n * y_stride_n + y_c * y_stride_c + offs_l * y_stride_l
    val = tl.load(src_ptr + src_idx, mask=mask, other=0.0)
    tl.store(y_ptr + y_idx, val, mask=mask)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool,
        # 4 transforms, each with 3 weights/biases
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
        transform_3_conv2_bias: torch.Tensor,
    ):
        """
        Triton-optimized forward: applies 4 transforms sequentially.
        Each transform:
          - Split x into x0 = x[:, :C/2, :], x1 = x[:, C/2:, :].
          - Compute conv0 -> ReLU -> conv1 -> ReLU -> conv2 in Triton.
          - h = h * x_mask
          - x1 = x1 + h if reverse == False else x1 = x1 - h
          - x = concat([x0, x1], dim=1) and x = x * x_mask
        """
        N, C, L = x.shape
        half_channels = C // 2
        x = x.contiguous()
        x_mask = x_mask.contiguous()  # [N, 1, L], ensure contiguous

        transforms = [
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
        ]

        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split
            x0 = x[:, :half_channels, :].contiguous()
            x1 = x[:, half_channels:, :].contiguous()

            # conv0: [Cin=96, Cout=192, K=5], bias=True, padding=0 => L_out = L - 4
            Cin0 = x0.shape[1]
            Cout0 = conv0_w.shape[0]
            L_in0 = x0.shape[2]
            L_out0 = L_in0 - 4
            h0 = torch.empty((N, Cout0, L_out0), device=x.device, dtype=torch.float32)
            grid0 = (N * Cout0, _ceil_div(L_out0, 128))
            conv1d_pad0_stride1_kernel[grid0](
                x0, conv0_w, conv0_b, h0,
                N, Cin0, Cout0, L_in0, L_out0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                num_warps=4
            )

            # ReLU
            h0 = torch.empty_like(h0)
            grid_relu0 = (N * Cout0, _ceil_div(L_out0, 128))
            relu_triton[grid_relu0](
                h0, h0, N, Cout0, L_out0,
                h0.stride(0), h0.stride(1), h0.stride(2),
                num_warps=4
            )

            # conv1: [Cin=Cout0=192, Cout=192, K=5], bias=True
            Cin1 = h0.shape[1]
            Cout1 = conv1_w.shape[0]
            L_in1 = h0.shape[2]
            L_out1 = L_in1 - 4
            h1 = torch.empty((N, Cout1, L_out1), device=x.device, dtype=torch.float32)
            grid1 = (N * Cout1, _ceil_div(L_out1, 128))
            conv1d_pad0_stride1_kernel[grid1](
                h0, conv1_w, conv1_b, h1,
                N, Cin1, Cout1, L_in1, L_out1,
                h0.stride(0), h0.stride(1), h0.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                num_warps=4
            )

            # ReLU
            h1 = torch.empty_like(h1)
            grid_relu1 = (N * Cout1, _ceil_div(L_out1, 128))
            relu_triton[grid_relu1](
                h1, h1, N, Cout1, L_out1,
                h1.stride(0), h1.stride(1), h1.stride(2),
                num_warps=4
            )

            # conv2: [Cin=Cout1=192, Cout=96, K=5], bias=False (bias argument unused in kernel)
            Cin2 = h1.shape[1]
            Cout2 = conv2_w.shape[0]
            L_in2 = h1.shape[2]
            L_out2 = L_in2 - 4
            h2 = torch.empty((N, Cout2, L_out2), device=x.device, dtype=torch.float32)
            grid2 = (N * Cout2, _ceil_div(L_out2, 128))
            # Pass a dummy bias pointer (not used)
            dummy_bias = torch.empty(1, device=x.device, dtype=torch.float32)
            conv1d_pad0_stride1_kernel[grid2](
                h1, conv2_w, dummy_bias, h2,
                N, Cin2, Cout2, L_in2, L_out2,
                h1.stride(0), h1.stride(1), h1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                num_warps=4
            )

            # Multiply by mask (broadcast over channels)
            h2_masked = torch.empty_like(h2)
            grid_mask = (N * Cout2, _ceil_div(L_out2, 128))
            multiply_mask_triton[grid_mask](
                h2, x_mask, h2_masked, N, Cout2, L_out2,
                h2.stride(0), h2.stride(1), h2.stride(2),
                x_mask.stride(0), x_mask.stride(2),
                h2_masked.stride(0), h2_masked.stride(1), h2_masked.stride(2),
                num_warps=4
            )

            # Affine coupling on x1: x1 = x1 + h2_masked (forward) or x1 = x1 - h2_masked (reverse)
            if not reverse:
                x1 = x1 + h2_masked
            else:
                x1 = x1 - h2_masked

            # Concatenate [x0, x1] along channels
            x_concat = torch.empty((N, 2 * half_channels, L), device=x.device, dtype=x1.dtype)
            grid_concat = (N * (2 * half_channels), _ceil_div(L, 128))
            concatenate_channels_triton[grid_concat](
                x0, x1, x_concat,
                N, half_channels, L,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                x_concat.stride(0), x_concat.stride(1), x_concat.stride(2),
                num_warps=4
            )

            # Multiply output by mask again
            x = x_concat * x_mask

        return x


def run(*args):
    return ModelNew()(*args)
