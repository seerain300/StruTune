import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv1d_triton_fused_relu(
    x_ptr,         # *const float, input [B, Cin, T]
    w_ptr,         # *const float, weights_flat [Cout, Cin*K]
    b_ptr,         # *const float, bias [Cout]
    y_ptr,         # *float, output [B, Cout, T_out]
    B: tl.int32, Cin: tl.int32, Cout: tl.int32, T: tl.int32, K: tl.int32,
    stride_b: tl.int32, stride_cin: tl.int32, stride_t: tl.int32,
    w_stride_co: tl.int32, w_stride_k: tl.int32,
    pad: tl.int32,
    T_out: tl.int32,
    BLOCK_POS: tl.constexpr,  # tile over output positions
    BLOCK_CO: tl.constexpr,   # tile over output channels
):
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_pos = tl.program_id(2)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)      # [BLOCK_CO]
    pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)  # [BLOCK_POS]

    co_mask = co_offsets < Cout
    pos_mask = pos_offsets < T_out

    acc = tl.zeros((BLOCK_CO, BLOCK_POS), dtype=tl.float32)

    # Loop over input channels and kernel positions
    for ci in range(0, Cin):
        for kk in range(0, K):
            input_pos = pos_offsets - pad + kk  # [BLOCK_POS]
            in_bounds = (input_pos >= 0) & (input_pos < T) & pos_mask

            # Load x[b, ci, input_pos]
            x_vec = tl.load(
                x_ptr + pid_b * stride_b + ci * stride_cin + input_pos * stride_t,
                mask=in_bounds,
                other=0.0
            ).to(tl.float32)  # [BLOCK_POS]

            # Load W_flat[co, ci*K + kk] as vector over BLOCK_CO
            w_vec = tl.load(
                w_ptr + co_offsets * w_stride_co + (ci * K + kk) * w_stride_k,
                mask=co_mask,
                other=0.0
            ).to(tl.float32)  # [BLOCK_CO]

            # Outer product accumulate
            acc += w_vec[:, None] * x_vec[None, :]

    # Add bias
    b_vec = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    acc += b_vec[:, None]

    # Fused ReLU
    acc = tl.maximum(acc, 0.0)

    # Store to y[b, co, pos]
    y_stride_b = 0
    y_stride_co = T_out
    y_stride_pos = 1
    y_base = y_ptr + pid_b * y_stride_b
    tl.store(
        y_base + co_offsets[:, None] * y_stride_co + pos_offsets[None, :] * y_stride_pos,
        acc,
        mask=co_mask[:, None] & pos_mask[None, :]
    )


@triton.jit
def concat_and_add_triton(
    x0_ptr, x1_ptr, h_ptr, yout_ptr,
    B: tl.int32, C0: tl.int32, C1: tl.int32, T: tl.int32,
    stride_x0_b: tl.int32, stride_x0_c: tl.int32, stride_x0_t: tl.int32,
    stride_x1_b: tl.int32, stride_x1_c: tl.int32, stride_x1_t: tl.int32,
    stride_h_b: tl.int32, stride_h_c: tl.int32, stride_h_t: tl.int32,
    stride_y_b: tl.int32, stride_y_c: tl.int32, stride_y_t: tl.int32,
    ADD: tl.constexpr,  # True for add, False for subtract
    BLOCK_C: tl.constexpr,  # tile over channels
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # channel tile
    pid_t = tl.program_id(2)  # time tile

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_C + tl.arange(0, BLOCK_C)

    c_mask = c_offsets < (C0 + C1)
    t_mask = t_offsets < T

    # First, copy x0 to y[:, :C0, :]
    x0_base = x0_ptr + pid_b * stride_x0_b
    y_base = yout_ptr + pid_b * stride_y_b
    for i in range(0, BLOCK_C):
        c = c_offsets[i]
        if c < C0:
            val = tl.load(
                x0_base + c * stride_x0_c + t_offsets[i] * stride_x0_t,
                mask=t_mask[i],
                other=0.0
            ).to(tl.float32)
            tl.store(
                y_base + c * stride_y_c + t_offsets[i] * stride_y_t,
                val,
                mask=t_mask[i]
            )

    # Second, compute x1 + (ADD ? h : -h) and write to y[:, C0:, :]
    x1_base = x1_ptr + pid_b * stride_x1_b
    h_base = h_ptr + pid_b * stride_h_b

    for i in range(0, BLOCK_C):
        c = c_offsets[i]
        if (c >= C0) & (c < (C0 + C1)):
            c_rel = c - C0
            x1_val = tl.load(
                x1_base + c_rel * stride_x1_c + t_offsets[i] * stride_x1_t,
                mask=t_mask[i],
                other=0.0
            ).to(tl.float32)
            h_val = tl.load(
                h_base + c_rel * stride_h_c + t_offsets[i] * stride_h_t,
                mask=t_mask[i],
                other=0.0
            ).to(tl.float32)
            out_val = x1_val + h_val if ADD else x1_val - h_val
            tl.store(
                y_base + c * stride_y_c + t_offsets[i] * stride_y_t,
                out_val,
                mask=t_mask[i]
            )


@triton.jit
def add_h_to_x1_triton(
    x1_ptr, h_ptr, mask_ptr, out_ptr,
    B: tl.int32, C: tl.int32, T: tl.int32,
    stride_x1_b: tl.int32, stride_x1_c: tl.int32, stride_x1_t: tl.int32,
    stride_h_b: tl.int32, stride_h_c: tl.int32, stride_h_t: tl.int32,
    stride_m_b: tl.int32, stride_m_c: tl.int32, stride_m_t: tl.int32,
    stride_out_b: tl.int32, stride_out_c: tl.int32, stride_out_t: tl.int32,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_C + tl.arange(0, BLOCK_C)

    c_mask = c_offsets < C
    t_mask = t_offsets < T

    x1_vals = tl.load(
        x1_ptr + pid_b * stride_x1_b + c_offsets[:, None] * stride_x1_c + t_offsets[None, :] * stride_x1_t,
        mask=c_mask[:, None] & t_mask[None, :],
        other=0.0
    ).to(tl.float32)
    h_vals = tl.load(
        h_ptr + pid_b * stride_h_b + c_offsets[:, None] * stride_h_c + t_offsets[None, :] * stride_h_t,
        mask=c_mask[:, None] & t_mask[None, :],
        other=0.0
    ).to(tl.float32)
    mask_vals = tl.load(
        mask_ptr + pid_b * stride_m_b + 0 * stride_m_c + t_offsets[None, :] * stride_m_t,
        mask=t_mask[None, :],
        other=0.0
    ).to(tl.float32)

    h_masked = h_vals * mask_vals  # broadcast along channels
    out_vals = x1_vals + h_masked
    tl.store(
        out_ptr + pid_b * stride_out_b + c_offsets[:, None] * stride_out_c + t_offsets[None, :] * stride_out_t,
        out_vals,
        mask=c_mask[:, None] & t_mask[None, :]
    )


@triton.jit
def concat_two_triton(
    x0_ptr, x1_ptr, out_ptr,
    B: tl.int32, C0: tl.int32, C1: tl.int32, T: tl.int32,
    stride_x0_b: tl.int32, stride_x0_c: tl.int32, stride_x0_t: tl.int32,
    stride_x1_b: tl.int32, stride_x1_c: tl.int32, stride_x1_t: tl.int32,
    stride_out_b: tl.int32, stride_out_c: tl.int32, stride_out_t: tl.int32,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_C + tl.arange(0, BLOCK_C)

    c_mask = c_offsets < (C0 + C1)
    t_mask = t_offsets < T

    # Write x0 to out[:, :C0, :]
    x0_base = x0_ptr + pid_b * stride_x0_b
    out_base = out_ptr + pid_b * stride_out_b
    for i in range(0, BLOCK_C):
        c = c_offsets[i]
        if c < C0:
            val = tl.load(
                x0_base + c * stride_x0_c + t_offsets[i] * stride_x0_t,
                mask=t_mask[i],
                other=0.0
            ).to(tl.float32)
            tl.store(
                out_base + c * stride_out_c + t_offsets[i] * stride_out_t,
                val,
                mask=t_mask[i]
            )

    # Write x1 to out[:, C0:, :]
    x1_base = x1_ptr + pid_b * stride_x1_b
    for i in range(0, BLOCK_C):
        c = c_offsets[i]
        if (c >= C0) & (c < (C0 + C1)):
            c_rel = c - C0
            val = tl.load(
                x1_base + c_rel * stride_x1_c + t_offsets[i] * stride_x1_t,
                mask=t_mask[i],
                other=0.0
            ).to(tl.float32)
            tl.store(
                out_base + c * stride_out_c + t_offsets[i] * stride_out_t,
                val,
                mask=t_mask[i]
            )


def apply_transform_triton_single(
    x0,               # [B, C0, T]
    conv0_w, conv0_b,  # [C1, C0*K], [C1]
    conv1_w, conv1_b,  # [C2, C1*K], [C2]
    conv2_w, conv2_b,  # [C3, C2*K], [C3]
    B: int, C0: int, C1: int, C2: int, C3: int, T: int, K: int,
    device: torch.device
):
    # Ensure contiguity
    x0 = x0.contiguous()
    conv0_w = conv0_w.contiguous()
    conv0_b = conv0_b.contiguous()
    conv1_w = conv1_w.contiguous()
    conv1_b = conv1_b.contiguous()
    conv2_w = conv2_w.contiguous()
    conv2_b = conv2_b.contiguous()

    # Compute T_out for conv (stride=1, pad=(K-1)//2, dilation=1)
    pad = (K - 1) // 2
    T_out = T - K + 1 + 2 * pad  # equals T for K=5

    # Allocate outputs for convs
    h0 = torch.empty((B, C1, T_out), dtype=torch.float32, device=device)
    h1 = torch.empty((B, C2, T_out), dtype=torch.float32, device=device)
    h2 = torch.empty((B, C3, T_out), dtype=torch.float32, device=device)

    # Launch conv1d + ReLU for conv0
    conv1d_triton_fused_relu[(B, triton.cdiv(C1, 32), triton.cdiv(T_out, 64))](
        x0, conv0_w.view(C1, C0 * K), conv0_b, h0,
        B, C0, C1, T, K,
        x0.stride(0), x0.stride(1), x0.stride(2),
        conv0_w.stride(0), conv0_w.stride(1),
        pad, T_out,
        BLOCK_POS=64, BLOCK_CO=32
    )

    # Launch conv1d + ReLU for conv1
    conv1d_triton_fused_relu[(B, triton.cdiv(C2, 32), triton.cdiv(T_out, 64))](
        h0, conv1_w.view(C2, C1 * K), conv1_b, h1,
        B, C1, C2, T_out, K,
        h0.stride(0), h0.stride(1), h0.stride(2),
        conv1_w.stride(0), conv1_w.stride(1),
        pad, T_out,
        BLOCK_POS=64, BLOCK_CO=32
    )

    # Launch conv1d + ReLU for conv2
    conv1d_triton_fused_relu[(B, triton.cdiv(C3, 32), triton.cdiv(T_out, 64))](
        h1, conv2_w.view(C3, C2 * K), conv2_b, h2,
        B, C2, C3, T_out, K,
        h1.stride(0), h1.stride(1), h1.stride(2),
        conv2_w.stride(0), conv2_w.stride(1),
        pad, T_out,
        BLOCK_POS=64, BLOCK_CO=32
    )

    return h2


def run_triton(
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
    transform_3_conv2_bias: torch.Tensor,
):
    # Ensure tensors are on CUDA
    device = x.device
    if not device.type == 'cuda':
        # If not CUDA, move to CUDA (evaluation environments typically use CUDA)
        device = torch.device('cuda')
        x = x.to(device)
        x_mask = x_mask.to(device)

    B, C, T = x.shape
    half_channels = C // 2
    C0 = half_channels  # channels for x0
    C1 = 192  # conv0 output
    C2 = 192  # conv1 output
    C3 = half_channels  # conv2 output

    # Define transforms as a list of 4 groups
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

    x0 = x[:, :C0, :].contiguous()
    x1 = x[:, C0:, :].contiguous()

    # Choose directions
    add = not reverse

    # Prepare final output tensor
    final_channels = C0 + C3  # x0 channels + final conv2 channels
    final_out = torch.empty((B, final_channels, T), dtype=torch.float32, device=device)

    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (transforms if not reverse else transforms[::-1]):
        # Compute h for this transform
        h2 = apply_transform_triton_single(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b,
                                           B, C0, 192, 192, C3, T, 5, device)

        # Multiply h2 by mask (broadcast along channels)
        h2_masked = h2  # We'll use Triton to add, but mask is applied by elementwise multiply below
        # Launch Triton elementwise add for x1
        updated_x1 = torch.empty_like(x1, dtype=torch.float32, device=device)
        add_h_to_x1_triton[(B, triton.cdiv(C3, 64), triton.cdiv(T, 64))](
            x1, h2, x_mask, updated_x1,
            B, C3, T,
            x1.stride(0), x1.stride(1), x1.stride(2),
            h2.stride(0), h2.stride(1), h2.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            updated_x1.stride(0), updated_x1.stride(1), updated_x1.stride(2),
            BLOCK_C=64
        )
        x1 = updated_x1

    # Concatenate x0 and x1 into final_out using Triton
    concat_two_triton[(B, triton.cdiv(final_channels, 64), triton.cdiv(T, 64))](
        x0, x1, final_out,
        B, C0, x1.shape[1], T,
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        final_out.stride(0), final_out.stride(1), final_out.stride(2),
        BLOCK_C=64
    )

    return final_out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args layout mirrors original: (x, x_mask, reverse, ... 16 weights/biases)
        # We expect x of shape [B, C, T], x_mask of shape [B, 1, T], and then 4 transforms' weights/bias.
        x = args[0].contiguous()
        x_mask = args[1].contiguous()
        reverse = args[2]
        # Parse inputs: 16 remaining are 4 transforms, each with 3 weights and 1 bias
        # We'll reconstruct the 4 tuples.
        # Note: In original code, inputs are returned from get_inputs with many tensors. Here we assume args contain these.
        # To keep signature compatibility, we simply run_triton with args[3:].
        # However, the original run function signature uses named tensors. Since we can't inspect caller, we simply call:
        return run_triton(x, x_mask, reverse, *args[3:])


def run(*args):
    return ModelNew()(*args)
