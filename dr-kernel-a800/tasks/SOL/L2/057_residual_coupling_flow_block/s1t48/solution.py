import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def mask_apply_triton(x_ptr, mask_ptr, out_ptr, N, C, T,
                       x_sN, x_sC, x_sT,
                       m_sN, m_sC, m_sT,
                       out_sN, out_sC, out_sT,
                       BLOCK: tl.constexpr):
    """
    Elementwise: out[n, c, t] = x[n, c, t] * mask[n, 0, t]
    x: [N, C, T], mask: [N, 1, T], out: [N, C, T]
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)

    t_idx = pid_t_block * BLOCK + tl.arange(0, BLOCK)
    valid = t_idx < T

    # Load mask[n, 0, t_idx] (broadcast along channels)
    m_offsets = pid_n * m_sN + 0 * m_sC + t_idx * m_sT
    mask_vals = tl.load(mask_ptr + m_offsets, mask=valid, other=1.0)

    # Load x[n, c, t_idx]
    x_offsets = pid_n * x_sN + pid_c * x_sC + t_idx * x_sT
    x_vals = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)

    # Multiply
    out_vals = x_vals * mask_vals

    # Store to out[n, c, t_idx]
    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid)


@triton.jit
def relu_triton(x_ptr, out_ptr, N, C, T,
                x_sN, x_sC, x_sT,
                out_sN, out_sC, out_sT,
                BLOCK: tl.constexpr):
    """
    Elementwise ReLU: out[n, c, t] = max(x[n, c, t], 0)
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)

    t_idx = pid_t_block * BLOCK + tl.arange(0, BLOCK)
    valid = t_idx < T

    x_offsets = pid_n * x_sN + pid_c * x_sC + t_idx * x_sT
    x_vals = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)

    out_vals = tl.maximum(x_vals, 0.0)

    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid)


@triton.jit
def add_h_triton(x1_ptr, h_ptr, out_ptr, N, C, T,
                 x1_sN, x1_sC, x1_sT,
                 h_sN, h_sC, h_sT,
                 out_sN, out_sC, out_sT,
                 is_add: tl.constexpr,
                 BLOCK: tl.constexpr):
    """
    Elementwise update: out[n, c, t] = x1[n, c, t] + is_add * h[n, c, t]
    For subtract in reverse: is_add = 0 -> out = x1 - h
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)

    t_idx = pid_t_block * BLOCK + tl.arange(0, BLOCK)
    valid = t_idx < T

    x1_offsets = pid_n * x1_sN + pid_c * x1_sC + t_idx * x1_sT
    h_offsets = pid_n * h_sN + pid_c * h_sC + t_idx * h_sT

    x1_vals = tl.load(x1_ptr + x1_offsets, mask=valid, other=0.0)
    h_vals = tl.load(h_ptr + h_offsets, mask=valid, other=0.0)

    out_vals = x1_vals + h_vals * is_add

    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid)


@triton.jit
def final_mask_scale_triton(x_ptr, mask_ptr, out_ptr, N, C, T,
                            x_sN, x_sC, x_sT,
                            m_sN, m_sC, m_sT,
                            out_sN, out_sC, out_sT,
                            BLOCK: tl.constexpr):
    """
    Elementwise: out[n, c, t] = x[n, c, t] * mask[n, 0, t]
    x: [N, C, T], mask: [N, 1, T], out: [N, C, T]
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)

    t_idx = pid_t_block * BLOCK + tl.arange(0, BLOCK)
    valid = t_idx < T

    m_offsets = pid_n * m_sN + 0 * m_sC + t_idx * m_sT
    mask_vals = tl.load(mask_ptr + m_offsets, mask=valid, other=1.0)

    x_offsets = pid_n * x_sN + pid_c * x_sC + t_idx * x_sT
    x_vals = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)

    out_vals = x_vals * mask_vals

    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid)


def _process_one_transform(x, x_mask, w0, b0, w1, b1, w2, b2, reverse: bool):
    """
    Process one transform using torch conv1d and Triton pointwise ops.
    x: [N, 192, T], split into x0 (first 96 channels) and x1 (last 96 channels).
    """
    N, C, T = x.shape
    assert C == 192
    half_channels = C // 2

    # Current x slices
    x0 = x[:, :half_channels, :].contiguous()  # [N, 96, T]
    x1 = x[:, half_channels:, :].contiguous() # [N, 96, T]

    # conv0: [N, 96, T] -> [N, 192, T]
    y0 = torch.nn.functional.conv1d(x0, w0, b0, padding=2)  # K=5, pad=2
    # mask and ReLU
    y0_masked = torch.empty_like(y0)
    mask_apply_triton[(N, 192, triton.cdiv(T, 256))](
        y0, x_mask, y0_masked, N, 192, T,
        y0.stride(0), y0.stride(1), y0.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
        BLOCK=256, num_warps=4
    )
    y0_relu = torch.empty_like(y0_masked)
    relu_triton[(N, 192, triton.cdiv(T, 256))](
        y0_masked, y0_relu, N, 192, T,
        y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
        y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
        BLOCK=256, num_warps=4
    )

    # conv1: [N, 192, T] -> [N, 192, T]
    y1 = torch.nn.functional.conv1d(y0_relu, w1, b1, padding=2)
    # mask and ReLU
    y1_masked = torch.empty_like(y1)
    mask_apply_triton[(N, 192, triton.cdiv(T, 256))](
        y1, x_mask, y1_masked, N, 192, T,
        y1.stride(0), y1.stride(1), y1.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
        BLOCK=256, num_warps=4
    )
    y1_relu = torch.empty_like(y1_masked)
    relu_triton[(N, 192, triton.cdiv(T, 256))](
        y1_masked, y1_relu, N, 192, T,
        y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
        y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
        BLOCK=256, num_warps=4
    )

    # conv2: [N, 192, T] -> [N, 96, T]
    h = torch.nn.functional.conv1d(y1_relu, w2, b2, padding=2)  # output channels = 96

    # mask h
    h_masked = torch.empty_like(h)
    mask_apply_triton[(N, 96, triton.cdiv(T, 256))](
        h, x_mask, h_masked, N, 96, T,
        h.stride(0), h.stride(1), h.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
        BLOCK=256, num_warps=4
    )

    # coupling update on x1
    out_x1 = torch.empty_like(x1)
    is_add = 1 if not reverse else 0  # 1 => add, 0 => subtract
    add_h_triton[(N, 96, triton.cdiv(T, 256))](
        x1, h_masked, out_x1, N, 96, T,
        x1.stride(0), x1.stride(1), x1.stride(2),
        h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
        out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
        is_add=is_add, BLOCK=256, num_warps=4
    )

    # Build final x: concatenate [x0, out_x1] -> [N, 192, T]
    final_x = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
    final_x[:, :96, :] = x0
    final_x[:, 96:, :] = out_x1

    # Finally, scale the entire output by x_mask (broadcast along channels)
    final_x_scaled = torch.empty_like(final_x)
    final_mask_scale_triton[(N, 192, triton.cdiv(T, 256))](
        final_x, x_mask, final_x_scaled, N, 192, T,
        final_x.stride(0), final_x.stride(1), final_x.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        final_x_scaled.stride(0), final_x_scaled.stride(1), final_x_scaled.stride(2),
        BLOCK=256, num_warps=4
    )

    return final_x_scaled


class ModelNew(nn.Module):
    def forward(self, *args):
        """
        Triton-enhanced forward: convs via torch, pointwise ops via Triton.
        """
        x = args[0]  # [N, 192, T]
        x_mask = args[1]  # [N, 1, T]
        reverse = args[2]
        # Remaining args: 4 transforms, each with 3 weights and 3 biases
        assert len(args) == 3 + 4 * 6, "Expected 13 arguments: x, x_mask, reverse, 12 weights/biases"
        # Extract weights for the first transform
        w0_0, b0_0, w1_0, b1_0, w2_0, b2_0 = args[3], args[4], args[5], args[6], args[7], args[8]
        # Build a list of tuples of weights/biases for remaining transforms
        transforms_list = []
        for i in range(9, len(args), 6):
            w0 = args[i]
            b0 = args[i + 1]
            w1 = args[i + 2]
            b1 = args[i + 3]
            w2 = args[i + 4]
            b2 = args[i + 5]
            transforms_list.append((w0, b0, w1, b1, w2, b2))

        N, C, T = x.shape
        assert C == 192, "Expected channels=192"

        if not reverse:
            # Process 4 transforms in forward order
            x = _process_one_transform(x, x_mask, w0_0, b0_0, w1_0, b1_0, w2_0, b2_0, reverse)
            for w0, b0, w1, b1, w2, b2 in transforms_list:
                x = _process_one_transform(x, x_mask, w0, b0, w1, b1, w2, b2, reverse)
        else:
            # For reverse, use existing _process_one_transform with given transforms in reverse order
            rev_transforms = [(w0_0, b0_0, w1_0, b1_0, w2_0, b2_0)] + list(reversed(transforms_list))
            for _ in range(4):
                x = _process_one_transform(x, x_mask, *rev_transforms.pop(0), reverse=True)

        return x


def run(*args):
    return ModelNew()(*args)
