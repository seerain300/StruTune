# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r2 score=-1.0 passed=False
I’m keeping the established convolution path and targeting the residual/mask launch, which is the only part of the current implementation that can be changed without disturbing convolution numerics. The main adjustment is a channel/time tiled Triton kernel so mask values are reused across channel lanes and flattened index arithmetic is reduced.import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _residual_mask_kernel(
    x_ptr,
    h_ptr,
    mask_ptr,
    output_ptr,
    time,
    sign: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    batch = tl.program_id(2)

    time_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    channel_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    valid_t = time_offsets < time
    valid_c = channel_offsets < 192
    valid = valid_c[:, None] & valid_t[None, :]

    x_offsets = (
        batch * (192 * time)
        + channel_offsets[:, None] * time
        + time_offsets[None, :]
    )

    x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)
    mask = tl.load(
        mask_ptr + batch * time + time_offsets,
        mask=valid_t,
        other=0.0,
    )

    second_half = channel_offsets >= 96
    h_channel = tl.where(second_half, channel_offsets - 96, 0)
    h_offsets = (
        batch * (96 * time)
        + h_channel[:, None] * time
        + time_offsets[None, :]
    )
    h = tl.load(
        h_ptr + h_offsets,
        mask=valid & second_half[:, None],
        other=0.0,
    )

    result = tl.where(
        second_half[:, None],
        (x + sign * h * mask[None, :]) * mask[None, :],
        x * mask[None, :],
    )

    tl.store(output_ptr + x_offsets, result, mask=valid)


def _apply_transform(
    x,
    x_mask,
    output,
    weights,
    biases,
    sign,
):
    h = F.conv1d(x[:, :96, :], weights[0], biases[0], padding=2)
    h.relu_()
    h = F.conv1d(h, weights[1], biases[1], padding=2)
    h.relu_()
    h = F.conv1d(h, weights[2], biases[2], padding=2)

    time = x.shape[2]
    grid = (
        triton.cdiv(time, 128),
        triton.cdiv(192, 32),
        x.shape[0],
    )

    _residual_mask_kernel[grid](
        x,
        h,
        x_mask,
        output,
        time,
        sign=sign,
        BLOCK_C=32,
        BLOCK_T=128,
        num_warps=8,
    )


@torch.no_grad()
def run(
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
    transforms = (
        (
            (
                transform_0_conv0_weight,
                transform_0_conv1_weight,
                transform_0_conv2_weight,
            ),
            (
                transform_0_conv0_bias,
                transform_0_conv1_bias,
                transform_0_conv2_bias,
            ),
        ),
        (
            (
                transform_1_conv0_weight,
                transform_1_conv1_weight,
                transform_1_conv2_weight,
            ),
            (
                transform_1_conv0_bias,
                transform_1_conv1_bias,
                transform_1_conv2_bias,
            ),
        ),
        (
            (
                transform_2_conv0_weight,
                transform_2_conv1_weight,
                transform_2_conv2_weight,
            ),
            (
                transform_2_conv0_bias,
                transform_2_conv1_bias,
                transform_2_conv2_bias,
            ),
        ),
        (
            (
                transform_3_conv0_weight,
                transform_3_conv1_weight,
                transform_3_conv2_weight,
            ),
            (
                transform_3_conv0_bias,
                transform_3_conv1_bias,
                transform_3_conv2_bias,
            ),
        ),
    )

    if reverse:
        transforms = transforms[::-1]
        sign = -1.0
    else:
        sign = 1.0

    output0 = torch.empty_like(x)
    output1 = torch.empty_like(x)

    current = x
    for layer, (weights, biases) in enumerate(transforms):
        output = output0 if layer % 2 == 0 else output1
        _apply_transform(
            current,
            x_mask,
            output,
            weights,
            biases,
            sign,
        )
        current = output

    return current