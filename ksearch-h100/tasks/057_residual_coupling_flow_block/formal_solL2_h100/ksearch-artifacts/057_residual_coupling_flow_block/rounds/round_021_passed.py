# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r21 score=1.111046020840633 passed=True
import torch
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
    FIRST_LAYER: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    batch = tl.program_id(0)
    channel_block = tl.program_id(1)
    time_block = tl.program_id(2)

    local_channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    if FIRST_LAYER:
        channels = local_channels
        channel_valid = local_channels < 192
    else:
        channels = local_channels + 96
        channel_valid = local_channels < 96

    time_offsets = time_block * BLOCK_T + tl.arange(0, BLOCK_T)
    time_valid = time_offsets < time
    valid = channel_valid[:, None] & time_valid[None, :]

    x_offsets = (
        batch * 192 * time
        + channels[:, None] * time
        + time_offsets[None, :]
    )
    x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)

    mask = tl.load(
        mask_ptr + batch * time + time_offsets,
        mask=time_valid,
        other=0.0,
    )

    second_half = channels[:, None] >= 96
    h_channels = channels - 96
    h_offsets = (
        batch * 96 * time
        + h_channels[:, None] * time
        + time_offsets[None, :]
    )
    h = tl.load(
        h_ptr + h_offsets,
        mask=valid & second_half,
        other=0.0,
    )

    masked_x = x * mask[None, :]
    result = tl.where(
        second_half,
        masked_x + sign * h * mask[None, :],
        masked_x,
    )

    tl.store(output_ptr + x_offsets, result, mask=valid)


def _apply_transform(
    x,
    x_mask,
    output,
    weights,
    biases,
    sign,
    first_layer,
):
    h = F.conv1d(x[:, :96, :], weights[0], biases[0], padding=2)
    h = F.relu(h)
    h = F.conv1d(h, weights[1], biases[1], padding=2)
    h = F.relu(h)
    h = F.conv1d(h, weights[2], biases[2], padding=2)

    block_c = 32
    block_t = 128
    processed_channels = 192 if first_layer else 96
    grid = (
        x.shape[0],
        triton.cdiv(processed_channels, block_c),
        triton.cdiv(x.shape[2], block_t),
    )

    _residual_mask_kernel[grid](
        x,
        h,
        x_mask,
        output,
        x.shape[2],
        sign=sign,
        FIRST_LAYER=first_layer,
        BLOCK_C=block_c,
        BLOCK_T=block_t,
        num_warps=4,
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

    output = torch.empty_like(x)
    current = x

    for layer, (weights, biases) in enumerate(transforms):
        first_layer = layer == 0
        destination = output if first_layer else current
        _apply_transform(
            current,
            x_mask,
            destination,
            weights,
            biases,
            sign,
            first_layer,
        )
        current = destination

    return current