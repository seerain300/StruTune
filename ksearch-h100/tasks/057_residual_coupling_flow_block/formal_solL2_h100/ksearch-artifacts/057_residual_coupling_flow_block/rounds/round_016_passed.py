# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r16 score=1.0822928221534593 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _coupling_update(
    x_ptr,
    h_ptr,
    mask_ptr,
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    time_size,
    SIGN: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_batch_channel = tl.program_id(0)
    pid_time = tl.program_id(1)

    batch_idx = pid_batch_channel // channels
    channel_idx = pid_batch_channel % channels

    time_offsets = pid_time * BLOCK_T + tl.arange(0, BLOCK_T)
    time_mask = time_offsets < time_size

    x_offset = (
        batch_idx * channels * time_size
        + channel_idx * time_size
        + time_offsets
    )
    mask_offset = batch_idx * time_size + time_offsets

    mask = tl.load(
        mask_ptr + mask_offset,
        mask=time_mask,
        other=0.0,
    )
    x_value = tl.load(
        x_ptr + x_offset,
        mask=time_mask,
        other=0.0,
    )

    is_second_half = channel_idx >= half_channels
    h_channel = channel_idx - half_channels
    h_offset = (
        batch_idx * half_channels * time_size
        + h_channel * time_size
        + time_offsets
    )
    h_value = tl.load(
        h_ptr + h_offset,
        mask=time_mask & is_second_half,
        other=0.0,
    )

    first_half_result = x_value * mask
    second_half_result = (x_value + SIGN * h_value * mask) * mask
    result = tl.where(
        is_second_half,
        second_half_result,
        first_half_result,
    )

    tl.store(
        x_ptr + x_offset,
        result,
        mask=time_mask,
    )


def _apply_layer(
    x,
    x_mask,
    conv0_weight,
    conv0_bias,
    conv1_weight,
    conv1_bias,
    conv2_weight,
    conv2_bias,
    reverse,
):
    half_channels = x.shape[1] // 2
    padding = conv0_weight.shape[2] // 2

    x0 = x[:, :half_channels, :]

    h = F.conv1d(
        x0,
        conv0_weight,
        conv0_bias,
        padding=padding,
    )
    h = F.relu(h)

    h = F.conv1d(
        h,
        conv1_weight,
        conv1_bias,
        padding=padding,
    )
    h = F.relu(h)

    h = F.conv1d(
        h,
        conv2_weight,
        conv2_bias,
        padding=padding,
    )

    batch_size, channels, time_size = x.shape

    grid = (
        batch_size * channels,
        triton.cdiv(time_size, 128),
    )

    _coupling_update[grid](
        x,
        h,
        x_mask,
        channels=channels,
        half_channels=half_channels,
        time_size=time_size,
        SIGN=-1.0 if reverse else 1.0,
        BLOCK_T=128,
    )

    return x


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
            transform_0_conv0_weight,
            transform_0_conv0_bias,
            transform_0_conv1_weight,
            transform_0_conv1_bias,
            transform_0_conv2_weight,
            transform_0_conv2_bias,
        ),
        (
            transform_1_conv0_weight,
            transform_1_conv0_bias,
            transform_1_conv1_weight,
            transform_1_conv1_bias,
            transform_1_conv2_weight,
            transform_1_conv2_bias,
        ),
        (
            transform_2_conv0_weight,
            transform_2_conv0_bias,
            transform_2_conv1_weight,
            transform_2_conv1_bias,
            transform_2_conv2_weight,
            transform_2_conv2_bias,
        ),
        (
            transform_3_conv0_weight,
            transform_3_conv0_bias,
            transform_3_conv1_weight,
            transform_3_conv1_bias,
            transform_3_conv2_weight,
            transform_3_conv2_bias,
        ),
    )

    if reverse:
        transforms = transforms[::-1]

    x = x.clone()

    for (
        conv0_weight,
        conv0_bias,
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
    ) in transforms:
        x = _apply_layer(
            x,
            x_mask,
            conv0_weight,
            conv0_bias,
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            reverse,
        )

    return x