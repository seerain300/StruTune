# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r18 score=-1.0 passed=False
I’m making a narrowly scoped launch/update optimization: specialize the first-layer paired update and the later second-half-only update so Triton does not carry the general branch and channel mapping through every elementwise launch. The convolution sequence and numerical operation order remain unchanged.import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _coupling_update_first(
    x_ptr,
    h_ptr,
    mask_ptr,
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    time_size,
    SIGN: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_time = tl.program_id(1)

    batch_idx = pid // half_channels
    channel_idx = pid % half_channels

    time_offsets = pid_time * BLOCK_T + tl.arange(0, BLOCK_T)
    time_mask = time_offsets < time_size

    base_x = (
        batch_idx * channels * time_size
        + channel_idx * time_size
        + time_offsets
    )
    second_x = base_x + half_channels * time_size
    h_offset = (
        batch_idx * half_channels * time_size
        + channel_idx * time_size
        + time_offsets
    )
    mask_offset = batch_idx * time_size + time_offsets

    mask = tl.load(
        mask_ptr + mask_offset,
        mask=time_mask,
        other=0.0,
    )
    x0 = tl.load(
        x_ptr + base_x,
        mask=time_mask,
        other=0.0,
    )
    x1 = tl.load(
        x_ptr + second_x,
        mask=time_mask,
        other=0.0,
    )
    h = tl.load(
        h_ptr + h_offset,
        mask=time_mask,
        other=0.0,
    )

    tl.store(
        x_ptr + base_x,
        x0 * mask,
        mask=time_mask,
    )
    tl.store(
        x_ptr + second_x,
        (x1 + SIGN * h * mask) * mask,
        mask=time_mask,
    )


@triton.jit
def _coupling_update_second(
    x_ptr,
    h_ptr,
    mask_ptr,
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    time_size,
    SIGN: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_time = tl.program_id(1)

    batch_idx = pid // half_channels
    channel_idx = pid % half_channels

    time_offsets = pid_time * BLOCK_T + tl.arange(0, BLOCK_T)
    time_mask = time_offsets < time_size

    x_offset = (
        batch_idx * channels * time_size
        + (channel_idx + half_channels) * time_size
        + time_offsets
    )
    h_offset = (
        batch_idx * half_channels * time_size
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
    h_value = tl.load(
        h_ptr + h_offset,
        mask=time_mask,
        other=0.0,
    )

    result = (x_value + SIGN * h_value * mask) * mask

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
    update_first_half,
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
    h = F.relu(h, inplace=True)

    h = F.conv1d(
        h,
        conv1_weight,
        conv1_bias,
        padding=padding,
    )
    h = F.relu(h, inplace=True)

    h = F.conv1d(
        h,
        conv2_weight,
        conv2_bias,
        padding=padding,
    )

    batch_size, channels, time_size = x.shape
    grid = (
        batch_size * half_channels,
        triton.cdiv(time_size, 256),
    )
    sign = -1.0 if reverse else 1.0

    if update_first_half:
        _coupling_update_first[grid](
            x,
            h,
            x_mask,
            channels=channels,
            half_channels=half_channels,
            time_size=time_size,
            SIGN=sign,
            BLOCK_T=256,
        )
    else:
        _coupling_update_second[grid](
            x,
            h,
            x_mask,
            channels=channels,
            half_channels=half_channels,
            time_size=time_size,
            SIGN=sign,
            BLOCK_T=256,
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

    for layer_idx, (
        conv0_weight,
        conv0_bias,
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
    ) in enumerate(transforms):
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
            layer_idx == 0,
        )

    return x