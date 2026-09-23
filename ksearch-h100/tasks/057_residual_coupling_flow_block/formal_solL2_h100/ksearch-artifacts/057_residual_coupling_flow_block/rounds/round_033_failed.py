# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r33 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_PACKED_TRANSFORMS = {}


@triton.jit
def _coupling_apply_four(
    input_ptr,
    output_ptr,
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

    first_offset = (
        batch_idx * channels * time_size
        + channel_idx * time_size
        + time_offsets
    )
    second_offset = first_offset + half_channels * time_size

    h_base = (
        batch_idx * (4 * half_channels) * time_size
        + channel_idx * time_size
        + time_offsets
    )
    h_stride = half_channels * time_size
    mask_offset = batch_idx * time_size + time_offsets

    mask = tl.load(
        mask_ptr + mask_offset,
        mask=time_mask,
        other=0.0,
    )
    x0 = tl.load(
        input_ptr + first_offset,
        mask=time_mask,
        other=0.0,
    )
    x1 = tl.load(
        input_ptr + second_offset,
        mask=time_mask,
        other=0.0,
    )

    h0 = tl.load(
        h_ptr + h_base,
        mask=time_mask,
        other=0.0,
    )
    h1 = tl.load(
        h_ptr + h_base + h_stride,
        mask=time_mask,
        other=0.0,
    )
    h2 = tl.load(
        h_ptr + h_base + 2 * h_stride,
        mask=time_mask,
        other=0.0,
    )
    h3 = tl.load(
        h_ptr + h_base + 3 * h_stride,
        mask=time_mask,
        other=0.0,
    )

    h_sum = (h0 + h1) + (h2 + h3)

    tl.store(
        output_ptr + first_offset,
        x0 * mask,
        mask=time_mask,
    )
    tl.store(
        output_ptr + second_offset,
        (x1 + SIGN * h_sum * mask) * mask,
        mask=time_mask,
    )


def _get_packed_transforms(transforms):
    tensors = tuple(tensor for transform in transforms for tensor in transform)
    key = tuple(tensor.data_ptr() for tensor in tensors)

    cached = _PACKED_TRANSFORMS.get(key)
    if cached is not None:
        return cached[0]

    packed = (
        torch.cat([transform[0] for transform in transforms], dim=0),
        torch.cat([transform[1] for transform in transforms], dim=0),
        torch.cat([transform[2] for transform in transforms], dim=0),
        torch.cat([transform[3] for transform in transforms], dim=0),
        torch.cat([transform[4] for transform in transforms], dim=0),
        torch.cat([transform[5] for transform in transforms], dim=0),
    )
    _PACKED_TRANSFORMS[key] = (packed, tensors)
    return packed


def _transform_four(x0, packed):
    conv0_weight, conv0_bias, conv1_weight, conv1_bias, conv2_weight, conv2_bias = packed
    padding = conv0_weight.shape[2] // 2

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
        groups=4,
    )
    h = F.relu(h, inplace=True)

    return F.conv1d(
        h,
        conv2_weight,
        conv2_bias,
        padding=padding,
        groups=4,
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

    batch_size, channels, time_size = x.shape
    half_channels = channels // 2
    sign = -1.0 if reverse else 1.0

    packed = _get_packed_transforms(transforms)
    h = _transform_four(x[:, :half_channels, :], packed)

    output = torch.empty_like(x)
    grid = (
        batch_size * half_channels,
        triton.cdiv(time_size, 256),
    )
    _coupling_apply_four[grid](
        x,
        output,
        h,
        x_mask,
        channels=channels,
        half_channels=half_channels,
        time_size=time_size,
        SIGN=sign,
        BLOCK_T=256,
    )

    return output