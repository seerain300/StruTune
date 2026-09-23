# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r35 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_packed_key = None
_packed_weights = None
_packed_sources = None


@triton.jit
def _coupling_initialize(
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
    h_offset = (
        batch_idx * half_channels * time_size
        + channel_idx * time_size
        + time_offsets
    )
    mask_offset = batch_idx * time_size + time_offsets

    mask = tl.load(mask_ptr + mask_offset, mask=time_mask, other=0.0)
    x0 = tl.load(input_ptr + first_offset, mask=time_mask, other=0.0)
    x1 = tl.load(input_ptr + second_offset, mask=time_mask, other=0.0)
    h = tl.load(h_ptr + h_offset, mask=time_mask, other=0.0)

    tl.store(output_ptr + first_offset, x0 * mask, mask=time_mask)
    tl.store(
        output_ptr + second_offset,
        (x1 + SIGN * h * mask) * mask,
        mask=time_mask,
    )


@triton.jit
def _coupling_update_packed_three(
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
        batch_idx * (3 * half_channels) * time_size
        + channel_idx * time_size
        + time_offsets
    )
    transform_stride = half_channels * time_size
    mask_offset = batch_idx * time_size + time_offsets

    x_value = tl.load(x_ptr + x_offset, mask=time_mask, other=0.0)
    h0 = tl.load(h_ptr + h_offset, mask=time_mask, other=0.0)
    h1 = tl.load(
        h_ptr + h_offset + transform_stride,
        mask=time_mask,
        other=0.0,
    )
    h2 = tl.load(
        h_ptr + h_offset + 2 * transform_stride,
        mask=time_mask,
        other=0.0,
    )
    mask = tl.load(mask_ptr + mask_offset, mask=time_mask, other=0.0)

    result = x_value + SIGN * ((h0 + h1) + h2) * mask
    tl.store(x_ptr + x_offset, result, mask=time_mask)


def _transform(
    x0,
    conv0_weight,
    conv0_bias,
    conv1_weight,
    conv1_bias,
    conv2_weight,
    conv2_bias,
):
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
    )
    h = F.relu(h, inplace=True)

    return F.conv1d(
        h,
        conv2_weight,
        conv2_bias,
        padding=padding,
    )


def _packed_transform(x0, packed):
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
        groups=3,
    )
    h = F.relu(h, inplace=True)

    return F.conv1d(
        h,
        conv2_weight,
        conv2_bias,
        padding=padding,
        groups=3,
    )


def _get_packed_weights(transforms):
    global _packed_key
    global _packed_weights
    global _packed_sources

    sources = tuple(tensor for transform in transforms for tensor in transform)
    key = tuple((tensor.data_ptr(), tensor._version) for tensor in sources)

    if key != _packed_key:
        _packed_weights = (
            torch.cat((transforms[0][0], transforms[1][0], transforms[2][0]), dim=0),
            torch.cat((transforms[0][1], transforms[1][1], transforms[2][1]), dim=0),
            torch.cat((transforms[0][2], transforms[1][2], transforms[2][2]), dim=0),
            torch.cat((transforms[0][3], transforms[1][3], transforms[2][3]), dim=0),
            torch.cat((transforms[0][4], transforms[1][4], transforms[2][4]), dim=0),
            torch.cat((transforms[0][5], transforms[1][5], transforms[2][5]), dim=0),
        )
        _packed_key = key
        _packed_sources = sources

    return _packed_weights


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
    block_t = 256
    grid = (
        batch_size * half_channels,
        triton.cdiv(time_size, block_t),
    )

    first = transforms[0]
    h_first = _transform(
        x[:, :half_channels, :],
        first[0],
        first[1],
        first[2],
        first[3],
        first[4],
        first[5],
    )

    output = torch.empty_like(x)
    _coupling_initialize[grid](
        x,
        output,
        h_first,
        x_mask,
        channels=channels,
        half_channels=half_channels,
        time_size=time_size,
        SIGN=sign,
        BLOCK_T=block_t,
    )

    packed = _get_packed_weights(transforms[1:])
    h_packed = _packed_transform(output[:, :half_channels, :], packed)

    _coupling_update_packed_three[grid](
        output,
        h_packed,
        x_mask,
        channels=channels,
        half_channels=half_channels,
        time_size=time_size,
        SIGN=sign,
        BLOCK_T=block_t,
    )

    return output