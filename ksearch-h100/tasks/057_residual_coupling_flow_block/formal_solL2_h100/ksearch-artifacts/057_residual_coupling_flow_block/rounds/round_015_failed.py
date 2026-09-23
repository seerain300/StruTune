# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r15 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_MERGED_CACHE = None


@triton.jit
def _coupling_update_merged(
    x_ptr,
    h_first_ptr,
    h_rest_ptr,
    mask_ptr,
    out_ptr,
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

    first_offset = (
        batch_idx * half_channels * time_size
        + h_channel * time_size
        + time_offsets
    )
    rest_batch_base = batch_idx * (3 * half_channels) * time_size
    rest_offset_0 = (
        rest_batch_base
        + h_channel * time_size
        + time_offsets
    )
    rest_offset_1 = rest_offset_0 + half_channels * time_size
    rest_offset_2 = rest_offset_1 + half_channels * time_size

    load_mask = time_mask & is_second_half
    h_first = tl.load(
        h_first_ptr + first_offset,
        mask=load_mask,
        other=0.0,
    )
    h_rest_0 = tl.load(
        h_rest_ptr + rest_offset_0,
        mask=load_mask,
        other=0.0,
    )
    h_rest_1 = tl.load(
        h_rest_ptr + rest_offset_1,
        mask=load_mask,
        other=0.0,
    )
    h_rest_2 = tl.load(
        h_rest_ptr + rest_offset_2,
        mask=load_mask,
        other=0.0,
    )

    h_sum = (h_first + h_rest_0) + (h_rest_1 + h_rest_2)
    first_half_result = x_value * mask
    second_half_result = (x_value + SIGN * h_sum) * mask
    result = tl.where(
        is_second_half,
        second_half_result,
        first_half_result,
    )

    tl.store(
        out_ptr + x_offset,
        result,
        mask=time_mask,
    )


def _apply_transform(
    x0,
    conv0_weight,
    conv0_bias,
    conv1_weight,
    conv1_bias,
    conv2_weight,
    conv2_bias,
    groups=1,
):
    h = F.conv1d(
        x0,
        conv0_weight,
        conv0_bias,
        padding=2,
        groups=groups,
    )
    h.relu_()

    h = F.conv1d(
        h,
        conv1_weight,
        conv1_bias,
        padding=2,
        groups=groups,
    )
    h.relu_()

    return F.conv1d(
        h,
        conv2_weight,
        conv2_bias,
        padding=2,
        groups=groups,
    )


def _get_merged_transform(transforms, first_index):
    global _MERGED_CACHE

    remaining_indices = tuple(i for i in range(4) if i != first_index)
    sources = tuple(
        tensor
        for i in remaining_indices
        for tensor in transforms[i]
    )
    key = (
        first_index,
        tuple((tensor.data_ptr(), tensor._version) for tensor in sources),
    )

    if _MERGED_CACHE is not None and _MERGED_CACHE[0] == key:
        return _MERGED_CACHE[1]

    remaining = [transforms[i] for i in remaining_indices]
    merged = (
        torch.cat([transform[0] for transform in remaining], dim=0),
        torch.cat([transform[1] for transform in remaining], dim=0),
        torch.cat([transform[2] for transform in remaining], dim=0),
        torch.cat([transform[3] for transform in remaining], dim=0),
        torch.cat([transform[4] for transform in remaining], dim=0),
        torch.cat([transform[5] for transform in remaining], dim=0),
    )
    _MERGED_CACHE = (key, merged, sources)
    return merged


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

    first_index = 3 if reverse else 0
    first = transforms[first_index]
    merged = _get_merged_transform(transforms, first_index)

    half_channels = 96
    x0 = x[:, :half_channels, :]

    h_first = _apply_transform(
        x0,
        first[0],
        first[1],
        first[2],
        first[3],
        first[4],
        first[5],
    )

    masked_x0 = x0 * x_mask
    h_rest = _apply_transform(
        masked_x0,
        merged[0],
        merged[1],
        merged[2],
        merged[3],
        merged[4],
        merged[5],
        groups=3,
    )

    batch_size, channels, time_size = x.shape
    output = torch.empty_like(x)

    grid = (
        batch_size * channels,
        triton.cdiv(time_size, 128),
    )
    _coupling_update_merged[grid](
        x,
        h_first,
        h_rest,
        x_mask,
        output,
        channels=channels,
        half_channels=half_channels,
        time_size=time_size,
        SIGN=-1.0 if reverse else 1.0,
        BLOCK_T=128,
    )

    return output