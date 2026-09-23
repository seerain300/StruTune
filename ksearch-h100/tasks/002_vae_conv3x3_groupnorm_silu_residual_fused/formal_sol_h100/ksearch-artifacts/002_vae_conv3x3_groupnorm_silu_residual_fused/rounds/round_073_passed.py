# solution=GPT-5.6-Sol_002_vae_conv3x3_groupnorm_silu_residual_fused_triton_optimized_r7 score=1.1351965839123288 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_CHANNELS = 256
_NUM_GROUPS = 32
_CHANNELS_PER_GROUP = _CHANNELS // _NUM_GROUPS

_STAGE1_SINGLE_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 64 * 128
_STAGE1_LOW_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 128 * 128
_STAGE1_MEDIUM_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 128 * 128
_STAGE1_BATCH4_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 192 * 128
_STAGE1_HIGH_MEDIUM_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 192 * 128
_STAGE1_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 192 * 128

_STAGE2_SINGLE_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 128 * 128
_STAGE2_LOW_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 192 * 128
_STAGE2_MEDIUM_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 192 * 128
_STAGE2_BATCH4_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 256 * 128
_STAGE2_HIGH_MEDIUM_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 256 * 128
_STAGE2_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 256 * 128

_GROUP_NORM_SMALL_BLOCK = 2048
_GROUP_NORM_LARGE_BLOCK = 4096
_GROUP_NORM_SMALL_SPATIAL_THRESHOLD = 64 * 64
_GROUP_NORM_MEDIUM_SPATIAL_THRESHOLD = 128 * 128
_POINTWISE_BLOCK = 4096


@triton.jit
def _group_norm_silu_kernel(
    input_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    residual_ptr,
    group_elements,
    spatial_elements,
    eps,
    ADD_RESIDUAL: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    ALIGNED_SPATIAL: tl.constexpr,
    BLOCKS_PER_CHANNEL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group_id = tl.program_id(0)
    group_start = group_id * group_elements

    if ALIGNED_SPATIAL:
        value_sum_0 = 0.0
        value_sum_1 = 0.0
        value_square_sum_0 = 0.0
        value_square_sum_1 = 0.0

        for block_start in range(
            0,
            CHANNELS_PER_GROUP * BLOCKS_PER_CHANNEL * BLOCK,
            2 * BLOCK,
        ):
            offsets_0 = block_start + tl.arange(0, BLOCK)
            offsets_1 = offsets_0 + BLOCK

            values_0 = tl.load(
                input_ptr + group_start + offsets_0,
            ).to(tl.float32)
            values_1 = tl.load(
                input_ptr + group_start + offsets_1,
            ).to(tl.float32)

            value_sum_0 += tl.sum(values_0, axis=0)
            value_sum_1 += tl.sum(values_1, axis=0)
            value_square_sum_0 += tl.sum(values_0 * values_0, axis=0)
            value_square_sum_1 += tl.sum(values_1 * values_1, axis=0)

        value_sum = value_sum_0 + value_sum_1
        value_square_sum = value_square_sum_0 + value_square_sum_1
    else:
        value_sum = 0.0
        value_square_sum = 0.0

        for block_start in range(0, group_elements, BLOCK):
            offsets = block_start + tl.arange(0, BLOCK)

            if FULL_BLOCKS:
                values = tl.load(
                    input_ptr + group_start + offsets,
                ).to(tl.float32)
            else:
                mask = offsets < group_elements
                values = tl.load(
                    input_ptr + group_start + offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)

            value_sum += tl.sum(values, axis=0)
            value_square_sum += tl.sum(values * values, axis=0)

    element_count = group_elements.to(tl.float32)
    mean = value_sum / element_count
    variance = tl.maximum(
        value_square_sum / element_count - mean * mean,
        0.0,
    )
    inverse_std = tl.rsqrt(variance + eps)

    group_index = group_id % NUM_GROUPS
    channel_base = group_index * CHANNELS_PER_GROUP

    if ALIGNED_SPATIAL:
        for channel_offset in range(0, CHANNELS_PER_GROUP):
            channel = channel_base + channel_offset
            scale = tl.load(
                norm_weight_ptr + channel,
            ).to(tl.float32)
            bias = tl.load(
                norm_bias_ptr + channel,
            ).to(tl.float32)
            channel_start = group_start + channel_offset * spatial_elements

            for block_index in range(0, BLOCKS_PER_CHANNEL):
                offsets = block_index * BLOCK + tl.arange(0, BLOCK)
                global_offsets = channel_start + offsets

                values = tl.load(
                    input_ptr + global_offsets,
                ).to(tl.float32)

                values = (values - mean) * inverse_std
                values = values * scale + bias
                values = values * tl.sigmoid(values)

                if ADD_RESIDUAL:
                    values += tl.load(
                        residual_ptr + global_offsets,
                    ).to(tl.float32)

                tl.store(input_ptr + global_offsets, values)
    else:
        for block_start in range(0, group_elements, BLOCK):
            offsets = block_start + tl.arange(0, BLOCK)
            global_offsets = group_start + offsets

            if FULL_BLOCKS:
                values = tl.load(
                    input_ptr + global_offsets,
                ).to(tl.float32)
            else:
                mask = offsets < group_elements
                values = tl.load(
                    input_ptr + global_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)

            channels = channel_base + offsets // spatial_elements

            if FULL_BLOCKS:
                scale = tl.load(
                    norm_weight_ptr + channels,
                ).to(tl.float32)
                bias = tl.load(
                    norm_bias_ptr + channels,
                ).to(tl.float32)
            else:
                mask = offsets < group_elements
                scale = tl.load(
                    norm_weight_ptr + channels,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                bias = tl.load(
                    norm_bias_ptr + channels,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)

            values = (values - mean) * inverse_std
            values = values * scale + bias
            values = values * tl.sigmoid(values)

            if ADD_RESIDUAL:
                if FULL_BLOCKS:
                    values += tl.load(
                        residual_ptr + global_offsets,
                    ).to(tl.float32)
                else:
                    mask = offsets < group_elements
                    values += tl.load(
                        residual_ptr + global_offsets,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)

            if FULL_BLOCKS:
                tl.store(input_ptr + global_offsets, values)
            else:
                mask = offsets < group_elements
                tl.store(
                    input_ptr + global_offsets,
                    values,
                    mask=mask,
                )


@triton.jit
def _silu_residual_kernel(
    input_ptr,
    residual_ptr,
    num_elements,
    FULL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    if FULL_BLOCKS:
        values = tl.load(
            input_ptr + offsets,
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
        ).to(tl.float32)
    else:
        mask = offsets < num_elements
        values = tl.load(
            input_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    values = values * tl.sigmoid(values) + residual

    if FULL_BLOCKS:
        tl.store(input_ptr + offsets, values)
    else:
        mask = offsets < num_elements
        tl.store(input_ptr + offsets, values, mask=mask)


def _group_norm_silu(
    input_tensor,
    norm_weight,
    norm_bias,
    eps,
    residual=None,
):
    batch_size, _, height, width = input_tensor.shape
    spatial_elements = height * width
    group_elements = _CHANNELS_PER_GROUP * spatial_elements

    if spatial_elements <= _GROUP_NORM_SMALL_SPATIAL_THRESHOLD:
        block = _GROUP_NORM_SMALL_BLOCK
        num_warps = 4
    elif spatial_elements <= _GROUP_NORM_MEDIUM_SPATIAL_THRESHOLD:
        block = _GROUP_NORM_SMALL_BLOCK
        num_warps = 8
    else:
        block = _GROUP_NORM_LARGE_BLOCK
        num_warps = 8

    aligned_spatial = spatial_elements % block == 0

    _group_norm_silu_kernel[(batch_size * _NUM_GROUPS,)](
        input_tensor,
        norm_weight,
        norm_bias,
        residual if residual is not None else input_tensor,
        group_elements,
        spatial_elements,
        eps,
        ADD_RESIDUAL=residual is not None,
        FULL_BLOCKS=group_elements % block == 0,
        ALIGNED_SPATIAL=aligned_spatial,
        BLOCKS_PER_CHANNEL=(
            spatial_elements // block
            if aligned_spatial
            else 1
        ),
        NUM_GROUPS=_NUM_GROUPS,
        CHANNELS_PER_GROUP=_CHANNELS_PER_GROUP,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return input_tensor


def _silu_residual(input_tensor, residual):
    num_elements = input_tensor.numel()

    _silu_residual_kernel[(triton.cdiv(num_elements, _POINTWISE_BLOCK),)](
        input_tensor,
        residual,
        num_elements,
        FULL_BLOCKS=num_elements % _POINTWISE_BLOCK == 0,
        BLOCK=_POINTWISE_BLOCK,
        num_warps=8,
    )
    return input_tensor


@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    assert x.shape[1] == _CHANNELS

    batch_size, _, height, width = x.shape
    spatial_elements = height * width

    if batch_size == 1:
        stage1_threshold = (
            _STAGE1_SINGLE_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        )
        stage2_threshold = (
            _STAGE2_SINGLE_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        )
    elif batch_size == 2:
        stage1_threshold = _STAGE1_LOW_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        stage2_threshold = _STAGE2_LOW_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
    elif batch_size == 3:
        stage1_threshold = (
            _STAGE1_MEDIUM_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        )
        stage2_threshold = (
            _STAGE2_MEDIUM_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        )
    elif batch_size == 4:
        stage1_threshold = _STAGE1_BATCH4_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        stage2_threshold = _STAGE2_BATCH4_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
    elif batch_size <= 8:
        stage1_threshold = (
            _STAGE1_HIGH_MEDIUM_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        )
        stage2_threshold = (
            _STAGE2_HIGH_MEDIUM_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        )
    else:
        stage1_threshold = _STAGE1_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        stage2_threshold = _STAGE2_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD

    use_native_group_norm_stage1 = spatial_elements >= stage1_threshold
    use_native_group_norm_stage2 = spatial_elements >= stage2_threshold

    output = F.conv2d(
        x,
        conv1_weight,
        bias=None,
        stride=1,
        padding=1,
    )

    if use_native_group_norm_stage1:
        output = F.group_norm(
            output,
            _NUM_GROUPS,
            weight=norm1_weight,
            bias=norm1_bias,
            eps=eps,
        )
        output = F.silu(output, inplace=True)
    else:
        output = _group_norm_silu(
            output,
            norm1_weight,
            norm1_bias,
            eps,
        )

    output = F.conv2d(
        output,
        conv2_weight,
        bias=None,
        stride=1,
        padding=1,
    )

    if use_native_group_norm_stage2:
        output = F.group_norm(
            output,
            _NUM_GROUPS,
            weight=norm2_weight,
            bias=norm2_bias,
            eps=eps,
        )
        output = _silu_residual(output, x)
    else:
        output = _group_norm_silu(
            output,
            norm2_weight,
            norm2_bias,
            eps,
            residual=x,
        )

    return output