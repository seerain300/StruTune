# solution=GPT-5.6-Sol_002_vae_conv3x3_groupnorm_silu_residual_fused_triton_optimized_r28 score=1.1348844323436371 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_CHANNELS = 256
_NUM_GROUPS = 32
_CHANNELS_PER_GROUP = _CHANNELS // _NUM_GROUPS
_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 128 * 128
_LOW_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD = 96 * 128
_GROUP_NORM_BLOCK = 4096


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

        for block_start in range(0, group_elements, 2 * BLOCK):
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
        for block_start in range(0, group_elements, 2 * BLOCK):
            offsets_0 = block_start + tl.arange(0, BLOCK)
            offsets_1 = offsets_0 + BLOCK
            global_offsets_0 = group_start + offsets_0
            global_offsets_1 = group_start + offsets_1

            values_0 = tl.load(
                input_ptr + global_offsets_0,
            ).to(tl.float32)
            values_1 = tl.load(
                input_ptr + global_offsets_1,
            ).to(tl.float32)

            block_index_0 = block_start // BLOCK
            block_index_1 = block_index_0 + 1
            channel_0 = (
                channel_base + block_index_0 // BLOCKS_PER_CHANNEL
            )
            channel_1 = (
                channel_base + block_index_1 // BLOCKS_PER_CHANNEL
            )

            scale_0 = tl.load(
                norm_weight_ptr + channel_0,
            ).to(tl.float32)
            bias_0 = tl.load(
                norm_bias_ptr + channel_0,
            ).to(tl.float32)
            scale_1 = tl.load(
                norm_weight_ptr + channel_1,
            ).to(tl.float32)
            bias_1 = tl.load(
                norm_bias_ptr + channel_1,
            ).to(tl.float32)

            values_0 = (values_0 - mean) * inverse_std
            values_0 = values_0 * scale_0 + bias_0
            values_0 = values_0 * tl.sigmoid(values_0)

            values_1 = (values_1 - mean) * inverse_std
            values_1 = values_1 * scale_1 + bias_1
            values_1 = values_1 * tl.sigmoid(values_1)

            if ADD_RESIDUAL:
                values_0 += tl.load(
                    residual_ptr + global_offsets_0,
                ).to(tl.float32)
                values_1 += tl.load(
                    residual_ptr + global_offsets_1,
                ).to(tl.float32)

            tl.store(input_ptr + global_offsets_0, values_0)
            tl.store(input_ptr + global_offsets_1, values_1)
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
    aligned_spatial = spatial_elements % _GROUP_NORM_BLOCK == 0

    _group_norm_silu_kernel[(batch_size * _NUM_GROUPS,)](
        input_tensor,
        norm_weight,
        norm_bias,
        residual if residual is not None else input_tensor,
        group_elements,
        spatial_elements,
        eps,
        ADD_RESIDUAL=residual is not None,
        FULL_BLOCKS=group_elements % _GROUP_NORM_BLOCK == 0,
        ALIGNED_SPATIAL=aligned_spatial,
        BLOCKS_PER_CHANNEL=(
            spatial_elements // _GROUP_NORM_BLOCK
            if aligned_spatial
            else 1
        ),
        NUM_GROUPS=_NUM_GROUPS,
        CHANNELS_PER_GROUP=_CHANNELS_PER_GROUP,
        BLOCK=_GROUP_NORM_BLOCK,
        num_warps=8,
    )
    return input_tensor


def _silu_residual(input_tensor, residual):
    num_elements = input_tensor.numel()
    block = 2048

    _silu_residual_kernel[(triton.cdiv(num_elements, block),)](
        input_tensor,
        residual,
        num_elements,
        FULL_BLOCKS=num_elements % block == 0,
        BLOCK=block,
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
    use_native_group_norm = (
        spatial_elements >= _NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        or (
            batch_size <= 2
            and spatial_elements
            >= _LOW_BATCH_NATIVE_GROUP_NORM_SPATIAL_THRESHOLD
        )
    )

    output = F.conv2d(
        x,
        conv1_weight,
        bias=None,
        stride=1,
        padding=1,
    )

    if use_native_group_norm:
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

    if use_native_group_norm:
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