# task: 002_vae_conv3x3_groupnorm_silu_residual_fused
# bench: SOL-L1 | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=20/20 geomean=1.389x
# feedback best (5-workload sample during search): 1.437x
# torch fallback audit: A·核心靠库 (conv2d×2)
# tokens: 2,173,042

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _groupnorm_stats_kernel(
    input_ptr,
    stats_ptr,
    group_size: tl.constexpr,
    num_group_instances,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    group_start = group_id * group_size
    lane_offsets = tl.arange(0, BLOCK_SIZE)

    value_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
    squared_value_sum = tl.zeros((BLOCK_SIZE,), tl.float32)

    for block_start in tl.static_range(0, group_size, BLOCK_SIZE):
        local_offsets = block_start + lane_offsets
        mask = local_offsets < group_size
        values = tl.load(
            input_ptr + group_start + local_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value_sum += values
        squared_value_sum += values * values

    value_sum = tl.sum(value_sum, axis=0)
    squared_value_sum = tl.sum(squared_value_sum, axis=0)

    mean = value_sum / group_size
    variance = squared_value_sum / group_size - mean * mean
    variance = tl.maximum(variance, 0.0)
    rstd = tl.rsqrt(variance + eps)

    tl.store(stats_ptr + group_id, mean)
    tl.store(stats_ptr + num_group_instances + group_id, rstd)


@triton.jit
def _groupnorm_partial_stats_kernel(
    input_ptr,
    partial_ptr,
    group_size: tl.constexpr,
    num_group_instances,
    BLOCK_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    split_id = tl.program_id(1)

    split_start = split_id * SPLIT_SIZE
    split_end = tl.minimum(split_start + SPLIT_SIZE, group_size)
    group_start = group_id * group_size
    lane_offsets = tl.arange(0, BLOCK_SIZE)

    value_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
    squared_value_sum = tl.zeros((BLOCK_SIZE,), tl.float32)

    for block_start in tl.static_range(0, SPLIT_SIZE, BLOCK_SIZE):
        local_offsets = split_start + block_start + lane_offsets
        mask = local_offsets < split_end
        values = tl.load(
            input_ptr + group_start + local_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value_sum += values
        squared_value_sum += values * values

    value_sum = tl.sum(value_sum, axis=0)
    squared_value_sum = tl.sum(squared_value_sum, axis=0)

    partial_offset = group_id * NUM_SPLITS + split_id
    partial_count = num_group_instances * NUM_SPLITS

    tl.store(partial_ptr + partial_offset, value_sum)
    tl.store(
        partial_ptr + partial_count + partial_offset,
        squared_value_sum,
    )


@triton.jit
def _groupnorm_finalize_stats_kernel(
    partial_ptr,
    stats_ptr,
    group_size,
    num_group_instances,
    eps,
    NUM_SPLITS: tl.constexpr,
):
    group_id = tl.program_id(0)
    split_offsets = tl.arange(0, NUM_SPLITS)
    partial_offsets = group_id * NUM_SPLITS + split_offsets
    partial_count = num_group_instances * NUM_SPLITS

    value_sum = tl.sum(
        tl.load(partial_ptr + partial_offsets).to(tl.float32),
        axis=0,
    )
    squared_value_sum = tl.sum(
        tl.load(
            partial_ptr + partial_count + partial_offsets
        ).to(tl.float32),
        axis=0,
    )

    mean = value_sum / group_size
    variance = squared_value_sum / group_size - mean * mean
    variance = tl.maximum(variance, 0.0)
    rstd = tl.rsqrt(variance + eps)

    tl.store(stats_ptr + group_id, mean)
    tl.store(stats_ptr + num_group_instances + group_id, rstd)


@triton.jit
def _groupnorm_silu_residual_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    stats_ptr,
    output_ptr,
    spatial_size: tl.constexpr,
    num_group_instances,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    spatial_block = tl.program_id(0)
    group_id = tl.program_id(1)

    spatial_offsets = (
        spatial_block * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    channel_in_group = tl.arange(0, 8)[:, None]
    mask = spatial_offsets < spatial_size

    channel_group = group_id % 32
    channel_ids = channel_group * 8 + channel_in_group

    group_start = group_id * 8 * spatial_size
    offsets = (
        group_start
        + channel_in_group * spatial_size
        + spatial_offsets
    )

    values = tl.load(
        input_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    mean = tl.load(stats_ptr + group_id)
    rstd = tl.load(stats_ptr + num_group_instances + group_id)
    weight = tl.load(weight_ptr + channel_ids)
    bias = tl.load(bias_ptr + channel_ids)

    normalized = (values - mean) * rstd
    normalized = normalized * weight + bias
    output = normalized * tl.sigmoid(normalized)

    if ADD_RESIDUAL:
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output += residual

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _persistent_groupnorm_silu_residual_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps,
    spatial_size: tl.constexpr,
    group_size: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    local_offsets = tl.arange(0, BLOCK_SIZE)
    mask = local_offsets < group_size

    group_start = group_id * group_size
    offsets = group_start + local_offsets

    values = tl.load(
        input_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    value_sum = tl.sum(values, axis=0)
    squared_value_sum = tl.sum(values * values, axis=0)

    mean = value_sum / group_size
    variance = squared_value_sum / group_size - mean * mean
    variance = tl.maximum(variance, 0.0)
    rstd = tl.rsqrt(variance + eps)

    channel_group = group_id % 32
    channel_in_group = local_offsets // spatial_size
    channel_ids = channel_group * 8 + channel_in_group

    weight = tl.load(
        weight_ptr + channel_ids,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + channel_ids,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    normalized = (values - mean) * rstd
    normalized = normalized * weight + bias
    output = normalized * tl.sigmoid(normalized)

    if ADD_RESIDUAL:
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output += residual

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _looped_groupnorm_silu_residual_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps,
    spatial_size: tl.constexpr,
    group_size: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    group_start = group_id * group_size
    lane_offsets = tl.arange(0, BLOCK_SIZE)

    value_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
    squared_value_sum = tl.zeros((BLOCK_SIZE,), tl.float32)

    for block_start in tl.static_range(0, group_size, BLOCK_SIZE):
        local_offsets = block_start + lane_offsets
        mask = local_offsets < group_size
        values = tl.load(
            input_ptr + group_start + local_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value_sum += values
        squared_value_sum += values * values

    value_sum = tl.sum(value_sum, axis=0)
    squared_value_sum = tl.sum(squared_value_sum, axis=0)

    mean = value_sum / group_size
    variance = squared_value_sum / group_size - mean * mean
    variance = tl.maximum(variance, 0.0)
    rstd = tl.rsqrt(variance + eps)

    channel_group = group_id % 32

    for block_start in tl.static_range(0, group_size, BLOCK_SIZE):
        local_offsets = block_start + lane_offsets
        mask = local_offsets < group_size
        offsets = group_start + local_offsets

        values = tl.load(
            input_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        if BLOCK_SIZE == spatial_size:
            channel_id = (
                channel_group * 8
                + block_start // spatial_size
            )
            weight = tl.load(weight_ptr + channel_id).to(tl.float32)
            bias = tl.load(bias_ptr + channel_id).to(tl.float32)
        else:
            channel_in_group = local_offsets // spatial_size
            channel_ids = channel_group * 8 + channel_in_group
            weight = tl.load(
                weight_ptr + channel_ids,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            bias = tl.load(
                bias_ptr + channel_ids,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

        normalized = (values - mean) * rstd
        normalized = normalized * weight + bias
        output = normalized * tl.sigmoid(normalized)

        if ADD_RESIDUAL:
            residual = tl.load(
                residual_ptr + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            output += residual

        tl.store(output_ptr + offsets, output, mask=mask)


def _groupnorm_silu_inplace(
    out: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    batch_size: int,
    spatial_size: int,
    add_residual: bool,
):
    group_size = 8 * spatial_size
    num_group_instances = batch_size * 32

    if spatial_size <= 256:
        persistent_block_size = triton.next_power_of_2(group_size)

        if persistent_block_size <= 256:
            persistent_num_warps = 1
        elif persistent_block_size <= 512:
            persistent_num_warps = 2
        elif persistent_block_size <= 1024:
            persistent_num_warps = 4
        else:
            persistent_num_warps = 8

        _persistent_groupnorm_silu_residual_kernel[
            (num_group_instances,)
        ](
            out,
            residual,
            weight,
            bias,
            out,
            eps,
            spatial_size=spatial_size,
            group_size=group_size,
            ADD_RESIDUAL=add_residual,
            BLOCK_SIZE=persistent_block_size,
            num_warps=persistent_num_warps,
        )
        return

    if spatial_size <= 4096 and num_group_instances >= 128:
        if spatial_size == 4096 and num_group_instances >= 1024:
            looped_block_size = 16384
            looped_num_warps = 8
        elif spatial_size >= 2048 and num_group_instances >= 256:
            looped_block_size = 2048
            looped_num_warps = 8
        elif spatial_size >= 1024:
            looped_block_size = 1024
            looped_num_warps = 8
        else:
            looped_block_size = 512
            looped_num_warps = 4

        _looped_groupnorm_silu_residual_kernel[
            (num_group_instances,)
        ](
            out,
            residual,
            weight,
            bias,
            out,
            eps,
            spatial_size=spatial_size,
            group_size=group_size,
            ADD_RESIDUAL=add_residual,
            BLOCK_SIZE=looped_block_size,
            num_warps=looped_num_warps,
        )
        return

    stats = torch.empty(
        2 * num_group_instances,
        device=out.device,
        dtype=torch.float32,
    )

    if group_size >= 131072 and num_group_instances < 128:
        if group_size >= 1048576:
            if num_group_instances <= 32:
                num_splits = 16
            elif num_group_instances <= 64:
                num_splits = 8
            else:
                num_splits = 4
        else:
            if num_group_instances <= 32:
                num_splits = 8
            elif num_group_instances <= 64:
                num_splits = 4
            else:
                num_splits = 2

        partial = torch.empty(
            2 * num_group_instances * num_splits,
            device=out.device,
            dtype=torch.float32,
        )

        split_size = triton.cdiv(group_size, num_splits)
        partial_block_size = min(
            1024,
            triton.next_power_of_2(split_size),
        )

        if partial_block_size <= 256:
            partial_num_warps = 1
        elif partial_block_size <= 512:
            partial_num_warps = 2
        else:
            partial_num_warps = 4

        _groupnorm_partial_stats_kernel[
            (num_group_instances, num_splits)
        ](
            out,
            partial,
            group_size,
            num_group_instances,
            BLOCK_SIZE=partial_block_size,
            NUM_SPLITS=num_splits,
            SPLIT_SIZE=split_size,
            num_warps=partial_num_warps,
        )

        _groupnorm_finalize_stats_kernel[
            (num_group_instances,)
        ](
            partial,
            stats,
            group_size,
            num_group_instances,
            eps,
            NUM_SPLITS=num_splits,
            num_warps=1,
        )
    else:
        stats_block_size = min(
            1024,
            triton.next_power_of_2(group_size),
        )

        if stats_block_size <= 256:
            stats_num_warps = 1
        elif stats_block_size <= 512:
            stats_num_warps = 2
        else:
            stats_num_warps = 4

        _groupnorm_stats_kernel[
            (num_group_instances,)
        ](
            out,
            stats,
            group_size,
            num_group_instances,
            eps,
            BLOCK_SIZE=stats_block_size,
            num_warps=stats_num_warps,
        )

    block_size = min(
        256,
        triton.next_power_of_2(spatial_size),
    )

    if block_size <= 32:
        epilogue_num_warps = 1
    elif block_size <= 128:
        epilogue_num_warps = 4
    else:
        epilogue_num_warps = 8

    _groupnorm_silu_residual_kernel[
        (
            triton.cdiv(spatial_size, block_size),
            num_group_instances,
        )
    ](
        out,
        residual,
        weight,
        bias,
        stats,
        out,
        spatial_size,
        num_group_instances,
        ADD_RESIDUAL=add_residual,
        BLOCK_SIZE=block_size,
        num_warps=epilogue_num_warps,
    )


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
    batch_size, channels, height, width = x.shape
    if channels != 256:
        raise ValueError(
            "This residual block requires exactly 256 channels"
        )

    spatial_size = height * width

    out = F.conv2d(
        x,
        conv1_weight,
        bias=None,
        stride=1,
        padding=1,
    )

    _groupnorm_silu_inplace(
        out,
        out,
        norm1_weight,
        norm1_bias,
        eps,
        batch_size,
        spatial_size,
        False,
    )

    out = F.conv2d(
        out,
        conv2_weight,
        bias=None,
        stride=1,
        padding=1,
    )

    _groupnorm_silu_inplace(
        out,
        x,
        norm2_weight,
        norm2_bias,
        eps,
        batch_size,
        spatial_size,
        True,
    )

    return out