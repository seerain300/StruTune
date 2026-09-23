# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r12 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _conv1d_stage(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    input_channels: tl.constexpr,
    output_channels: tl.constexpr,
    time_size,
    input_batch_stride,
    input_channel_offset,
    kernel_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    APPLY_RELU: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_t = tl.program_id(1)

    batch_idx = pid_bc // output_channels
    output_channel = pid_bc % output_channels

    time_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    time_mask = time_offsets < time_size
    output = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for kernel_idx in range(kernel_size):
        input_time = time_offsets + kernel_idx - kernel_size // 2
        valid_time = (input_time >= 0) & (input_time < time_size)

        for channel_start in range(0, input_channels, BLOCK_C):
            channel_offsets = channel_start + tl.arange(0, BLOCK_C)
            channel_mask = channel_offsets < input_channels
            load_mask = channel_mask[:, None] & valid_time[None, :]

            input_offsets = (
                batch_idx * input_batch_stride
                + (input_channel_offset + channel_offsets[:, None]) * time_size
                + input_time[None, :]
            )
            x_values = tl.load(
                x_ptr + input_offsets,
                mask=load_mask,
                other=0.0,
            )

            weight_offsets = (
                output_channel * input_channels * kernel_size
                + channel_offsets * kernel_size
                + kernel_idx
            )
            weight_values = tl.load(
                w_ptr + weight_offsets,
                mask=channel_mask,
                other=0.0,
            )

            output += tl.sum(
                x_values * weight_values[:, None],
                axis=0,
            )

    output += tl.load(bias_ptr + output_channel)

    if APPLY_RELU:
        output = tl.maximum(output, 0.0)

    output_offsets = (
        batch_idx * output_channels * time_size
        + output_channel * time_size
        + time_offsets
    )
    tl.store(
        out_ptr + output_offsets,
        output,
        mask=time_mask,
    )


@triton.jit
def _coupling_update(
    x_ptr,
    h_ptr,
    mask_ptr,
    out_ptr,
    batch_size,
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    time_size,
    SIGN: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels

    time_offsets = tl.arange(0, BLOCK_T)
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

    if channel_idx < half_channels:
        result = x_value * mask
    else:
        h_offset = (
            batch_idx * half_channels * time_size
            + (channel_idx - half_channels) * time_size
            + time_offsets
        )
        h_value = tl.load(
            h_ptr + h_offset,
            mask=time_mask,
            other=0.0,
        )
        result = (x_value + SIGN * h_value * mask) * mask

    tl.store(
        out_ptr + x_offset,
        result,
        mask=time_mask,
    )


def _run_conv(
    x,
    weight,
    bias,
    output,
    input_channels,
    input_channel_offset=0,
):
    batch_size = x.shape[0]
    time_size = x.shape[2]
    output_channels = weight.shape[0]

    grid = (
        batch_size * output_channels,
        triton.cdiv(time_size, 128),
    )

    _conv1d_stage[grid](
        x,
        weight,
        bias,
        output,
        batch_size,
        input_channels=input_channels,
        output_channels=output_channels,
        time_size=time_size,
        input_batch_stride=x.stride(0),
        input_channel_offset=input_channel_offset,
        kernel_size=weight.shape[2],
        BLOCK_T=128,
        BLOCK_C=32,
        APPLY_RELU=output is not None and output_channels != weight.shape[0],
    )


def _run_conv_stage(
    x,
    weight,
    bias,
    output,
    input_channels,
    input_channel_offset,
    apply_relu,
):
    batch_size = x.shape[0]
    time_size = x.shape[2]
    output_channels = weight.shape[0]

    grid = (
        batch_size * output_channels,
        triton.cdiv(time_size, 128),
    )

    _conv1d_stage[grid](
        x,
        weight,
        bias,
        output,
        batch_size,
        input_channels=input_channels,
        output_channels=output_channels,
        time_size=time_size,
        input_batch_stride=x.stride(0),
        input_channel_offset=input_channel_offset,
        kernel_size=weight.shape[2],
        BLOCK_T=128,
        BLOCK_C=32,
        APPLY_RELU=apply_relu,
    )


def _apply_layer(
    x,
    x_mask,
    h0,
    h1,
    h2,
    conv0_weight,
    conv0_bias,
    conv1_weight,
    conv1_bias,
    conv2_weight,
    conv2_bias,
    reverse,
):
    batch_size, channels, time_size = x.shape
    half_channels = channels // 2

    _run_conv_stage(
        x,
        conv0_weight,
        conv0_bias,
        h0,
        half_channels,
        0,
        True,
    )
    _run_conv_stage(
        h0,
        conv1_weight,
        conv1_bias,
        h1,
        half_channels * 2,
        0,
        True,
    )
    _run_conv_stage(
        h1,
        conv2_weight,
        conv2_bias,
        h2,
        h1.shape[1],
        0,
        False,
    )

    output = torch.empty_like(x)
    grid = (
        batch_size * channels,
        triton.cdiv(time_size, 128),
    )

    _coupling_update[grid](
        x,
        h2,
        x_mask,
        output,
        batch_size,
        channels=channels,
        half_channels=half_channels,
        time_size=time_size,
        SIGN=-1.0 if reverse else 1.0,
        BLOCK_T=128,
    )
    return output


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
    batch_size, channels, time_size = x.shape
    half_channels = channels // 2
    hidden_channels = transform_0_conv0_weight.shape[0]

    transforms = [
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
    ]

    h0 = torch.empty(
        (batch_size, hidden_channels, time_size),
        device=x.device,
        dtype=x.dtype,
    )
    h1 = torch.empty_like(h0)
    h2 = torch.empty(
        (batch_size, half_channels, time_size),
        device=x.device,
        dtype=x.dtype,
    )

    if reverse:
        transforms = reversed(transforms)

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
            h0,
            h1,
            h2,
            conv0_weight,
            conv0_bias,
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            reverse,
        )

    return x