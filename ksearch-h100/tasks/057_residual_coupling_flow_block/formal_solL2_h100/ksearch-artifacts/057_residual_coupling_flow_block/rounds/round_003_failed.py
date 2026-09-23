# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r3 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _conv1d_stage_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    time,
    input_batch_stride,
    output_batch_stride,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    time_block = tl.program_id(1)
    channel_block = tl.program_id(2)

    time_offsets = time_block * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = channel_block * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    reduction_size: tl.constexpr = CIN * 5

    for reduction_start in range(0, reduction_size, BLOCK_K):
        reduction_offsets = reduction_start + tl.arange(0, BLOCK_K)
        input_channels = reduction_offsets // 5
        kernel_offsets = reduction_offsets % 5
        input_times = time_offsets[None, :] + kernel_offsets[:, None] - 2

        input_addresses = (
            batch_idx * input_batch_stride
            + input_channels[:, None] * time
            + input_times
        )
        input_mask = (
            (reduction_offsets[:, None] < reduction_size)
            & (input_times >= 0)
            & (input_times < time)
        )
        input_values = tl.load(
            input_ptr + input_addresses,
            mask=input_mask,
            other=0.0,
        )

        weight_addresses = (
            output_channels[:, None] * reduction_size
            + reduction_offsets[None, :]
        )
        weight_mask = (
            (output_channels[:, None] < COUT)
            & (reduction_offsets[None, :] < reduction_size)
        )
        weights = tl.load(
            weight_ptr + weight_addresses,
            mask=weight_mask,
            other=0.0,
        )

        accumulator += tl.dot(
            weights,
            input_values,
            input_precision="tf32",
        )

    bias = tl.load(
        bias_ptr + output_channels,
        mask=output_channels < COUT,
        other=0.0,
    )
    result = accumulator + bias[:, None]

    if APPLY_RELU:
        result = tl.maximum(result, 0.0)

    output_addresses = (
        batch_idx * output_batch_stride
        + output_channels[:, None] * time
        + time_offsets[None, :]
    )
    output_mask = (
        (output_channels[:, None] < COUT)
        & (time_offsets[None, :] < time)
    )
    tl.store(output_ptr + output_addresses, result, mask=output_mask)


@triton.jit
def _conv1d_residual_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    state_ptr,
    original_x_ptr,
    mask_ptr,
    output_ptr,
    time,
    input_batch_stride,
    SIGN: tl.constexpr,
    FIRST_LAYER: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    time_block = tl.program_id(1)
    channel_block = tl.program_id(2)

    time_offsets = time_block * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = channel_block * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    reduction_size: tl.constexpr = CIN * 5

    for reduction_start in range(0, reduction_size, BLOCK_K):
        reduction_offsets = reduction_start + tl.arange(0, BLOCK_K)
        input_channels = reduction_offsets // 5
        kernel_offsets = reduction_offsets % 5
        input_times = time_offsets[None, :] + kernel_offsets[:, None] - 2

        input_addresses = (
            batch_idx * input_batch_stride
            + input_channels[:, None] * time
            + input_times
        )
        input_mask = (
            (reduction_offsets[:, None] < reduction_size)
            & (input_times >= 0)
            & (input_times < time)
        )
        input_values = tl.load(
            input_ptr + input_addresses,
            mask=input_mask,
            other=0.0,
        )

        weight_addresses = (
            output_channels[:, None] * reduction_size
            + reduction_offsets[None, :]
        )
        weight_mask = (
            (output_channels[:, None] < COUT)
            & (reduction_offsets[None, :] < reduction_size)
        )
        weights = tl.load(
            weight_ptr + weight_addresses,
            mask=weight_mask,
            other=0.0,
        )

        accumulator += tl.dot(
            weights,
            input_values,
            input_precision="tf32",
        )

    bias = tl.load(
        bias_ptr + output_channels,
        mask=output_channels < COUT,
        other=0.0,
    )
    transform = accumulator + bias[:, None]

    element_mask = (
        (output_channels[:, None] < COUT)
        & (time_offsets[None, :] < time)
    )
    mask_values = tl.load(
        mask_ptr + batch_idx * time + time_offsets,
        mask=time_offsets < time,
        other=0.0,
    )

    state_addresses = (
        batch_idx * 192 * time
        + (96 + output_channels[:, None]) * time
        + time_offsets[None, :]
    )
    state = tl.load(
        state_ptr + state_addresses,
        mask=element_mask,
        other=0.0,
    )
    updated = (
        state + SIGN * transform * mask_values[None, :]
    ) * mask_values[None, :]
    tl.store(output_ptr + state_addresses, updated, mask=element_mask)

    if FIRST_LAYER:
        first_half_addresses = (
            batch_idx * 192 * time
            + output_channels[:, None] * time
            + time_offsets[None, :]
        )
        first_half = tl.load(
            original_x_ptr + first_half_addresses,
            mask=element_mask,
            other=0.0,
        )
        first_half *= mask_values[None, :]
        tl.store(
            output_ptr + first_half_addresses,
            first_half,
            mask=element_mask,
        )


def _launch_conv_stage(
    input_tensor,
    weight,
    bias,
    output_tensor,
    time,
    cin,
    cout,
    apply_relu,
):
    block_m = 32
    block_n = 64 if cout == 192 else 32
    grid = (
        input_tensor.shape[0],
        triton.cdiv(time, block_m),
        triton.cdiv(cout, block_n),
    )
    _conv1d_stage_kernel[grid](
        input_tensor,
        weight,
        bias,
        output_tensor,
        time,
        input_tensor.stride(0),
        output_tensor.stride(0),
        CIN=cin,
        COUT=cout,
        APPLY_RELU=apply_relu,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        num_warps=8 if block_n == 64 else 4,
        num_stages=2,
    )


def _launch_residual_stage(
    hidden,
    weight,
    bias,
    state,
    original_x,
    x_mask,
    output,
    time,
    sign,
    first_layer,
):
    block_m = 32
    block_n = 32
    grid = (
        hidden.shape[0],
        triton.cdiv(time, block_m),
        triton.cdiv(96, block_n),
    )
    _conv1d_residual_kernel[grid](
        hidden,
        weight,
        bias,
        state,
        original_x,
        x_mask,
        output,
        time,
        hidden.stride(0),
        SIGN=sign,
        FIRST_LAYER=first_layer,
        CIN=192,
        COUT=96,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
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

    if reverse:
        transforms = transforms[::-1]
        sign = -1.0
    else:
        sign = 1.0

    batch_size, _, time = x.shape
    hidden0 = torch.empty(
        (batch_size, 192, time),
        device=x.device,
        dtype=x.dtype,
    )
    hidden1 = torch.empty_like(hidden0)
    output = torch.empty_like(x)

    for layer_index, (
        conv0_weight,
        conv0_bias,
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
    ) in enumerate(transforms):
        conditioning = x if layer_index == 0 else output

        _launch_conv_stage(
            conditioning,
            conv0_weight,
            conv0_bias,
            hidden0,
            time,
            96,
            192,
            True,
        )
        _launch_conv_stage(
            hidden0,
            conv1_weight,
            conv1_bias,
            hidden1,
            time,
            192,
            192,
            True,
        )

        state = x if layer_index == 0 else output
        _launch_residual_stage(
            hidden1,
            conv2_weight,
            conv2_bias,
            state,
            x,
            x_mask,
            output,
            time,
            sign,
            layer_index == 0,
        )

    return output