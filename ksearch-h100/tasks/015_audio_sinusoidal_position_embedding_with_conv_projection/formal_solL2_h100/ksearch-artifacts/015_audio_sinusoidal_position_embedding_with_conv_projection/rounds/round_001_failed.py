# solution=GPT-5.6-Sol_015_audio_sinusoidal_position_embedding_with_conv_projection_triton_optimized_r1 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _conv2d_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    input_height,
    input_width,
    output_height,
    output_width,
    K: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    num_rows = batch_size * output_height * output_width
    num_pid_m = tl.cdiv(num_rows, BLOCK_M)
    num_pid_n = tl.cdiv(OUT_CHANNELS, BLOCK_N)

    pid = tl.program_id(0)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_channels = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    spatial = output_height * output_width
    batch = rows // spatial
    output_position = rows - batch * spatial
    output_y = output_position // output_width
    output_x = output_position - output_y * output_width

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        input_channel = k // 9
        kernel_position = k - input_channel * 9
        kernel_y = kernel_position // 3
        kernel_x = kernel_position - kernel_y * 3

        input_y = output_y[:, None] * 2 + kernel_y[None, :] - 1
        input_x = output_x[:, None] * 2 + kernel_x[None, :] - 1

        input_offsets = (
            ((batch[:, None] * IN_CHANNELS + input_channel[None, :])
             * input_height + input_y)
            * input_width + input_x
        )
        input_mask = (
            (rows[:, None] < num_rows)
            & (k[None, :] < K)
            & (input_y >= 0)
            & (input_y < input_height)
            & (input_x >= 0)
            & (input_x < input_width)
        )
        activations = tl.load(
            input_ptr + input_offsets,
            mask=input_mask,
            other=0.0,
        )

        weight_offsets = out_channels[None, :] * K + k[:, None]
        weight_mask = (
            (k[:, None] < K)
            & (out_channels[None, :] < OUT_CHANNELS)
        )
        weights = tl.load(
            weight_ptr + weight_offsets,
            mask=weight_mask,
            other=0.0,
        )

        accumulator += tl.dot(activations, weights)

    bias = tl.load(
        bias_ptr + out_channels,
        mask=out_channels < OUT_CHANNELS,
        other=0.0,
    )
    result = _gelu(accumulator + bias[None, :])

    output_offsets = (
        ((batch[:, None] * OUT_CHANNELS + out_channels[None, :])
         * output_height + output_y[:, None])
        * output_width + output_x[:, None]
    )
    output_mask = (
        (rows[:, None] < num_rows)
        & (out_channels[None, :] < OUT_CHANNELS)
    )
    tl.store(output_ptr + output_offsets, result, mask=output_mask)


@triton.jit
def _projection_position_kernel(
    input_ptr,
    weight_ptr,
    position_ptr,
    output_ptr,
    batch_size,
    time_size,
    embed_scale,
    INPUT_CHANNELS: tl.constexpr,
    INPUT_HEIGHT: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    num_rows = batch_size * time_size
    num_pid_m = tl.cdiv(num_rows, BLOCK_M)
    num_pid_n = tl.cdiv(OUTPUT_DIM, BLOCK_N)

    pid = tl.program_id(0)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_features = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    batch = rows // time_size
    time = rows - batch * time_size
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        input_channel = k // INPUT_HEIGHT
        input_y = k - input_channel * INPUT_HEIGHT

        input_offsets = (
            ((batch[:, None] * INPUT_CHANNELS + input_channel[None, :])
             * INPUT_HEIGHT + input_y[None, :])
            * time_size + time[:, None]
        )
        input_mask = (rows[:, None] < num_rows) & (k[None, :] < K)
        activations = tl.load(
            input_ptr + input_offsets,
            mask=input_mask,
            other=0.0,
        )

        weight_offsets = output_features[None, :] * K + k[:, None]
        weight_mask = (
            (k[:, None] < K)
            & (output_features[None, :] < OUTPUT_DIM)
        )
        weights = tl.load(
            weight_ptr + weight_offsets,
            mask=weight_mask,
            other=0.0,
        )
        accumulator += tl.dot(activations, weights)

    position_offsets = time[:, None] * OUTPUT_DIM + output_features[None, :]
    output_mask = (
        (rows[:, None] < num_rows)
        & (output_features[None, :] < OUTPUT_DIM)
    )
    position = tl.load(
        position_ptr + position_offsets,
        mask=output_mask,
        other=0.0,
    ).to(tl.float32)

    result = accumulator * embed_scale + position
    output_offsets = rows[:, None] * OUTPUT_DIM + output_features[None, :]
    tl.store(output_ptr + output_offsets, result, mask=output_mask)


def _conv2d_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    batch_size, input_channels, input_height, input_width = x.shape
    output_channels = weight.shape[0]
    output_height = (input_height + 1) // 2
    output_width = (input_width + 1) // 2
    reduction_size = input_channels * 9

    output = torch.empty(
        (batch_size, output_channels, output_height, output_width),
        device=x.device,
        dtype=torch.bfloat16,
    )

    block_m = 32
    block_n = 64
    block_k = 16 if input_channels == 1 else 32
    num_rows = batch_size * output_height * output_width
    grid = (
        triton.cdiv(num_rows, block_m)
        * triton.cdiv(output_channels, block_n),
    )

    _conv2d_gelu_kernel[grid](
        x,
        weight,
        bias,
        output,
        batch_size,
        input_height,
        input_width,
        output_height,
        output_width,
        K=reduction_size,
        IN_CHANNELS=input_channels,
        OUT_CHANNELS=output_channels,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )
    return output


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    x = _conv2d_gelu(input_features, conv2d1_weight, conv2d1_bias)
    x = _conv2d_gelu(x, conv2d2_weight, conv2d2_bias)
    x = _conv2d_gelu(x, conv2d3_weight, conv2d3_bias)

    batch_size = x.shape[0]
    time_size = x.shape[3]
    output_dim = 1024
    reduction_size = 3840

    output = torch.empty(
        (batch_size, time_size, output_dim),
        device=x.device,
        dtype=torch.bfloat16,
    )

    block_m = 32
    block_n = 64
    block_k = 32
    grid = (
        triton.cdiv(batch_size * time_size, block_m)
        * triton.cdiv(output_dim, block_n),
    )

    _projection_position_kernel[grid](
        x,
        conv_out_weight,
        positional_embedding,
        output,
        batch_size,
        time_size,
        embed_scale,
        INPUT_CHANNELS=384,
        INPUT_HEIGHT=10,
        OUTPUT_DIM=output_dim,
        K=reduction_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
        num_stages=4,
    )
    return output