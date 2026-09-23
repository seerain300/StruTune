# solution=GPT-5.6-Sol_015_audio_sinusoidal_position_embedding_with_conv_projection_triton_optimized_r16 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv3_bias_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    NUM_M: tl.constexpr,
    TIME_IN: tl.constexpr,
    TIME_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m < NUM_M

    time_out = m % TIME_OUT
    m_remaining = m // TIME_OUT
    freq_out = m_remaining % 10
    batch = m_remaining // 10

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 3456, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        input_channels = k // 9
        kernel_offset = k - input_channels * 9
        kernel_freq = kernel_offset // 3
        kernel_time = kernel_offset - kernel_freq * 3

        freq_in = freq_out[:, None] * 2 + kernel_freq[None, :] - 1
        time_in = time_out[:, None] * 2 + kernel_time[None, :] - 1

        input_offsets = (
            (
                batch[:, None] * 384
                + input_channels[None, :]
            )
            * 20
            + freq_in
        ) * TIME_IN + time_in

        activations = tl.load(
            input_ptr + input_offsets,
            mask=(
                m_mask[:, None]
                & (freq_in >= 0)
                & (freq_in < 20)
                & (time_in >= 0)
                & (time_in < TIME_IN)
            ),
            other=0.0,
        )

        weights = tl.load(
            weight_ptr
            + output_channels[None, :] * 3456
            + k[:, None]
        )

        accumulator += tl.dot(activations, weights)

    values = accumulator + tl.load(bias_ptr + output_channels)[None, :]
    values = 0.5 * values * (
        1.0 + tl.erf(values * 0.7071067811865476)
    )

    output_offsets = (
        (
            batch[:, None] * 384
            + output_channels[None, :]
        )
        * 10
        + freq_out[:, None]
    ) * TIME_OUT + time_out[:, None]

    tl.store(
        output_ptr + output_offsets,
        values,
        mask=m_mask[:, None],
    )


@triton.jit
def _projection_epilogue_kernel(
    output_ptr,
    position_ptr,
    embed_scale,
    NUM_ROWS: tl.constexpr,
    TIME_SIZE: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    features = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = rows < NUM_ROWS

    output_offsets = rows[:, None] * OUTPUT_DIM + features[None, :]
    times = rows % TIME_SIZE
    position_offsets = times[:, None] * OUTPUT_DIM + features[None, :]

    values = tl.load(
        output_ptr + output_offsets,
        mask=row_mask[:, None],
        other=0.0,
    )
    position = tl.load(
        position_ptr + position_offsets,
        mask=row_mask[:, None],
        other=0.0,
    )

    scaled = (values * embed_scale).to(tl.bfloat16)
    tl.store(
        output_ptr + output_offsets,
        scaled + position,
        mask=row_mask[:, None],
    )


@triton.jit
def _projection_position_kernel(
    input_ptr,
    weight_ptr,
    position_ptr,
    output_ptr,
    embed_scale,
    TIME_SIZE: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    K: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_features = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < NUM_ROWS

    batch = rows // TIME_SIZE
    time = rows - batch * TIME_SIZE
    input_base = batch * (K * TIME_SIZE) + time

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)

        activations = tl.load(
            input_ptr + input_base[:, None] + k[None, :] * TIME_SIZE,
            mask=row_mask[:, None],
            other=0.0,
        )
        weights = tl.load(
            weight_ptr + k[:, None] + output_features[None, :] * K,
        )
        accumulator += tl.dot(activations, weights)

    position = tl.load(
        position_ptr + time[:, None] * OUTPUT_DIM + output_features[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )

    projected = accumulator.to(tl.bfloat16)
    result = (projected * embed_scale).to(tl.bfloat16) + position

    tl.store(
        output_ptr + rows[:, None] * OUTPUT_DIM + output_features[None, :],
        result,
        mask=row_mask[:, None],
    )


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
    x = F.conv2d(
        input_features,
        conv2d1_weight,
        conv2d1_bias,
        stride=2,
        padding=1,
    )
    x = torch.ops.aten.gelu_.default(x)

    x = F.conv2d(
        x,
        conv2d2_weight,
        conv2d2_bias,
        stride=2,
        padding=1,
    )
    x = torch.ops.aten.gelu_.default(x)

    batch_size = x.shape[0]
    time_in = x.shape[3]
    time_size = (time_in + 1) // 2
    conv3_output = torch.empty(
        (batch_size, 384, 10, time_size),
        device=x.device,
        dtype=torch.bfloat16,
    )

    conv3_num_m = batch_size * 10 * time_size
    conv3_grid = (
        triton.cdiv(conv3_num_m, 32),
        triton.cdiv(384, 64),
    )

    _conv3_bias_gelu_kernel[conv3_grid](
        x,
        conv2d3_weight,
        conv2d3_bias,
        conv3_output,
        NUM_M=conv3_num_m,
        TIME_IN=time_in,
        TIME_OUT=time_size,
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    x = conv3_output

    num_rows = batch_size * time_size
    output_dim = 1024

    if num_rows <= 640:
        x = x.permute(0, 3, 1, 2).contiguous().view(
            batch_size,
            time_size,
            3840,
        )
        x = F.linear(x, conv_out_weight)

        grid = (
            triton.cdiv(num_rows, 4),
            triton.cdiv(output_dim, 256),
        )

        _projection_epilogue_kernel[grid](
            x,
            positional_embedding,
            embed_scale,
            NUM_ROWS=num_rows,
            TIME_SIZE=time_size,
            OUTPUT_DIM=output_dim,
            BLOCK_M=4,
            BLOCK_N=256,
            num_warps=4,
        )
        return x

    output = torch.empty(
        (batch_size, time_size, output_dim),
        device=x.device,
        dtype=torch.bfloat16,
    )

    if num_rows <= 4096:
        block_m = 32
        block_n = 256
        num_warps = 8
        num_stages = 3
    elif num_rows <= 8192:
        block_m = 32
        block_n = 128
        num_warps = 4
        num_stages = 3
    else:
        block_m = 64
        block_n = 128
        num_warps = 8
        num_stages = 4

    grid = (
        triton.cdiv(num_rows, block_m),
        triton.cdiv(output_dim, block_n),
    )

    _projection_position_kernel[grid](
        x,
        conv_out_weight,
        positional_embedding,
        output,
        embed_scale,
        TIME_SIZE=time_size,
        NUM_ROWS=num_rows,
        K=3840,
        OUTPUT_DIM=output_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output