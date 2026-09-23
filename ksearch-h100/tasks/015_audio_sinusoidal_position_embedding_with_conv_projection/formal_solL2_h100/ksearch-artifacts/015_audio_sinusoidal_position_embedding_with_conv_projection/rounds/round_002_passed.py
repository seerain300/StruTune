# solution=GPT-5.6-Sol_015_audio_sinusoidal_position_embedding_with_conv_projection_triton_optimized_r2 score=1.0108947306910372 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _projection_epilogue_kernel(
    output_ptr,
    position_ptr,
    NUM_ROWS: tl.constexpr,
    TIME_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    features = tl.arange(0, 1024)

    rows = tl.max_contiguous(rows, BLOCK_M)
    features = tl.max_contiguous(features, 1024)

    row_mask = rows < NUM_ROWS
    output_offsets = rows[:, None] * 1024 + features[None, :]
    times = rows % TIME_SIZE
    position_offsets = times[:, None] * 1024 + features[None, :]

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

    tl.store(
        output_ptr + output_offsets,
        values + position,
        mask=row_mask[:, None],
    )


@triton.jit
def _projection_epilogue_2d_kernel(
    output_ptr,
    position_ptr,
    TIME_SIZE: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BATCH_GROUP: tl.constexpr,
):
    batch = tl.program_id(1) * BATCH_GROUP
    times = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    features = tl.arange(0, 1024)

    times = tl.max_contiguous(times, BLOCK_M)
    features = tl.max_contiguous(features, 1024)

    time_mask = times < TIME_SIZE
    position_offsets = times[:, None] * 1024 + features[None, :]
    position = tl.load(
        position_ptr + position_offsets,
        mask=time_mask[:, None],
        other=0.0,
    )

    for batch_offset in tl.static_range(0, BATCH_GROUP):
        current_batch = batch + batch_offset
        output_mask = time_mask & (current_batch < BATCH_SIZE)
        rows = current_batch * TIME_SIZE + times
        output_offsets = rows[:, None] * 1024 + features[None, :]
        values = tl.load(
            output_ptr + output_offsets,
            mask=output_mask[:, None],
            other=0.0,
        )
        tl.store(
            output_ptr + output_offsets,
            values + position,
            mask=output_mask[:, None],
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

    x = F.conv2d(
        x,
        conv2d3_weight,
        conv2d3_bias,
        stride=2,
        padding=1,
    )
    x = torch.ops.aten.gelu_.default(x)

    batch_size = x.shape[0]
    time_size = x.shape[3]
    num_rows = batch_size * time_size

    x = x.permute(0, 3, 1, 2).contiguous().view(
        batch_size,
        time_size,
        3840,
    )

    if batch_size == 1:
        return torch.addmm(
            positional_embedding[:time_size, :],
            x.view(time_size, 3840),
            conv_out_weight.t(),
            beta=1,
            alpha=embed_scale,
        ).unsqueeze(0)

    x = torch.addmm(
        positional_embedding[:1, :1],
        x.view(num_rows, 3840),
        conv_out_weight.t(),
        beta=0,
        alpha=embed_scale,
    ).view(batch_size, time_size, 1024)

    if num_rows < 128:
        block_m = 1
        num_warps = 2
        _projection_epilogue_kernel[(triton.cdiv(num_rows, block_m),)](
            x,
            positional_embedding,
            NUM_ROWS=num_rows,
            TIME_SIZE=time_size,
            BLOCK_M=block_m,
            num_warps=num_warps,
        )
    elif num_rows < 2048:
        block_m = 4
        num_warps = 4
        _projection_epilogue_kernel[(triton.cdiv(num_rows, block_m),)](
            x,
            positional_embedding,
            NUM_ROWS=num_rows,
            TIME_SIZE=time_size,
            BLOCK_M=block_m,
            num_warps=num_warps,
        )
    else:
        block_m = 8
        num_warps = 8
        batch_group = 2 if time_size < 64 else 4
        _projection_epilogue_2d_kernel[
            (
                triton.cdiv(time_size, block_m),
                triton.cdiv(batch_size, batch_group),
            )
        ](
            x,
            positional_embedding,
            TIME_SIZE=time_size,
            BATCH_SIZE=batch_size,
            BLOCK_M=block_m,
            BATCH_GROUP=batch_group,
            num_warps=num_warps,
        )

    return x