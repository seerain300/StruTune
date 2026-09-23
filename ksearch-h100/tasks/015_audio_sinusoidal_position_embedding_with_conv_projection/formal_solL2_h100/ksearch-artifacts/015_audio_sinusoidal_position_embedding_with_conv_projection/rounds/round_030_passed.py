# solution=GPT-5.6-Sol_015_audio_sinusoidal_position_embedding_with_conv_projection_triton_optimized_r30 score=0.997136027551179 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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
    output_dim = 1024

    x = x.permute(0, 3, 1, 2).contiguous().view(
        batch_size,
        time_size,
        3840,
    )

    if batch_size == 1:
        x = torch.addmm(
            positional_embedding[:time_size, :],
            x.view(time_size, 3840),
            conv_out_weight.t(),
            beta=1,
            alpha=embed_scale,
        ).unsqueeze(0)
        return x

    x = F.linear(x, conv_out_weight)

    if num_rows >= 2048:
        block_m = 8
        num_warps = 8
    else:
        block_m = 4
        num_warps = 4

    grid = (
        triton.cdiv(num_rows, block_m),
        triton.cdiv(output_dim, 256),
    )
    _projection_epilogue_kernel[grid](
        x,
        positional_embedding,
        embed_scale,
        NUM_ROWS=num_rows,
        TIME_SIZE=time_size,
        OUTPUT_DIM=output_dim,
        BLOCK_M=block_m,
        BLOCK_N=256,
        num_warps=num_warps,
    )

    return x