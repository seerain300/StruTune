# solution=GPT-5.6-Sol_015_audio_sinusoidal_position_embedding_with_conv_projection_triton_optimized_r40 score=0.9953947056906993 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _projection_epilogue_kernel(
    output_ptr,
    position_ptr,
    embed_scale,
    TIME_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    times = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    batch = tl.program_id(1)
    features = tl.arange(0, 1024)

    times = tl.max_contiguous(times, BLOCK_M)
    features = tl.max_contiguous(features, 1024)

    time_mask = times < TIME_SIZE
    rows = batch * TIME_SIZE + times
    output_offsets = rows[:, None] * 1024 + features[None, :]
    position_offsets = times[:, None] * 1024 + features[None, :]

    values = tl.load(
        output_ptr + output_offsets,
        mask=time_mask[:, None],
        other=0.0,
    )
    position = tl.load(
        position_ptr + position_offsets,
        mask=time_mask[:, None],
        other=0.0,
    )

    scaled = (values * embed_scale).to(tl.bfloat16)
    tl.store(
        output_ptr + output_offsets,
        scaled + position,
        mask=time_mask[:, None],
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

    x = F.linear(x, conv_out_weight)

    if num_rows < 128:
        block_m = 1
        num_warps = 4
    elif num_rows < 1024:
        block_m = 4
        num_warps = 8
    else:
        block_m = 8
        num_warps = 8

    _projection_epilogue_kernel[
        (triton.cdiv(time_size, block_m), batch_size)
    ](
        x,
        positional_embedding,
        embed_scale,
        TIME_SIZE=time_size,
        BLOCK_M=block_m,
        num_warps=num_warps,
    )

    return x