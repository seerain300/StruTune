# solution=GPT-5.6-Sol_015_audio_sinusoidal_position_embedding_with_conv_projection_triton_optimized_r20 score=0.998758581189377 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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

    use_native_projection = (
        num_rows <= 640
        or 1024 < num_rows <= 2048
        or num_rows >= 8192
    )

    if use_native_projection:
        x = x.permute(0, 3, 1, 2).contiguous().view(
            batch_size, time_size, 3840
        )
        x = F.linear(x, conv_out_weight)
        x = x * embed_scale
        x = x + positional_embedding[:time_size].unsqueeze(0)
        return x

    output_dim = 1024
    reduction_size = 3840

    output = torch.empty(
        (batch_size, time_size, output_dim),
        device=x.device,
        dtype=torch.bfloat16,
    )

    if num_rows < 1536:
        block_m = 32
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 3
    else:
        block_m = 64
        block_n = 128
        block_k = 64
        num_warps = 8
        num_stages = 3

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
        K=reduction_size,
        OUTPUT_DIM=output_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output