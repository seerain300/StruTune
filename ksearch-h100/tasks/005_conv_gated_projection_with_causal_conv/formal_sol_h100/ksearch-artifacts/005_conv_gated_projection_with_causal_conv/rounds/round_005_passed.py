# solution=GPT-5.6-Sol_005_conv_gated_projection_with_causal_conv_triton_optimized_r5 score=1.4349758514675568 passed=True
import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 2048
_CONV_KERNEL_SIZE = 4


@triton.jit
def _linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    x_row_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    PLANAR_OUTPUT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = out_features // BLOCK_N
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, in_features, BLOCK_K):
        k = k_start + offsets_k

        if EVEN_M:
            x = tl.load(
                x_ptr + offsets_m[:, None] * x_row_stride + k[None, :]
            )
        else:
            x = tl.load(
                x_ptr + offsets_m[:, None] * x_row_stride + k[None, :],
                mask=offsets_m[:, None] < rows,
                other=0.0,
            )

        weight = tl.load(
            weight_ptr + offsets_n[None, :] * in_features + k[:, None]
        )
        accumulator += tl.dot(x, weight)

    bias = tl.load(bias_ptr + offsets_n).to(tl.float32)
    accumulator += bias[None, :]

    if PLANAR_OUTPUT:
        planes = offsets_n // in_features
        channels = offsets_n - planes * in_features
        output_offsets = (
            planes[None, :] * rows * in_features
            + offsets_m[:, None] * in_features
            + channels[None, :]
        )
    else:
        output_offsets = (
            offsets_m[:, None] * out_features + offsets_n[None, :]
        )

    if EVEN_M:
        tl.store(output_ptr + output_offsets, accumulator)
    else:
        tl.store(
            output_ptr + output_offsets,
            accumulator,
            mask=offsets_m[:, None] < rows,
        )


@triton.jit
def _causal_gated_conv_kernel(
    projected_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    projected_plane_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    channel_block = tl.program_id(0)
    sequence_block = tl.program_id(1)
    batch = tl.program_id(2)

    positions = sequence_block * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = channel_block * BLOCK_H + tl.arange(0, BLOCK_H)

    position_mask = positions < seq_len
    output_mask = position_mask[:, None]
    tokens = batch * seq_len + positions

    bias = tl.load(conv_bias_ptr + channels).to(tl.float32)
    accumulator = tl.broadcast_to(
        bias[None, :],
        (BLOCK_T, BLOCK_H),
    )

    source_positions = positions - 3
    source_tokens = batch * seq_len + source_positions
    source_mask = (
        (source_positions[:, None] >= 0)
        & position_mask[:, None]
    )
    b = tl.load(
        projected_ptr
        + source_tokens[:, None] * hidden_size
        + channels[None, :],
        mask=source_mask,
        other=0.0,
    )
    x_proj = tl.load(
        projected_ptr
        + 2 * projected_plane_stride
        + source_tokens[:, None] * hidden_size
        + channels[None, :],
        mask=source_mask,
        other=0.0,
    )
    conv_weight = tl.load(conv_weight_ptr + channels * 4)
    gated_input = (b * x_proj).to(tl.bfloat16)
    accumulator += (
        gated_input.to(tl.float32)
        * conv_weight[None, :].to(tl.float32)
    )

    source_positions = positions - 2
    source_tokens = batch * seq_len + source_positions
    source_mask = (
        (source_positions[:, None] >= 0)
        & position_mask[:, None]
    )
    b = tl.load(
        projected_ptr
        + source_tokens[:, None] * hidden_size
        + channels[None, :],
        mask=source_mask,
        other=0.0,
    )
    x_proj = tl.load(
        projected_ptr
        + 2 * projected_plane_stride
        + source_tokens[:, None] * hidden_size
        + channels[None, :],
        mask=source_mask,
        other=0.0,
    )
    conv_weight = tl.load(conv_weight_ptr + channels * 4 + 1)
    gated_input = (b * x_proj).to(tl.bfloat16)
    accumulator += (
        gated_input.to(tl.float32)
        * conv_weight[None, :].to(tl.float32)
    )

    source_positions = positions - 1
    source_tokens = batch * seq_len + source_positions
    source_mask = (
        (source_positions[:, None] >= 0)
        & position_mask[:, None]
    )
    b = tl.load(
        projected_ptr
        + source_tokens[:, None] * hidden_size
        + channels[None, :],
        mask=source_mask,
        other=0.0,
    )
    x_proj = tl.load(
        projected_ptr
        + 2 * projected_plane_stride
        + source_tokens[:, None] * hidden_size
        + channels[None, :],
        mask=source_mask,
        other=0.0,
    )
    conv_weight = tl.load(conv_weight_ptr + channels * 4 + 2)
    gated_input = (b * x_proj).to(tl.bfloat16)
    accumulator += (
        gated_input.to(tl.float32)
        * conv_weight[None, :].to(tl.float32)
    )

    source_positions = positions
    source_tokens = batch * seq_len + source_positions
    source_mask = position_mask[:, None]
    b = tl.load(
        projected_ptr
        + source_tokens[:, None] * hidden_size
        + channels[None, :],
        mask=source_mask,
        other=0.0,
    )
    x_proj = tl.load(
        projected_ptr
        + 2 * projected_plane_stride
        + source_tokens[:, None] * hidden_size
        + channels[None, :],
        mask=source_mask,
        other=0.0,
    )
    conv_weight = tl.load(conv_weight_ptr + channels * 4 + 3)
    gated_input = (b * x_proj).to(tl.bfloat16)
    accumulator += (
        gated_input.to(tl.float32)
        * conv_weight[None, :].to(tl.float32)
    )

    c = tl.load(
        projected_ptr
        + projected_plane_stride
        + tokens[:, None] * hidden_size
        + channels[None, :],
        mask=output_mask,
        other=0.0,
    )
    conv_output = accumulator.to(tl.bfloat16)
    gated_output = (c * conv_output).to(tl.bfloat16)

    tl.store(
        output_ptr
        + tokens[:, None] * output_row_stride
        + channels[None, :],
        gated_output,
        mask=output_mask,
    )


def _gemm_config(token_count, out_features):
    is_input_projection = out_features > _HIDDEN_SIZE

    if token_count <= 128:
        return 32, 128, 64, 4, 3

    if token_count <= 256:
        if is_input_projection:
            return 64, 128, 64, 4, 3
        return 32, 128, 64, 4, 3

    if token_count <= 512:
        if is_input_projection:
            if token_count % 512 == 0:
                return 64, 256, 64, 4, 3
            return 64, 128, 64, 4, 3
        return 64, 128, 64, 4, 4

    if token_count <= 1024:
        if is_input_projection:
            return 128, 256, 64, 8, 3
        return 64, 256, 64, 4, 4

    return 128, 256, 64, 8, 3


def _conv_config(token_count, seq_len):
    if token_count <= 1024:
        return 16, 256, 8
    if seq_len >= 2048:
        return 64, 256, 8
    return 32, 256, 8


def _linear(x, weight, bias):
    rows = x.numel() // x.shape[-1]
    in_features = x.shape[-1]
    out_features = weight.shape[0]
    x_row_stride = x.stride(-2)
    planar_output = out_features == 3 * _HIDDEN_SIZE

    output = torch.empty(
        (rows, out_features),
        device=x.device,
        dtype=x.dtype,
    )

    block_m, block_n, block_k, num_warps, num_stages = _gemm_config(
        rows,
        out_features,
    )
    grid = (
        triton.cdiv(rows, block_m)
        * (out_features // block_n),
    )

    _linear_kernel[grid](
        x,
        weight,
        bias,
        output,
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        x_row_stride=x_row_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        EVEN_M=rows % block_m == 0,
        PLANAR_OUTPUT=planar_output,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    batch_size, seq_len, hidden_size = x.shape
    token_count = batch_size * seq_len
    projected_plane_stride = token_count * hidden_size

    projected = _linear(
        x,
        in_proj_weight,
        in_proj_bias,
    )

    gated_conv_output = projected.view(-1)[
        projected_plane_stride:2 * projected_plane_stride
    ].view(token_count, hidden_size)

    block_t, block_h, conv_num_warps = _conv_config(
        token_count,
        seq_len,
    )
    grid = (
        hidden_size // block_h,
        triton.cdiv(seq_len, block_t),
        batch_size,
    )

    _causal_gated_conv_kernel[grid](
        projected,
        conv_weight,
        conv_bias,
        gated_conv_output,
        seq_len=seq_len,
        hidden_size=hidden_size,
        projected_plane_stride=projected_plane_stride,
        output_row_stride=hidden_size,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        num_warps=conv_num_warps,
    )

    output = _linear(
        gated_conv_output,
        out_proj_weight,
        out_proj_bias,
    )

    return output.reshape(batch_size, seq_len, hidden_size)