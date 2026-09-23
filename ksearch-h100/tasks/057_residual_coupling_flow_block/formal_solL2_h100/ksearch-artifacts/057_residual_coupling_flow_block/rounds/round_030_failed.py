# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r30 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _combine_residuals_kernel(
    x_ptr,
    h0_ptr,
    h1_ptr,
    h2_ptr,
    h3_ptr,
    output_ptr,
    time,
    sign: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    batch = tl.program_id(0)
    channel_block = tl.program_id(1)
    time_block = tl.program_id(2)

    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    time_offsets = time_block * BLOCK_T + tl.arange(0, BLOCK_T)

    channel_valid = channels < 192
    time_valid = time_offsets < time
    valid = channel_valid[:, None] & time_valid[None, :]

    x_offsets = (
        batch * 192 * time
        + channels[:, None] * time
        + time_offsets[None, :]
    )
    x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)

    second_half = channels[:, None] >= 96
    h_channels = channels - 96
    h_offsets = (
        batch * 96 * time
        + h_channels[:, None] * time
        + time_offsets[None, :]
    )
    h_valid = valid & second_half

    result = x

    h = tl.load(h0_ptr + h_offsets, mask=h_valid, other=0.0)
    result += sign * h

    h = tl.load(h1_ptr + h_offsets, mask=h_valid, other=0.0)
    result += sign * h

    h = tl.load(h2_ptr + h_offsets, mask=h_valid, other=0.0)
    result += sign * h

    h = tl.load(h3_ptr + h_offsets, mask=h_valid, other=0.0)
    result += sign * h

    tl.store(output_ptr + x_offsets, result, mask=valid)


def _apply_transform(x0, weights, biases):
    h = F.conv1d(x0, weights[0], biases[0], padding=2)
    h = F.relu(h)
    h = F.conv1d(h, weights[1], biases[1], padding=2)
    h = F.relu(h)
    return F.conv1d(h, weights[2], biases[2], padding=2)


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
    transforms = (
        (
            (
                transform_0_conv0_weight,
                transform_0_conv1_weight,
                transform_0_conv2_weight,
            ),
            (
                transform_0_conv0_bias,
                transform_0_conv1_bias,
                transform_0_conv2_bias,
            ),
        ),
        (
            (
                transform_1_conv0_weight,
                transform_1_conv1_weight,
                transform_1_conv2_weight,
            ),
            (
                transform_1_conv0_bias,
                transform_1_conv1_bias,
                transform_1_conv2_bias,
            ),
        ),
        (
            (
                transform_2_conv0_weight,
                transform_2_conv1_weight,
                transform_2_conv2_weight,
            ),
            (
                transform_2_conv0_bias,
                transform_2_conv1_bias,
                transform_2_conv2_bias,
            ),
        ),
        (
            (
                transform_3_conv0_weight,
                transform_3_conv1_weight,
                transform_3_conv2_weight,
            ),
            (
                transform_3_conv0_bias,
                transform_3_conv1_bias,
                transform_3_conv2_bias,
            ),
        ),
    )

    if reverse:
        transforms = transforms[::-1]
        sign = -1.0
    else:
        sign = 1.0

    x0 = x[:, :96, :]
    residuals = tuple(
        _apply_transform(x0, weights, biases)
        for weights, biases in transforms
    )

    output = torch.empty_like(x)
    block_c = 32
    block_t = 128
    grid = (
        x.shape[0],
        triton.cdiv(192, block_c),
        triton.cdiv(x.shape[2], block_t),
    )

    _combine_residuals_kernel[grid](
        x,
        residuals[0],
        residuals[1],
        residuals[2],
        residuals[3],
        output,
        x.shape[2],
        sign=sign,
        BLOCK_C=block_c,
        BLOCK_T=block_t,
        num_warps=4,
    )

    return output