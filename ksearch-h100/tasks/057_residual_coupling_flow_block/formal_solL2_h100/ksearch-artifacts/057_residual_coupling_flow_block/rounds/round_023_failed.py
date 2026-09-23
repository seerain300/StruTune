# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r23 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_PACKED_SOURCES = None
_PACKED_PARAMS = None


@triton.jit
def _combine_kernel(
    x_ptr,
    h_ptr,
    mask_ptr,
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
    times = time_block * BLOCK_T + tl.arange(0, BLOCK_T)

    channel_valid = channels < 192
    time_valid = times < time
    valid = channel_valid[:, None] & time_valid[None, :]

    x_offsets = (
        batch * 192 * time
        + channels[:, None] * time
        + times[None, :]
    )
    x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)

    mask = tl.load(
        mask_ptr + batch * time + times,
        mask=time_valid,
        other=0.0,
    )[None, :]

    second_half = channels[:, None] >= 96
    h_channels = channels - 96
    h_offsets = (
        batch * 384 * time
        + h_channels[:, None] * time
        + times[None, :]
    )
    h_valid = valid & second_half

    h0 = tl.load(h_ptr + h_offsets, mask=h_valid, other=0.0)
    h1 = tl.load(
        h_ptr + h_offsets + 96 * time,
        mask=h_valid,
        other=0.0,
    )
    h2 = tl.load(
        h_ptr + h_offsets + 192 * time,
        mask=h_valid,
        other=0.0,
    )
    h3 = tl.load(
        h_ptr + h_offsets + 288 * time,
        mask=h_valid,
        other=0.0,
    )

    residual = (h0 + h1) + (h2 + h3)
    result = tl.where(second_half, x + sign * residual, x) * mask
    tl.store(output_ptr + x_offsets, result, mask=valid)


def _packed_parameters(weights, biases):
    global _PACKED_SOURCES, _PACKED_PARAMS

    tensors = (
        weights[0][0],
        weights[1][0],
        weights[2][0],
        weights[3][0],
        biases[0][0],
        biases[1][0],
        biases[2][0],
        biases[3][0],
        weights[0][1],
        weights[1][1],
        weights[2][1],
        weights[3][1],
        biases[0][1],
        biases[1][1],
        biases[2][1],
        biases[3][1],
        weights[0][2],
        weights[1][2],
        weights[2][2],
        weights[3][2],
        biases[0][2],
        biases[1][2],
        biases[2][2],
        biases[3][2],
    )

    cache_valid = (
        _PACKED_SOURCES is not None
        and all(current is cached for current, cached in zip(tensors, _PACKED_SOURCES))
    )

    if not cache_valid:
        _PACKED_PARAMS = (
            torch.cat(
                (
                    weights[0][0],
                    weights[1][0],
                    weights[2][0],
                    weights[3][0],
                ),
                dim=0,
            ),
            torch.cat(
                (
                    biases[0][0],
                    biases[1][0],
                    biases[2][0],
                    biases[3][0],
                ),
                dim=0,
            ),
            torch.cat(
                (
                    weights[0][1],
                    weights[1][1],
                    weights[2][1],
                    weights[3][1],
                ),
                dim=0,
            ),
            torch.cat(
                (
                    biases[0][1],
                    biases[1][1],
                    biases[2][1],
                    biases[3][1],
                ),
                dim=0,
            ),
            torch.cat(
                (
                    weights[0][2],
                    weights[1][2],
                    weights[2][2],
                    weights[3][2],
                ),
                dim=0,
            ),
            torch.cat(
                (
                    biases[0][2],
                    biases[1][2],
                    biases[2][2],
                    biases[3][2],
                ),
                dim=0,
            ),
        )
        _PACKED_SOURCES = tensors

    return _PACKED_PARAMS


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
    weights = (
        (
            transform_0_conv0_weight,
            transform_0_conv1_weight,
            transform_0_conv2_weight,
        ),
        (
            transform_1_conv0_weight,
            transform_1_conv1_weight,
            transform_1_conv2_weight,
        ),
        (
            transform_2_conv0_weight,
            transform_2_conv1_weight,
            transform_2_conv2_weight,
        ),
        (
            transform_3_conv0_weight,
            transform_3_conv1_weight,
            transform_3_conv2_weight,
        ),
    )
    biases = (
        (
            transform_0_conv0_bias,
            transform_0_conv1_bias,
            transform_0_conv2_bias,
        ),
        (
            transform_1_conv0_bias,
            transform_1_conv1_bias,
            transform_1_conv2_bias,
        ),
        (
            transform_2_conv0_bias,
            transform_2_conv1_bias,
            transform_2_conv2_bias,
        ),
        (
            transform_3_conv0_bias,
            transform_3_conv1_bias,
            transform_3_conv2_bias,
        ),
    )

    w0, b0, w1, b1, w2, b2 = _packed_parameters(weights, biases)

    h = F.conv1d(x[:, :96, :], w0, b0, padding=2)
    h = F.relu(h)
    h = F.conv1d(h, w1, b1, padding=2, groups=4)
    h = F.relu(h)
    h = F.conv1d(h, w2, b2, padding=2, groups=4)

    output = torch.empty_like(x)
    block_c = 32
    block_t = 128
    grid = (
        x.shape[0],
        triton.cdiv(192, block_c),
        triton.cdiv(x.shape[2], block_t),
    )

    _combine_kernel[grid](
        x,
        h,
        x_mask,
        output,
        x.shape[2],
        sign=-1.0 if reverse else 1.0,
        BLOCK_C=block_c,
        BLOCK_T=block_t,
        num_warps=4,
    )
    return output