# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r25 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_packed_cache = {}


@triton.jit
def _residual_mask_kernel(
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
    h_channel = channels[:, None] - 96
    h_base = batch * 384 * time + h_channel * time + times[None, :]

    h0 = tl.load(
        h_ptr + h_base,
        mask=valid & second_half,
        other=0.0,
    )
    h1 = tl.load(
        h_ptr + h_base + 96 * time,
        mask=valid & second_half,
        other=0.0,
    )
    h2 = tl.load(
        h_ptr + h_base + 192 * time,
        mask=valid & second_half,
        other=0.0,
    )
    h3 = tl.load(
        h_ptr + h_base + 288 * time,
        mask=valid & second_half,
        other=0.0,
    )

    residual = (x + sign * h0) * mask
    residual = (residual + sign * h1) * mask
    residual = (residual + sign * h2) * mask
    residual = (residual + sign * h3) * mask
    result = tl.where(second_half, residual, x * mask)

    tl.store(output_ptr + x_offsets, result, mask=valid)


def _pack_transforms(transforms, reverse):
    sources = tuple(
        tensor
        for weights, biases in transforms
        for tensor in (
            weights[0],
            biases[0],
            weights[1],
            biases[1],
            weights[2],
            biases[2],
        )
    )
    versions = tuple(tensor._version for tensor in sources)
    cached = _packed_cache.get(reverse)

    if cached is not None:
        cached_sources, cached_versions, packed = cached
        if (
            cached_versions == versions
            and all(current is previous for current, previous in zip(sources, cached_sources))
        ):
            return packed

    packed = (
        torch.cat(tuple(weights[0] for weights, _ in transforms), dim=0),
        torch.cat(tuple(biases[0] for _, biases in transforms), dim=0),
        torch.cat(tuple(weights[1] for weights, _ in transforms), dim=0),
        torch.cat(tuple(biases[1] for _, biases in transforms), dim=0),
        torch.cat(tuple(weights[2] for weights, _ in transforms), dim=0),
        torch.cat(tuple(biases[2] for _, biases in transforms), dim=0),
    )
    _packed_cache[reverse] = (sources, versions, packed)
    return packed


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

    w0, b0, w1, b1, w2, b2 = _pack_transforms(transforms, reverse)

    h = F.conv1d(x[:, :96, :], w0, b0, padding=2)
    h = F.relu(h, inplace=True)
    h = F.conv1d(h, w1, b1, padding=2, groups=4)
    h = F.relu(h, inplace=True)
    h = F.conv1d(h, w2, b2, padding=2, groups=4)

    output = torch.empty_like(x)
    block_c = 32
    block_t = 128
    grid = (
        x.shape[0],
        triton.cdiv(192, block_c),
        triton.cdiv(x.shape[2], block_t),
    )

    _residual_mask_kernel[grid](
        x,
        h,
        x_mask,
        output,
        x.shape[2],
        sign=sign,
        BLOCK_C=block_c,
        BLOCK_T=block_t,
        num_warps=4,
    )

    return output