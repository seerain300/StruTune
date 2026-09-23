# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r52 score=1.2912023095658416 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _coupling_finalize(
    input_ptr,
    output_ptr,
    h0_ptr,
    h1_ptr,
    h2_ptr,
    h3_ptr,
    mask_ptr,
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    time_size,
    SIGN: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_time = tl.program_id(1)

    batch_idx = pid // half_channels
    channel_idx = pid % half_channels

    time_offsets = pid_time * BLOCK_T + tl.arange(0, BLOCK_T)
    time_mask = time_offsets < time_size

    first_offset = (
        batch_idx * channels * time_size
        + channel_idx * time_size
        + time_offsets
    )
    second_offset = first_offset + half_channels * time_size
    h_offset = (
        batch_idx * half_channels * time_size
        + channel_idx * time_size
        + time_offsets
    )
    mask_offset = batch_idx * time_size + time_offsets

    mask = tl.load(mask_ptr + mask_offset, mask=time_mask, other=0.0)
    x0 = tl.load(input_ptr + first_offset, mask=time_mask, other=0.0)
    x1 = tl.load(input_ptr + second_offset, mask=time_mask, other=0.0)

    h0 = tl.load(h0_ptr + h_offset, mask=time_mask, other=0.0)
    h1 = tl.load(h1_ptr + h_offset, mask=time_mask, other=0.0)
    h2 = tl.load(h2_ptr + h_offset, mask=time_mask, other=0.0)
    h3 = tl.load(h3_ptr + h_offset, mask=time_mask, other=0.0)

    tl.store(
        output_ptr + first_offset,
        x0 * mask,
        mask=time_mask,
    )
    tl.store(
        output_ptr + second_offset,
        (x1 + SIGN * (h0 + h1 + h2 + h3) * mask) * mask,
        mask=time_mask,
    )


def _transform(
    x0,
    conv0_weight,
    conv0_bias,
    conv1_weight,
    conv1_bias,
    conv2_weight,
    conv2_bias,
):
    h = F.conv1d(x0, conv0_weight, conv0_bias, padding=2)
    h = F.relu_(h)
    h = F.conv1d(h, conv1_weight, conv1_bias, padding=2)
    h = F.relu_(h)
    return F.conv1d(h, conv2_weight, conv2_bias, padding=2)


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
            transform_0_conv0_weight,
            transform_0_conv0_bias,
            transform_0_conv1_weight,
            transform_0_conv1_bias,
            transform_0_conv2_weight,
            transform_0_conv2_bias,
        ),
        (
            transform_1_conv0_weight,
            transform_1_conv0_bias,
            transform_1_conv1_weight,
            transform_1_conv1_bias,
            transform_1_conv2_weight,
            transform_1_conv2_bias,
        ),
        (
            transform_2_conv0_weight,
            transform_2_conv0_bias,
            transform_2_conv1_weight,
            transform_2_conv1_bias,
            transform_2_conv2_weight,
            transform_2_conv2_bias,
        ),
        (
            transform_3_conv0_weight,
            transform_3_conv0_bias,
            transform_3_conv1_weight,
            transform_3_conv1_bias,
            transform_3_conv2_weight,
            transform_3_conv2_bias,
        ),
    )

    if reverse:
        transforms = transforms[::-1]

    batch_size, channels, time_size = x.shape
    half_channels = channels // 2
    sign = -1.0 if reverse else 1.0

    first = transforms[0]
    unmasked_x0 = x[:, :half_channels, :]
    h0 = _transform(
        unmasked_x0,
        first[0],
        first[1],
        first[2],
        first[3],
        first[4],
        first[5],
    )

    masked_x0 = unmasked_x0 * x_mask
    h_values = [h0]

    for transform in transforms[1:]:
        h_values.append(
            _transform(
                masked_x0,
                transform[0],
                transform[1],
                transform[2],
                transform[3],
                transform[4],
                transform[5],
            )
        )

    output = torch.empty_like(x)
    grid = (
        batch_size * half_channels,
        triton.cdiv(time_size, 256),
    )

    _coupling_finalize[grid](
        x,
        output,
        h_values[0],
        h_values[1],
        h_values[2],
        h_values[3],
        x_mask,
        channels=channels,
        half_channels=half_channels,
        time_size=time_size,
        SIGN=sign,
        BLOCK_T=256,
    )

    return output