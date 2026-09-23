# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r8 score=1.1053149998262144 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _residual_mask_kernel(
    x_ptr,
    h_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    time,
    sign: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements

    batch_stride = 192 * time
    batch = offsets // batch_stride
    within_batch = offsets - batch * batch_stride
    channel = within_batch // time
    time_index = within_batch - channel * time

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0)
    mask = tl.load(
        mask_ptr + batch * time + time_index,
        mask=valid,
        other=0.0,
    )

    second_half = channel >= 96
    h_channel = channel - 96
    h_offsets = batch * (96 * time) + h_channel * time + time_index
    h = tl.load(
        h_ptr + h_offsets,
        mask=valid & second_half,
        other=0.0,
    )

    result = tl.where(
        second_half,
        (x + sign * h * mask) * mask,
        x * mask,
    )
    tl.store(output_ptr + offsets, result, mask=valid)


def _apply_transform(
    x,
    x_mask,
    output,
    weights,
    biases,
    sign,
):
    h = F.conv1d(x[:, :96, :], weights[0], biases[0], padding=2)
    h = F.relu(h)
    h = F.conv1d(h, weights[1], biases[1], padding=2)
    h = F.relu(h)
    h = F.conv1d(h, weights[2], biases[2], padding=2)

    n_elements = x.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)

    _residual_mask_kernel[grid](
        x,
        h,
        x_mask,
        output,
        n_elements,
        x.shape[2],
        sign=sign,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


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

    output0 = torch.empty_like(x)
    output1 = torch.empty_like(x)

    current = x
    for layer, (weights, biases) in enumerate(transforms):
        output = output0 if layer % 2 == 0 else output1
        _apply_transform(
            current,
            x_mask,
            output,
            weights,
            biases,
            sign,
        )
        current = output

    return current