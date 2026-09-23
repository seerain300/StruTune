# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r6 score=1.1910882804534315 passed=True
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


@triton.jit
def _residual_update3_kernel(
    h0_ptr,
    h1_ptr,
    h2_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    time,
    sign: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements

    batch_stride = 96 * time
    batch = offsets // batch_stride
    within_batch = offsets - batch * batch_stride
    channel = within_batch // time
    time_index = within_batch - channel * time

    output_offsets = (
        batch * (192 * time)
        + (channel + 96) * time
        + time_index
    )

    x1 = tl.load(output_ptr + output_offsets, mask=valid, other=0.0)
    h0 = tl.load(h0_ptr + offsets, mask=valid, other=0.0)
    h1 = tl.load(h1_ptr + offsets, mask=valid, other=0.0)
    h2 = tl.load(h2_ptr + offsets, mask=valid, other=0.0)
    mask = tl.load(
        mask_ptr + batch * time + time_index,
        mask=valid,
        other=0.0,
    )

    result = (x1 + sign * h0 * mask) * mask
    result = (result + sign * h1 * mask) * mask
    result = (result + sign * h2 * mask) * mask
    tl.store(output_ptr + output_offsets, result, mask=valid)


def _compute_transform(x, weights, biases):
    h = F.conv1d(x[:, :96, :], weights[0], biases[0], padding=2)
    h.relu_()
    h = F.conv1d(h, weights[1], biases[1], padding=2)
    h.relu_()
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

    output = torch.empty_like(x)
    first_weights, first_biases = transforms[0]
    h = _compute_transform(x, first_weights, first_biases)

    time = x.shape[2]
    n_elements = x.numel()
    if n_elements < 262144:
        block_size = 256
        num_warps = 4
    else:
        block_size = 512
        num_warps = 8

    _residual_mask_kernel[(triton.cdiv(n_elements, block_size),)](
        x,
        h,
        x_mask,
        output,
        n_elements,
        time,
        sign=sign,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    h0 = _compute_transform(output, transforms[1][0], transforms[1][1])
    h1 = _compute_transform(output, transforms[2][0], transforms[2][1])
    h2 = _compute_transform(output, transforms[3][0], transforms[3][1])

    n_elements = h0.numel()
    if n_elements < 131072:
        block_size = 256
        num_warps = 4
    else:
        block_size = 512
        num_warps = 8

    _residual_update3_kernel[(triton.cdiv(n_elements, block_size),)](
        h0,
        h1,
        h2,
        x_mask,
        output,
        n_elements,
        time,
        sign=sign,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output