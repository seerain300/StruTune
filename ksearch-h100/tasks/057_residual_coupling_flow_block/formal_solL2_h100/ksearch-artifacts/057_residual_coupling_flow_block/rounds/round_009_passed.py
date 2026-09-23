# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r9 score=1.2761486064025909 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _final_residual_kernel(
    x_ptr,
    h0_ptr,
    h1_ptr,
    h2_ptr,
    h3_ptr,
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
    h_valid = valid & second_half

    h0 = tl.load(h0_ptr + h_offsets, mask=h_valid, other=0.0)
    h1 = tl.load(h1_ptr + h_offsets, mask=h_valid, other=0.0)
    h2 = tl.load(h2_ptr + h_offsets, mask=h_valid, other=0.0)
    h3 = tl.load(h3_ptr + h_offsets, mask=h_valid, other=0.0)

    first_half_result = x * mask

    second_half_result = (x + sign * h0 * mask) * mask
    second_half_result = (second_half_result + sign * h1 * mask) * mask
    second_half_result = (second_half_result + sign * h2 * mask) * mask
    second_half_result = (second_half_result + sign * h3 * mask) * mask

    result = tl.where(second_half, second_half_result, first_half_result)
    tl.store(output_ptr + offsets, result, mask=valid)


def _compute_transform(x0, weights, biases):
    h = F.conv1d(x0, weights[0], biases[0], padding=2)
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

    unmasked_x0 = x[:, :96, :]
    h0 = _compute_transform(
        unmasked_x0,
        transforms[0][0],
        transforms[0][1],
    )

    masked_x0 = unmasked_x0 * x_mask
    h1 = _compute_transform(masked_x0, transforms[1][0], transforms[1][1])
    h2 = _compute_transform(masked_x0, transforms[2][0], transforms[2][1])
    h3 = _compute_transform(masked_x0, transforms[3][0], transforms[3][1])

    output = torch.empty_like(x)
    time = x.shape[2]
    n_elements = x.numel()

    if n_elements < 262144:
        block_size = 256
        num_warps = 4
    else:
        block_size = 512
        num_warps = 8

    _final_residual_kernel[(triton.cdiv(n_elements, block_size),)](
        x,
        h0,
        h1,
        h2,
        h3,
        x_mask,
        output,
        n_elements,
        time,
        sign=sign,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output