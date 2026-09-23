# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r13 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _final_residual_kernel(
    x_ptr,
    h0_ptr,
    h123_ptr,
    mask_ptr,
    output_ptr,
    n_pairs,
    time,
    sign: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = pair_offsets < n_pairs

    pair_time = 96 * time
    batch = pair_offsets // pair_time
    within_batch = pair_offsets - batch * pair_time
    channel = within_batch // time
    time_index = within_batch - channel * time

    x0_offsets = batch * (192 * time) + channel * time + time_index
    x1_offsets = x0_offsets + 96 * time

    packed_offsets = batch * (288 * time) + channel * time + time_index
    transform_stride = 96 * time

    x0 = tl.load(x_ptr + x0_offsets, mask=valid, other=0.0)
    x1 = tl.load(x_ptr + x1_offsets, mask=valid, other=0.0)
    mask = tl.load(
        mask_ptr + batch * time + time_index,
        mask=valid,
        other=0.0,
    )

    h0 = tl.load(h0_ptr + pair_offsets, mask=valid, other=0.0)
    h1 = tl.load(h123_ptr + packed_offsets, mask=valid, other=0.0)
    h2 = tl.load(
        h123_ptr + packed_offsets + transform_stride,
        mask=valid,
        other=0.0,
    )
    h3 = tl.load(
        h123_ptr + packed_offsets + 2 * transform_stride,
        mask=valid,
        other=0.0,
    )

    second_half_result = x1 + sign * h0
    second_half_result += sign * h1
    second_half_result += sign * h2
    second_half_result += sign * h3

    tl.store(output_ptr + x0_offsets, x0 * mask, mask=valid)
    tl.store(output_ptr + x1_offsets, second_half_result * mask, mask=valid)


def _compute_transform(x0, weights, biases):
    h = F.conv1d(x0, weights[0], biases[0], padding=2)
    h.relu_()
    h = F.conv1d(h, weights[1], biases[1], padding=2)
    h.relu_()
    return F.conv1d(h, weights[2], biases[2], padding=2)


def _compute_three_transforms(x0, transforms):
    conv0_weight = torch.cat(
        (
            transforms[0][0][0],
            transforms[1][0][0],
            transforms[2][0][0],
        ),
        dim=0,
    )
    conv0_bias = torch.cat(
        (
            transforms[0][1][0],
            transforms[1][1][0],
            transforms[2][1][0],
        ),
        dim=0,
    )
    h = F.conv1d(x0, conv0_weight, conv0_bias, padding=2)
    h.relu_()

    conv1_weight = torch.cat(
        (
            transforms[0][0][1],
            transforms[1][0][1],
            transforms[2][0][1],
        ),
        dim=0,
    )
    conv1_bias = torch.cat(
        (
            transforms[0][1][1],
            transforms[1][1][1],
            transforms[2][1][1],
        ),
        dim=0,
    )
    h = F.conv1d(h, conv1_weight, conv1_bias, padding=2, groups=3)
    h.relu_()

    conv2_weight = torch.cat(
        (
            transforms[0][0][2],
            transforms[1][0][2],
            transforms[2][0][2],
        ),
        dim=0,
    )
    conv2_bias = torch.cat(
        (
            transforms[0][1][2],
            transforms[1][1][2],
            transforms[2][1][2],
        ),
        dim=0,
    )
    return F.conv1d(h, conv2_weight, conv2_bias, padding=2, groups=3)


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
    h123 = _compute_three_transforms(masked_x0, transforms[1:])

    output = torch.empty_like(x)
    time = x.shape[2]
    n_elements = x.numel()
    n_pairs = n_elements // 2

    if n_elements < 262144:
        block_size = 256
        num_warps = 4
    else:
        block_size = 512
        num_warps = 8

    _final_residual_kernel[(triton.cdiv(n_pairs, block_size),)](
        x,
        h0,
        h123,
        x_mask,
        output,
        n_pairs,
        time,
        sign=sign,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output