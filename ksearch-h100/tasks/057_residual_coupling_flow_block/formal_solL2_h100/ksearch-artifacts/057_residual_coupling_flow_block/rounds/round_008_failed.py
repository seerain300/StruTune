# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r8 score=-1.0 passed=False
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
    time,
    sign: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    time_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)[None, :]
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    batch = tl.program_id(2)

    valid = (channels < 192) & (time_offsets < time)
    x_offsets = (batch * 192 + channels) * time + time_offsets

    x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)
    mask = tl.load(
        mask_ptr + batch * time + time_offsets,
        mask=time_offsets < time,
        other=0.0,
    )

    second_half = channels >= 96
    h_offsets = (batch * 96 + channels - 96) * time + time_offsets
    h_valid = valid & second_half

    h0 = tl.load(h0_ptr + h_offsets, mask=h_valid, other=0.0)
    h1 = tl.load(h1_ptr + h_offsets, mask=h_valid, other=0.0)
    h2 = tl.load(h2_ptr + h_offsets, mask=h_valid, other=0.0)
    h3 = tl.load(h3_ptr + h_offsets, mask=h_valid, other=0.0)

    second_half_result = x + sign * h0
    second_half_result += sign * h1
    second_half_result += sign * h2
    second_half_result += sign * h3

    result = tl.where(second_half, second_half_result, x) * mask
    tl.store(output_ptr + x_offsets, result, mask=valid)


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

    x0 = x[:, :96, :]
    h0 = _compute_transform(x0, transforms[0][0], transforms[0][1])
    h1 = _compute_transform(x0, transforms[1][0], transforms[1][1])
    h2 = _compute_transform(x0, transforms[2][0], transforms[2][1])
    h3 = _compute_transform(x0, transforms[3][0], transforms[3][1])

    output = torch.empty_like(x)
    batch = x.shape[0]
    time = x.shape[2]
    n_elements = x.numel()

    if n_elements < 262144:
        block_c = 2
        num_warps = 4
    else:
        block_c = 4
        num_warps = 8

    block_t = 128
    grid = (
        triton.cdiv(time, block_t),
        triton.cdiv(192, block_c),
        batch,
    )

    _final_residual_kernel[grid](
        x,
        h0,
        h1,
        h2,
        h3,
        x_mask,
        output,
        time,
        sign=sign,
        BLOCK_C=block_c,
        BLOCK_T=block_t,
        num_warps=num_warps,
    )

    return output