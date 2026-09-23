# solution=GPT-5.6-Sol_057_residual_coupling_flow_block_triton_optimized_r7 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_transform_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    conv0_w_ptr,
    conv0_b_ptr,
    conv1_w_ptr,
    conv1_b_ptr,
    conv2_w_ptr,
    conv2_b_ptr,
    time,
    sign: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    base_t = tile * BLOCK_T

    t_out = base_t + tl.arange(0, BLOCK_T)
    valid_out = t_out < time

    # The three size-5 convolutions have a combined receptive-field radius of 6.
    t_h0 = base_t - 4 + tl.arange(0, BLOCK_T + 8)
    t_h1 = base_t - 2 + tl.arange(0, BLOCK_T + 4)

    valid_h0 = (t_h0 >= 0) & (t_h0 < time)
    valid_h1 = (t_h1 >= 0) & (t_h1 < time)

    ic96 = tl.arange(0, 96)
    c192 = tl.arange(0, 192)
    oc96 = tl.arange(0, 96)

    x_batch = x_ptr + batch * 192 * time
    output_batch = output_ptr + batch * 192 * time
    mask_batch = mask_ptr + batch * time

    h0 = tl.zeros((192, BLOCK_T + 8), dtype=tl.float32)
    for k in range(5):
        w0 = tl.load(
            conv0_w_ptr
            + c192[:, None] * (96 * 5)
            + ic96[None, :] * 5
            + k
        )
        x_t = t_h0 + k - 2
        x0 = tl.load(
            x_batch + ic96[:, None] * time + x_t[None, :],
            mask=(x_t[None, :] >= 0) & (x_t[None, :] < time),
            other=0.0,
        )
        h0 += tl.dot(w0, x0, input_precision="ieee")

    b0 = tl.load(conv0_b_ptr + c192)
    h0 = tl.maximum(h0 + b0[:, None], 0.0)
    h0 = tl.where(valid_h0[None, :], h0, 0.0)

    h1 = tl.zeros((192, BLOCK_T + 4), dtype=tl.float32)
    for k in range(5):
        w1 = tl.load(
            conv1_w_ptr
            + c192[:, None] * (192 * 5)
            + c192[None, :] * 5
            + k
        )
        h0_slice = h0[:, k : k + BLOCK_T + 4]
        h1 += tl.dot(w1, h0_slice, input_precision="ieee")

    b1 = tl.load(conv1_b_ptr + c192)
    h1 = tl.maximum(h1 + b1[:, None], 0.0)
    h1 = tl.where(valid_h1[None, :], h1, 0.0)

    h2 = tl.zeros((96, BLOCK_T), dtype=tl.float32)
    for k in range(5):
        w2 = tl.load(
            conv2_w_ptr
            + oc96[:, None] * (192 * 5)
            + c192[None, :] * 5
            + k
        )
        h1_slice = h1[:, k : k + BLOCK_T]
        h2 += tl.dot(w2, h1_slice, input_precision="ieee")

    b2 = tl.load(conv2_b_ptr + oc96)
    h2 += b2[:, None]

    mask = tl.load(mask_batch + t_out, mask=valid_out, other=0.0)

    x0 = tl.load(
        x_batch + ic96[:, None] * time + t_out[None, :],
        mask=valid_out[None, :],
        other=0.0,
    )
    x1 = tl.load(
        x_batch + (ic96[:, None] + 96) * time + t_out[None, :],
        mask=valid_out[None, :],
        other=0.0,
    )

    x0_result = x0 * mask[None, :]
    x1_result = (x1 + sign * h2 * mask[None, :]) * mask[None, :]

    tl.store(
        output_batch + ic96[:, None] * time + t_out[None, :],
        x0_result,
        mask=valid_out[None, :],
    )
    tl.store(
        output_batch + (ic96[:, None] + 96) * time + t_out[None, :],
        x1_result,
        mask=valid_out[None, :],
    )


def _apply_transform(
    x,
    x_mask,
    output,
    weights,
    biases,
    sign,
):
    block_t = 64
    grid = (x.shape[0], triton.cdiv(x.shape[2], block_t))

    _fused_transform_kernel[grid](
        x,
        x_mask,
        output,
        weights[0],
        biases[0],
        weights[1],
        biases[1],
        weights[2],
        biases[2],
        x.shape[2],
        sign=sign,
        BLOCK_T=block_t,
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