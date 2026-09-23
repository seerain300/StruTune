# solution=GPT-5.6-Sol_036_convnextv2_layer_with_nhwc_persistence_backward_triton_optimized_r18 score=23.34738055088559 passed=True
import torch
import triton
import triton.language as tl


C = 128
C4 = 512
TC = tl.constexpr(128)
TC4 = tl.constexpr(512)


@triton.jit
def _prepare_projected_kernel(
    grad_output,
    drop_mask,
    projected,
    spatial_size: tl.constexpr,
    total_elements,
    keep_scale,
    apply_drop: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total_elements

    channel = offsets % TC
    token = offsets // TC
    spatial = token % spatial_size
    batch = token // spatial_size

    nchw_offset = (batch * TC + channel) * spatial_size + spatial
    values = tl.load(grad_output + nchw_offset, mask=valid, other=0.0)

    if apply_drop:
        mask = tl.load(drop_mask + batch, mask=valid, other=0.0)
        values *= mask * keep_scale

    tl.store(projected + offsets, values, mask=valid)


@triton.jit
def _persistent_grn_backward_kernel(
    grad_x_grn,
    x_gelu,
    x_expanded,
    global_features,
    gf_mean,
    norm_features,
    grn_weight,
    grad_x_expanded,
    partial_grn_weight,
    partial_grn_bias,
    partial_pwconv1_bias,
    spatial_size: tl.constexpr,
    eps,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    channel_block = tl.program_id(1)

    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < TC4
    channel_offsets = batch * TC4 + channels

    weight = tl.load(
        grn_weight + channels,
        mask=channel_mask,
        other=0.0,
    )
    norm = tl.load(
        norm_features + channel_offsets,
        mask=channel_mask,
        other=0.0,
    )
    global_feature = tl.load(
        global_features + channel_offsets,
        mask=channel_mask,
        other=0.0,
    )
    mean_feature = tl.load(gf_mean + batch)

    grad_norm = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for spatial_start in tl.range(0, spatial_size, BLOCK_S):
        spatial = spatial_start + tl.arange(0, BLOCK_S)
        mask = (
            (spatial[:, None] < spatial_size)
            & channel_mask[None, :]
        )
        offsets = (
            (batch * spatial_size + spatial[:, None]) * TC4
            + channels[None, :]
        )

        grad = tl.load(
            grad_x_grn + offsets,
            mask=mask,
            other=0.0,
        )
        gelu = tl.load(
            x_gelu + offsets,
            mask=mask,
            other=0.0,
        )
        grad_norm += tl.sum(
            grad * weight[None, :] * gelu,
            axis=0,
        )

    denominator = mean_feature + eps
    grad_global = (
        grad_norm / denominator
        - grad_norm * global_feature
        / (denominator * denominator * TC4)
    )

    grn_weight_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    grn_bias_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    pwconv1_bias_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for spatial_start in tl.range(0, spatial_size, BLOCK_S):
        spatial = spatial_start + tl.arange(0, BLOCK_S)
        mask = (
            (spatial[:, None] < spatial_size)
            & channel_mask[None, :]
        )
        offsets = (
            (batch * spatial_size + spatial[:, None]) * TC4
            + channels[None, :]
        )

        grad = tl.load(
            grad_x_grn + offsets,
            mask=mask,
            other=0.0,
        )
        gelu = tl.load(
            x_gelu + offsets,
            mask=mask,
            other=0.0,
        )
        expanded = tl.load(
            x_expanded + offsets,
            mask=mask,
            other=0.0,
        )

        grad_gelu = (
            grad
            + grad * weight[None, :] * norm[None, :]
            + gelu
            * grad_global[None, :]
            / (global_feature[None, :] + eps)
        )

        expanded_sq = expanded * expanded
        inner = 0.7978845608028654 * (
            expanded
            + 0.044715 * expanded * expanded_sq
        )
        tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        cdf = 0.5 * (1.0 + tanh_inner)
        pdf = (
            0.5
            * (1.0 - tanh_inner * tanh_inner)
            * 0.7978845608028654
            * (1.0 + 0.134145 * expanded_sq)
        )
        gelu_grad = cdf + expanded * pdf
        grad_expanded = grad_gelu * gelu_grad

        tl.store(
            grad_x_expanded + offsets,
            grad_expanded,
            mask=mask,
        )

        grn_weight_acc += tl.sum(
            grad * gelu * norm[None, :],
            axis=0,
        )
        grn_bias_acc += tl.sum(grad, axis=0)
        pwconv1_bias_acc += tl.sum(grad_expanded, axis=0)

    tl.store(
        partial_grn_weight + channel_offsets,
        grn_weight_acc,
        mask=channel_mask,
    )
    tl.store(
        partial_grn_bias + channel_offsets,
        grn_bias_acc,
        mask=channel_mask,
    )
    tl.store(
        partial_pwconv1_bias + channel_offsets,
        pwconv1_bias_acc,
        mask=channel_mask,
    )


@triton.jit
def _grn_batch_reduce_kernel(
    partial_grn_weight,
    partial_grn_bias,
    partial_pwconv1_bias,
    grad_grn_weight,
    grad_grn_bias,
    grad_pwconv1_bias,
    batch_size,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_block = tl.program_id(0)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < TC4
    batches = tl.arange(0, BLOCK_B)

    mask = (
        (batches[:, None] < batch_size)
        & channel_mask[None, :]
    )
    offsets = batches[:, None] * TC4 + channels[None, :]

    grn_weight_values = tl.load(
        partial_grn_weight + offsets,
        mask=mask,
        other=0.0,
    )
    grn_bias_values = tl.load(
        partial_grn_bias + offsets,
        mask=mask,
        other=0.0,
    )
    pwconv1_bias_values = tl.load(
        partial_pwconv1_bias + offsets,
        mask=mask,
        other=0.0,
    )

    tl.store(
        grad_grn_weight + channels,
        tl.sum(grn_weight_values, axis=0),
        mask=channel_mask,
    )
    tl.store(
        grad_grn_bias + channels,
        tl.sum(grn_bias_values, axis=0),
        mask=channel_mask,
    )
    tl.store(
        grad_pwconv1_bias + channels,
        tl.sum(pwconv1_bias_values, axis=0),
        mask=channel_mask,
    )


@triton.jit
def _column_sum_kernel(
    values,
    output,
    row_count,
    column_count: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_block = tl.program_id(0)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < column_count
    accum = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for row_start in tl.range(0, row_count, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < row_count) & channel_mask[None, :]
        offsets = rows[:, None] * column_count + channels[None, :]
        values_block = tl.load(
            values + offsets,
            mask=mask,
            other=0.0,
        )
        accum += tl.sum(values_block, axis=0)

    tl.store(output + channels, accum, mask=channel_mask)


@triton.jit
def _layernorm_parameter_reduce_kernel(
    grad_x_ln,
    x_normalized,
    grad_weight,
    grad_bias,
    token_count,
    height: tl.constexpr,
    width: tl.constexpr,
    xn_s0,
    xn_s1,
    xn_s2,
    xn_s3,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_block = tl.program_id(0)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < TC

    weight_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    bias_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for token_start in tl.range(0, token_count, BLOCK_M):
        tokens = token_start + tl.arange(0, BLOCK_M)
        spatial = tokens % (height * width)
        batch = tokens // (height * width)
        h = spatial // width
        w = spatial % width

        mask = (tokens[:, None] < token_count) & channel_mask[None, :]
        grad_offsets = tokens[:, None] * TC + channels[None, :]
        normalized_offsets = (
            batch[:, None] * xn_s0
            + h[:, None] * xn_s1
            + w[:, None] * xn_s2
            + channels[None, :] * xn_s3
        )

        grad = tl.load(
            grad_x_ln + grad_offsets,
            mask=mask,
            other=0.0,
        )
        normalized = tl.load(
            x_normalized + normalized_offsets,
            mask=mask,
            other=0.0,
        )

        weight_acc += tl.sum(grad * normalized, axis=0)
        bias_acc += tl.sum(grad, axis=0)

    tl.store(
        grad_weight + channels,
        weight_acc,
        mask=channel_mask,
    )
    tl.store(
        grad_bias + channels,
        bias_acc,
        mask=channel_mask,
    )


@triton.jit
def _layernorm_input_kernel(
    grad_x_ln,
    x_normalized,
    var,
    layernorm_weight,
    grad_x_dwconv,
    token_count,
    height: tl.constexpr,
    width: tl.constexpr,
    xn_s0,
    xn_s1,
    xn_s2,
    xn_s3,
    var_s0,
    var_s1,
    var_s2,
    eps,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    channels = tl.arange(0, BLOCK_C)
    channel_mask = channels < TC
    valid_token = token < token_count

    spatial = token % (height * width)
    batch = token // (height * width)
    h = spatial // width
    w = spatial % width

    normalized_offsets = (
        batch * xn_s0
        + h * xn_s1
        + w * xn_s2
        + channels * xn_s3
    )
    grad_offsets = token * TC + channels
    var_offset = batch * var_s0 + h * var_s1 + w * var_s2

    mask = valid_token & channel_mask
    normalized = tl.load(
        x_normalized + normalized_offsets,
        mask=mask,
        other=0.0,
    )
    grad = tl.load(
        grad_x_ln + grad_offsets,
        mask=mask,
        other=0.0,
    )
    weight = tl.load(
        layernorm_weight + channels,
        mask=mask,
        other=0.0,
    )
    token_var = tl.load(
        var + var_offset,
        mask=valid_token,
        other=0.0,
    )

    grad_normalized = grad * weight
    grad_sum = tl.sum(grad_normalized, axis=0)
    grad_normalized_sum = tl.sum(
        grad_normalized * normalized,
        axis=0,
    )
    inv_std = tl.rsqrt(token_var + eps)

    grad_input = inv_std * (
        grad_normalized
        - grad_sum / TC
        - normalized * grad_normalized_sum / TC
    )

    output_offsets = (
        (batch * TC + channels) * (height * width) + spatial
    )
    tl.store(
        grad_x_dwconv + output_offsets,
        grad_input,
        mask=mask,
    )


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    residual: torch.Tensor,
    x_dwconv: torch.Tensor,
    x_nhwc: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    x_normalized: torch.Tensor,
    x_ln: torch.Tensor,
    x_expanded: torch.Tensor,
    x_gelu: torch.Tensor,
    global_features: torch.Tensor,
    gf_mean: torch.Tensor,
    norm_features: torch.Tensor,
    x_grn_scaled: torch.Tensor,
    x_grn: torch.Tensor,
    dwconv_weight: torch.Tensor,
    layernorm_weight: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    grn_weight: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    drop_mask: torch.Tensor,
    drop_path_prob: float,
    eps: float,
):
    batch, _, height, width = grad_output.shape
    spatial_size = height * width
    token_count = batch * spatial_size
    device = grad_output.device

    grad_projected = torch.empty(
        (token_count, C),
        device=device,
        dtype=torch.float32,
    )

    projected_elements = token_count * C
    apply_drop = drop_path_prob > 0.0
    keep_scale = (
        1.0 / (1.0 - drop_path_prob)
        if apply_drop
        else 1.0
    )

    _prepare_projected_kernel[
        (triton.cdiv(projected_elements, 256),)
    ](
        grad_output,
        drop_mask,
        grad_projected,
        spatial_size,
        projected_elements,
        keep_scale,
        apply_drop=apply_drop,
        BLOCK=256,
        num_warps=4,
    )

    x_grn_flat = x_grn.reshape(token_count, C4)
    grad_x_grn = torch.mm(grad_projected, pwconv2_weight)

    grad_pwconv2_weight = torch.mm(
        grad_projected.transpose(0, 1),
        x_grn_flat,
    )

    grad_pwconv2_bias = torch.empty(
        (C,),
        device=device,
        dtype=torch.float32,
    )
    _column_sum_kernel[
        (triton.cdiv(C, 8),)
    ](
        grad_projected,
        grad_pwconv2_bias,
        token_count,
        column_count=C,
        BLOCK_M=256,
        BLOCK_C=8,
        num_warps=4,
    )

    grad_x_expanded = torch.empty(
        (token_count, C4),
        device=device,
        dtype=torch.float32,
    )
    grn_partials = torch.empty(
        (3, batch, C4),
        device=device,
        dtype=torch.float32,
    )

    _persistent_grn_backward_kernel[
        (batch, triton.cdiv(C4, 8))
    ](
        grad_x_grn,
        x_gelu.reshape(token_count, C4),
        x_expanded.reshape(token_count, C4),
        global_features,
        gf_mean,
        norm_features,
        grn_weight,
        grad_x_expanded,
        grn_partials[0],
        grn_partials[1],
        grn_partials[2],
        spatial_size,
        eps,
        BLOCK_S=128,
        BLOCK_C=8,
        num_warps=4,
    )

    grad_grn_weight = torch.empty(
        (1, 1, 1, C4),
        device=device,
        dtype=torch.float32,
    )
    grad_grn_bias = torch.empty_like(grad_grn_weight)
    grad_pwconv1_bias = torch.empty(
        (C4,),
        device=device,
        dtype=torch.float32,
    )

    _grn_batch_reduce_kernel[
        (triton.cdiv(C4, 8),)
    ](
        grn_partials[0],
        grn_partials[1],
        grn_partials[2],
        grad_grn_weight,
        grad_grn_bias,
        grad_pwconv1_bias,
        batch,
        BLOCK_B=64,
        BLOCK_C=8,
        num_warps=4,
    )

    x_ln_flat = x_ln.reshape(token_count, C)
    grad_x_ln = torch.mm(grad_x_expanded, pwconv1_weight)

    grad_pwconv1_weight = torch.mm(
        grad_x_expanded.transpose(0, 1),
        x_ln_flat,
    )

    grad_layernorm_weight = torch.empty(
        (C,),
        device=device,
        dtype=torch.float32,
    )
    grad_layernorm_bias = torch.empty_like(grad_layernorm_weight)

    _layernorm_parameter_reduce_kernel[
        (triton.cdiv(C, 8),)
    ](
        grad_x_ln,
        x_normalized,
        grad_layernorm_weight,
        grad_layernorm_bias,
        token_count,
        height,
        width,
        x_normalized.stride(0),
        x_normalized.stride(1),
        x_normalized.stride(2),
        x_normalized.stride(3),
        BLOCK_M=256,
        BLOCK_C=8,
        num_warps=4,
    )

    grad_x_dwconv = torch.empty_like(grad_output)

    _layernorm_input_kernel[
        (token_count,)
    ](
        grad_x_ln,
        x_normalized,
        var,
        layernorm_weight,
        grad_x_dwconv,
        token_count,
        height,
        width,
        x_normalized.stride(0),
        x_normalized.stride(1),
        x_normalized.stride(2),
        x_normalized.stride(3),
        var.stride(0),
        var.stride(1),
        var.stride(2),
        eps,
        BLOCK_C=128,
        num_warps=4,
    )

    grad_conv_input, grad_dwconv_weight, grad_dwconv_bias = (
        torch.ops.aten.convolution_backward(
            grad_x_dwconv,
            residual,
            dwconv_weight,
            [C],
            [1, 1],
            [3, 3],
            [1, 1],
            False,
            [0, 0],
            C,
            [True, True, True],
        )
    )

    grad_x = grad_conv_input + grad_output

    return (
        grad_x,
        grad_dwconv_weight,
        grad_dwconv_bias,
        grad_layernorm_weight,
        grad_layernorm_bias,
        grad_pwconv1_weight,
        grad_pwconv1_bias,
        grad_grn_weight,
        grad_grn_bias,
        grad_pwconv2_weight,
        grad_pwconv2_bias,
    )