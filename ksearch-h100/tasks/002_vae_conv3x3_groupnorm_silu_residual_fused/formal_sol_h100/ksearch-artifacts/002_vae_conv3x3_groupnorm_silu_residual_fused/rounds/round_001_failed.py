# solution=GPT-5.6-Sol_002_vae_conv3x3_groupnorm_silu_residual_fused_triton_optimized_r1 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


_CHANNELS = 256
_NUM_GROUPS = 32
_CHANNELS_PER_GROUP = 8
_CONV_K = 256 * 3 * 3


@triton.jit
def _conv3x3_implicit_gemm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    height,
    width,
    spatial_size,
    total_spatial,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    spatial_block = tl.program_id(0)
    channel_block = tl.program_id(1)

    offs_s = spatial_block * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)

    batch = offs_s // spatial_size
    spatial = offs_s - batch * spatial_size
    out_y = spatial // width
    out_x = spatial - out_y * width
    valid_s = offs_s < total_spatial

    accumulator = tl.zeros((BLOCK_C, BLOCK_S), dtype=tl.float32)

    for k_start in range(0, _CONV_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        in_channel = offs_k // 9
        kernel_pos = offs_k - in_channel * 9
        kernel_y = kernel_pos // 3
        kernel_x = kernel_pos - kernel_y * 3

        in_y = out_y[None, :] + kernel_y[:, None] - 1
        in_x = out_x[None, :] + kernel_x[:, None] - 1

        input_offsets = (
            batch[None, :] * (_CHANNELS * spatial_size)
            + in_channel[:, None] * spatial_size
            + in_y * width
            + in_x
        )
        input_mask = (
            (offs_k[:, None] < _CONV_K)
            & valid_s[None, :]
            & (in_y >= 0)
            & (in_y < height)
            & (in_x >= 0)
            & (in_x < width)
        )
        inputs = tl.load(
            x_ptr + input_offsets,
            mask=input_mask,
            other=0.0,
        )

        weight_offsets = offs_c[:, None] * _CONV_K + offs_k[None, :]
        weights = tl.load(
            weight_ptr + weight_offsets,
            mask=(offs_c[:, None] < _CHANNELS)
            & (offs_k[None, :] < _CONV_K),
            other=0.0,
        )

        accumulator += tl.dot(weights, inputs, input_precision="tf32")

    output_offsets = (
        batch[None, :] * (_CHANNELS * spatial_size)
        + offs_c[:, None] * spatial_size
        + spatial[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=(offs_c[:, None] < _CHANNELS) & valid_s[None, :],
    )


@triton.jit
def _group_sum_partials_kernel(
    input_ptr,
    partial_ptr,
    group_size,
    num_chunks,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    group = program // num_chunks
    chunk = program - group * num_chunks

    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        input_ptr + group * group_size + offsets,
        mask=offsets < group_size,
        other=0.0,
    )
    tl.store(partial_ptr + program, tl.sum(values, axis=0))


@triton.jit
def _group_mean_kernel(
    partial_ptr,
    mean_ptr,
    group_size,
    num_chunks,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    partials = tl.load(
        partial_ptr + group * num_chunks + offsets,
        mask=offsets < num_chunks,
        other=0.0,
    )
    mean = tl.sum(partials, axis=0) / group_size
    tl.store(mean_ptr + group, mean)


@triton.jit
def _group_variance_partials_kernel(
    input_ptr,
    mean_ptr,
    partial_ptr,
    group_size,
    num_chunks,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    group = program // num_chunks
    chunk = program - group * num_chunks

    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < group_size
    values = tl.load(
        input_ptr + group * group_size + offsets,
        mask=mask,
        other=0.0,
    )
    mean = tl.load(mean_ptr + group)
    differences = tl.where(mask, values - mean, 0.0)
    tl.store(partial_ptr + program, tl.sum(differences * differences, axis=0))


@triton.jit
def _group_rstd_kernel(
    partial_ptr,
    rstd_ptr,
    group_size,
    num_chunks,
    eps,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    partials = tl.load(
        partial_ptr + group * num_chunks + offsets,
        mask=offsets < num_chunks,
        other=0.0,
    )
    variance = tl.sum(partials, axis=0) / group_size
    tl.store(rstd_ptr + group, tl.rsqrt(variance + eps))


@triton.jit
def _group_norm_silu_kernel(
    input_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    output_ptr,
    total_elements,
    spatial_size,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    channel = (offsets // spatial_size) % _CHANNELS
    batch = offsets // (_CHANNELS * spatial_size)
    group = batch * _NUM_GROUPS + channel // _CHANNELS_PER_GROUP

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + group, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + group, mask=mask, other=0.0)
    scale = tl.load(norm_weight_ptr + channel, mask=mask, other=0.0)
    bias = tl.load(norm_bias_ptr + channel, mask=mask, other=0.0)

    normalized = (values - mean) * rstd
    affine = normalized * scale + bias
    activated = affine * tl.sigmoid(affine)

    if ADD_RESIDUAL:
        activated += tl.load(residual_ptr + offsets, mask=mask, other=0.0)

    tl.store(output_ptr + offsets, activated, mask=mask)


def _conv3x3(x, weight, output, height, width):
    spatial_size = height * width
    total_spatial = x.shape[0] * spatial_size
    grid = (
        triton.cdiv(total_spatial, 32),
        triton.cdiv(_CHANNELS, 32),
    )
    _conv3x3_implicit_gemm_kernel[grid](
        x,
        weight,
        output,
        height,
        width,
        spatial_size,
        total_spatial,
        BLOCK_C=32,
        BLOCK_S=32,
        BLOCK_K=32,
        num_warps=8,
        num_stages=3,
    )


def _group_norm_silu(
    input_tensor,
    norm_weight,
    norm_bias,
    eps,
    output,
    residual=None,
):
    batch_size, _, height, width = input_tensor.shape
    spatial_size = height * width
    group_size = _CHANNELS_PER_GROUP * spatial_size
    total_groups = batch_size * _NUM_GROUPS
    total_elements = batch_size * _CHANNELS * spatial_size

    reduction_block = 4096
    num_chunks = triton.cdiv(group_size, reduction_block)
    final_block = triton.next_power_of_2(num_chunks)

    partials = torch.empty(
        (total_groups, num_chunks),
        device=input_tensor.device,
        dtype=torch.float32,
    )
    means = torch.empty(
        (total_groups,),
        device=input_tensor.device,
        dtype=torch.float32,
    )
    rstd = torch.empty_like(means)

    partial_grid = (total_groups * num_chunks,)
    _group_sum_partials_kernel[partial_grid](
        input_tensor,
        partials,
        group_size,
        num_chunks,
        BLOCK=reduction_block,
        num_warps=8,
    )
    _group_mean_kernel[(total_groups,)](
        partials,
        means,
        group_size,
        num_chunks,
        BLOCK=final_block,
        num_warps=8,
    )
    _group_variance_partials_kernel[partial_grid](
        input_tensor,
        means,
        partials,
        group_size,
        num_chunks,
        BLOCK=reduction_block,
        num_warps=8,
    )
    _group_rstd_kernel[(total_groups,)](
        partials,
        rstd,
        group_size,
        num_chunks,
        eps,
        BLOCK=final_block,
        num_warps=8,
    )

    element_block = 256
    _group_norm_silu_kernel[(triton.cdiv(total_elements, element_block),)](
        input_tensor,
        norm_weight,
        norm_bias,
        means,
        rstd,
        residual if residual is not None else input_tensor,
        output,
        total_elements,
        spatial_size,
        ADD_RESIDUAL=residual is not None,
        BLOCK=element_block,
        num_warps=8,
    )


@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    batch_size, channels, height, width = x.shape
    assert channels == _CHANNELS

    conv_buffer = torch.empty_like(x)
    activation_buffer = torch.empty_like(x)
    output = torch.empty_like(x)

    _conv3x3(x, conv1_weight, conv_buffer, height, width)
    _group_norm_silu(
        conv_buffer,
        norm1_weight,
        norm1_bias,
        eps,
        activation_buffer,
    )

    _conv3x3(activation_buffer, conv2_weight, conv_buffer, height, width)
    _group_norm_silu(
        conv_buffer,
        norm2_weight,
        norm2_bias,
        eps,
        output,
        residual=x,
    )

    return output