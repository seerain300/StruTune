# solution=GPT-5.6-Sol_002_vae_conv3x3_groupnorm_silu_residual_fused_triton_optimized_r9 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_CHANNELS = 256
_NUM_GROUPS = 32


@triton.jit
def _group_norm_epilogue_kernel(
    input_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    residual_ptr,
    spatial_size: tl.constexpr,
    tiles_per_channel: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid % tiles_per_channel
    batch_channel = pid // tiles_per_channel
    channel = batch_channel % _CHANNELS

    spatial_offsets = tile * BLOCK + tl.arange(0, BLOCK)
    offsets = batch_channel * spatial_size + spatial_offsets
    mask = spatial_offsets < spatial_size

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(norm_weight_ptr + channel)
    bias = tl.load(norm_bias_ptr + channel)

    values = values * scale + bias
    values = values * tl.sigmoid(values)

    if ADD_RESIDUAL:
        values += tl.load(residual_ptr + offsets, mask=mask, other=0.0)

    tl.store(input_ptr + offsets, values, mask=mask)


def _group_norm_silu(
    input_tensor,
    norm_weight,
    norm_bias,
    eps,
    residual=None,
):
    batch_size, channels, height, width = input_tensor.shape
    spatial_size = height * width

    output, _, _ = torch.ops.aten.native_group_norm.default(
        input_tensor,
        None,
        None,
        batch_size,
        channels,
        spatial_size,
        _NUM_GROUPS,
        eps,
    )

    block = 1024
    tiles_per_channel = triton.cdiv(spatial_size, block)
    grid = (batch_size * channels * tiles_per_channel,)

    _group_norm_epilogue_kernel[grid](
        output,
        norm_weight,
        norm_bias,
        residual if residual is not None else output,
        spatial_size=spatial_size,
        tiles_per_channel=tiles_per_channel,
        ADD_RESIDUAL=residual is not None,
        BLOCK=block,
        num_warps=8,
    )
    return output


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
    assert x.shape[1] == _CHANNELS

    output = F.conv2d(
        x,
        conv1_weight,
        bias=None,
        stride=1,
        padding=1,
    )
    output = _group_norm_silu(
        output,
        norm1_weight,
        norm1_bias,
        eps,
    )

    output = F.conv2d(
        output,
        conv2_weight,
        bias=None,
        stride=1,
        padding=1,
    )
    output = _group_norm_silu(
        output,
        norm2_weight,
        norm2_bias,
        eps,
        residual=x,
    )

    return output