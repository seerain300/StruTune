# solution=GPT-5.6-Sol_002_vae_conv3x3_groupnorm_silu_residual_fused_triton_optimized_r6 score=-1.0 passed=False
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
    spatial_size,
    total_elements,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    channels = (offsets // spatial_size) % _CHANNELS
    weight = tl.load(norm_weight_ptr + channels, mask=mask, other=0.0)
    bias = tl.load(norm_bias_ptr + channels, mask=mask, other=0.0)

    values = values * weight + bias
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

    output, _, _ = torch.ops.aten.native_group_norm.default(
        input_tensor,
        None,
        None,
        batch_size,
        channels,
        height * width,
        _NUM_GROUPS,
        eps,
    )

    total_elements = output.numel()
    block = 1024
    _group_norm_epilogue_kernel[(triton.cdiv(total_elements, block),)](
        output,
        norm_weight,
        norm_bias,
        residual if residual is not None else output,
        height * width,
        total_elements,
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