# task: 020_vision_patch_merger_spatial_shuffle_mlp
# batch: ablation_full_feedback_20260917 (ablation)
# final eval: valid=True pass=15/15 geomean=2.198x

import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 1536
_EXPANDED_SIZE = 6144
_OUT_HIDDEN_SIZE = 3584


@triton.jit
def _normalize_shuffle_kernel(
    hidden_ptr,
    grid_ptr,
    merged_ptr,
    eps,
    NUM_GRIDS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    EXPANDED_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    merged_row = tl.program_id(0)
    quadrant = tl.program_id(1)

    if NUM_GRIDS == 1:
        selected_patch_prefix = 0
        selected_merged_prefix = 0
        selected_h = tl.load(grid_ptr + 1).to(tl.int32)
        selected_w = tl.load(grid_ptr + 2).to(tl.int32)
    else:
        patch_prefix = 0
        merged_prefix = 0

        selected_patch_prefix = 0
        selected_merged_prefix = 0
        selected_h = 2
        selected_w = 2

        for grid_idx in range(NUM_GRIDS):
            t = tl.load(grid_ptr + grid_idx * 3).to(tl.int32)
            h = tl.load(grid_ptr + grid_idx * 3 + 1).to(tl.int32)
            w = tl.load(grid_ptr + grid_idx * 3 + 2).to(tl.int32)

            grid_patches = t * h * w
            grid_merged = t * (h // 2) * (w // 2)

            belongs = (
                (merged_row >= merged_prefix)
                & (merged_row < merged_prefix + grid_merged)
            )

            selected_patch_prefix = tl.where(
                belongs, patch_prefix, selected_patch_prefix
            )
            selected_merged_prefix = tl.where(
                belongs, merged_prefix, selected_merged_prefix
            )
            selected_h = tl.where(belongs, h, selected_h)
            selected_w = tl.where(belongs, w, selected_w)

            patch_prefix += grid_patches
            merged_prefix += grid_merged

    local_merged = merged_row - selected_merged_prefix
    merged_w = selected_w // 2
    merged_plane = (selected_h // 2) * merged_w

    time_idx = local_merged // merged_plane
    spatial_idx = local_merged - time_idx * merged_plane
    merged_y = spatial_idx // merged_w
    merged_x = spatial_idx - merged_y * merged_w

    source_y = merged_y * 2 + quadrant // 2
    source_x = merged_x * 2 + quadrant % 2

    source_row = (
        selected_patch_prefix
        + time_idx * selected_h * selected_w
        + source_y * selected_w
        + source_x
    )

    channels = tl.arange(0, BLOCK_C)
    channel_mask = channels < HIDDEN_SIZE

    values = tl.load(
        hidden_ptr + source_row * HIDDEN_SIZE + channels,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)

    value_sum = tl.sum(values, axis=0)
    square_sum = tl.sum(values * values, axis=0)
    mean = value_sum / HIDDEN_SIZE
    variance = square_sum / HIDDEN_SIZE - mean * mean
    normalized = (values - mean) * tl.rsqrt(variance + eps)

    destination = (
        merged_ptr
        + merged_row * EXPANDED_SIZE
        + quadrant * HIDDEN_SIZE
        + channels
    )
    tl.store(destination, normalized, mask=channel_mask)


@triton.jit
def _gelu_kernel(
    values_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)

    columns = block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = columns < N
    offsets = row * N + columns

    values = tl.load(
        values_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    values_cubed = values * values * values
    sigmoid_arg = 1.5957691216057308 * (
        values + 0.044715 * values_cubed
    )
    gelu = values * tl.sigmoid(sigmoid_arg)

    tl.store(values_ptr + offsets, gelu, mask=mask)


@torch.no_grad()
def run(
    hidden: torch.Tensor,
    grid_thw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    eps: float,
):
    num_merged_patches = hidden.shape[0] // 4

    merged = torch.empty(
        (num_merged_patches, _EXPANDED_SIZE),
        dtype=torch.bfloat16,
        device=hidden.device,
    )

    _normalize_shuffle_kernel[(num_merged_patches, 4)](
        hidden,
        grid_thw,
        merged,
        eps,
        NUM_GRIDS=grid_thw.shape[0],
        HIDDEN_SIZE=_HIDDEN_SIZE,
        EXPANDED_SIZE=_EXPANDED_SIZE,
        BLOCK_C=2048,
        num_warps=4,
        num_stages=1,
    )

    if num_merged_patches > 512:
        hidden_fc1 = torch._addmm_activation(
            fc1_bias,
            merged,
            fc1_weight.t(),
            beta=1,
            alpha=1,
            use_gelu=True,
        )
    else:
        hidden_fc1 = torch.nn.functional.linear(
            merged,
            fc1_weight,
            fc1_bias,
        )

        if num_merged_patches <= 16:
            block_n = 256
            num_warps = 4
        elif num_merged_patches <= 128:
            block_n = 512
            num_warps = 4
        else:
            block_n = 1024
            num_warps = 8

        _gelu_kernel[
            (
                num_merged_patches,
                triton.cdiv(_EXPANDED_SIZE, block_n),
            )
        ](
            hidden_fc1,
            N=_EXPANDED_SIZE,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=1,
        )

    return torch.nn.functional.linear(
        hidden_fc1,
        fc2_weight,
        fc2_bias,
    )