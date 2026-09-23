# solution=GPT-5.6-Sol_020_vision_patch_merger_spatial_shuffle_mlp_triton_optimized_r3 score=1.7345271762098196 passed=True
import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 1536
_EXPANDED_SIZE = 6144
_OUT_HIDDEN_SIZE = 3584


@triton.jit
def _layer_norm_shuffle_kernel(
    hidden,
    grid_thw,
    ln_weight,
    ln_bias,
    shuffled,
    num_grids: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    merged_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    quadrants = tl.arange(0, 4)
    col_mask = cols < 1536

    if num_grids == 1:
        h = tl.load(grid_thw + 1).to(tl.int32)
        w = tl.load(grid_thw + 2).to(tl.int32)

        merged_h = h // 2
        merged_w = w // 2
        merged_hw = merged_h * merged_w

        time_idx = merged_idx // merged_hw
        spatial_idx = merged_idx - time_idx * merged_hw
        merged_row = spatial_idx // merged_w
        merged_col = spatial_idx - merged_row * merged_w

        patch_base = (
            time_idx * h * w
            + (merged_row * 2) * w
            + merged_col * 2
        )
        patch_indices = (
            patch_base
            + (quadrants // 2) * w
            + quadrants % 2
        )
    else:
        patch_offset = tl.full((), 0, tl.int32)
        merged_offset = tl.full((), 0, tl.int32)
        local_merged_idx = tl.full((), 0, tl.int32)
        selected_h = tl.full((), 0, tl.int32)
        selected_w = tl.full((), 0, tl.int32)
        selected_patch_offset = tl.full((), 0, tl.int32)
        found = tl.full((), False, tl.int1)

        for grid_idx in tl.static_range(0, 8):
            valid_grid = grid_idx < num_grids

            t = tl.load(
                grid_thw + grid_idx * 3,
                mask=valid_grid,
                other=0,
            ).to(tl.int32)
            h = tl.load(
                grid_thw + grid_idx * 3 + 1,
                mask=valid_grid,
                other=0,
            ).to(tl.int32)
            w = tl.load(
                grid_thw + grid_idx * 3 + 2,
                mask=valid_grid,
                other=0,
            ).to(tl.int32)

            grid_patches = t * h * w
            grid_merged = grid_patches // 4
            candidate = merged_idx - merged_offset

            in_grid = (
                valid_grid
                & (~found)
                & (candidate >= 0)
                & (candidate < grid_merged)
            )

            local_merged_idx = tl.where(
                in_grid,
                candidate,
                local_merged_idx,
            )
            selected_h = tl.where(in_grid, h, selected_h)
            selected_w = tl.where(in_grid, w, selected_w)
            selected_patch_offset = tl.where(
                in_grid,
                patch_offset,
                selected_patch_offset,
            )
            found = found | in_grid

            patch_offset += grid_patches
            merged_offset += grid_merged

        merged_h = selected_h // 2
        merged_w = selected_w // 2
        merged_hw = tl.maximum(merged_h * merged_w, 1)

        time_idx = local_merged_idx // merged_hw
        spatial_idx = local_merged_idx - time_idx * merged_hw
        merged_row = spatial_idx // tl.maximum(merged_w, 1)
        merged_col = spatial_idx - merged_row * merged_w

        patch_base = (
            selected_patch_offset
            + time_idx * selected_h * selected_w
            + (merged_row * 2) * selected_w
            + merged_col * 2
        )
        patch_indices = (
            patch_base
            + (quadrants // 2) * selected_w
            + quadrants % 2
        )

    x = tl.load(
        hidden
        + patch_indices[:, None] * 1536
        + cols[None, :],
        mask=col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    mean = tl.sum(x, axis=1) / 1536.0
    centered = tl.where(
        col_mask[None, :],
        x - mean[:, None],
        0.0,
    )
    variance = tl.sum(centered * centered, axis=1) / 1536.0
    inv_std = tl.rsqrt(variance + eps)

    weight = tl.load(
        ln_weight + cols,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        ln_bias + cols,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    normalized = (
        centered * inv_std[:, None] * weight[None, :]
        + bias[None, :]
    )
    output_cols = quadrants[:, None] * 1536 + cols[None, :]

    tl.store(
        shuffled + merged_idx * 6144 + output_cols,
        normalized,
        mask=col_mask[None, :],
    )


@triton.jit
def _projection_kernel(
    x,
    weight,
    bias,
    output,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    GELU: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tile_stride = tl.num_programs(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    programs_per_group = GROUP_M * num_pid_n

    while tile_id < num_tiles:
        group_id = tile_id // programs_per_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(
            num_pid_m - first_pid_m,
            GROUP_M,
        )
        pid_in_group = tile_id - group_id * programs_per_group

        pid_m = first_pid_m + pid_in_group % group_size_m
        pid_n = pid_in_group // group_size_m

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)

        accumulator = tl.zeros(
            (BLOCK_M, BLOCK_N),
            dtype=tl.float32,
        )

        for k_start in range(0, 6144, BLOCK_K):
            k = k_start + k_offsets

            if EVEN_M:
                x_tile = tl.load(
                    x
                    + rows[:, None] * 6144
                    + k[None, :],
                )
            else:
                x_tile = tl.load(
                    x
                    + rows[:, None] * 6144
                    + k[None, :],
                    mask=rows[:, None] < M,
                    other=0.0,
                )

            weight_tile = tl.load(
                weight
                + cols[None, :] * 6144
                + k[:, None],
            )
            accumulator = tl.dot(
                x_tile,
                weight_tile,
                accumulator,
            )

        accumulator += tl.load(
            bias + cols,
        )[None, :].to(tl.float32)

        if GELU:
            linear = accumulator.to(tl.bfloat16).to(tl.float32)
            result = 0.5 * linear * (
                1.0
                + tl.erf(linear * 0.7071067811865476)
            )
        else:
            result = accumulator

        if EVEN_M:
            tl.store(
                output + rows[:, None] * N + cols[None, :],
                result,
            )
        else:
            tl.store(
                output + rows[:, None] * N + cols[None, :],
                result,
                mask=rows[:, None] < M,
            )

        tile_id += tile_stride


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
    num_patches = hidden.shape[0]
    num_merged_patches = num_patches // 4
    num_grids = grid_thw.shape[0]

    shuffled = torch.empty(
        (num_merged_patches, _EXPANDED_SIZE),
        device=hidden.device,
        dtype=torch.bfloat16,
    )
    fc1_output = torch.empty_like(shuffled)
    output = torch.empty(
        (num_merged_patches, _OUT_HIDDEN_SIZE),
        device=hidden.device,
        dtype=torch.bfloat16,
    )

    _layer_norm_shuffle_kernel[(num_merged_patches,)](
        hidden,
        grid_thw,
        ln_weight,
        ln_bias,
        shuffled,
        num_grids=num_grids,
        eps=eps,
        BLOCK_SIZE=2048,
        num_warps=8,
    )

    if num_merged_patches <= 256:
        fc1_block_m = 32
        fc1_block_n = 256
        fc1_block_k = 128
        fc1_group_m = 4
        fc1_warps = 8
        fc1_stages = 2

        fc2_block_m = 32
        fc2_block_n = 128
        fc2_block_k = 128
        fc2_group_m = 4
        fc2_warps = 4
        fc2_stages = 2
    elif num_merged_patches <= 512:
        fc1_block_m = 64
        fc1_block_n = 256
        fc1_block_k = 128
        fc1_group_m = 8
        fc1_warps = 8
        fc1_stages = 2

        fc2_block_m = 64
        fc2_block_n = 128
        fc2_block_k = 128
        fc2_group_m = 8
        fc2_warps = 4
        fc2_stages = 3
    elif num_merged_patches <= 1024:
        fc1_block_m = 64
        fc1_block_n = 256
        fc1_block_k = 64
        fc1_group_m = 8
        fc1_warps = 8
        fc1_stages = 4

        fc2_block_m = 128
        fc2_block_n = 128
        fc2_block_k = 64
        fc2_group_m = 8
        fc2_warps = 8
        fc2_stages = 3
    elif num_merged_patches < 3072:
        fc1_block_m = 128
        fc1_block_n = 256
        fc1_block_k = 64
        fc1_group_m = 8
        fc1_warps = 8
        fc1_stages = 4

        fc2_block_m = 128
        fc2_block_n = 128
        fc2_block_k = 64
        fc2_group_m = 8
        fc2_warps = 8
        fc2_stages = 4
    else:
        fc1_block_m = 128
        fc1_block_n = 256
        fc1_block_k = 64
        fc1_group_m = 8
        fc1_warps = 8
        fc1_stages = 4

        fc2_block_m = 128
        fc2_block_n = 256
        fc2_block_k = 64
        fc2_group_m = 8
        fc2_warps = 8
        fc2_stages = 3

    fc1_tiles = (
        triton.cdiv(num_merged_patches, fc1_block_m)
        * triton.cdiv(_EXPANDED_SIZE, fc1_block_n)
    )

    if num_merged_patches < 1024:
        fc1_grid = (fc1_tiles,)
    else:
        num_sms = torch.cuda.get_device_properties(
            hidden.device
        ).multi_processor_count

        if num_merged_patches < 3072:
            fc1_grid = (min(fc1_tiles, num_sms * 2),)
        elif num_merged_patches < 8192:
            fc1_grid = (min(fc1_tiles, num_sms * 3),)
        else:
            fc1_grid = (min(fc1_tiles, num_sms * 4),)

    _projection_kernel[fc1_grid](
        shuffled,
        fc1_weight,
        fc1_bias,
        fc1_output,
        num_merged_patches,
        N=_EXPANDED_SIZE,
        BLOCK_M=fc1_block_m,
        BLOCK_N=fc1_block_n,
        BLOCK_K=fc1_block_k,
        GROUP_M=fc1_group_m,
        GELU=True,
        EVEN_M=num_merged_patches % fc1_block_m == 0,
        num_warps=fc1_warps,
        num_stages=fc1_stages,
    )

    fc2_tiles = (
        triton.cdiv(num_merged_patches, fc2_block_m)
        * triton.cdiv(_OUT_HIDDEN_SIZE, fc2_block_n)
    )

    if num_merged_patches < 1024:
        fc2_grid = (fc2_tiles,)
    else:
        if num_merged_patches < 3072:
            fc2_grid = (min(fc2_tiles, num_sms * 2),)
        elif num_merged_patches < 8192:
            fc2_grid = (min(fc2_tiles, num_sms * 3),)
        else:
            fc2_grid = (min(fc2_tiles, num_sms * 4),)

    _projection_kernel[fc2_grid](
        fc1_output,
        fc2_weight,
        fc2_bias,
        output,
        num_merged_patches,
        N=_OUT_HIDDEN_SIZE,
        BLOCK_M=fc2_block_m,
        BLOCK_N=fc2_block_n,
        BLOCK_K=fc2_block_k,
        GROUP_M=fc2_group_m,
        GELU=False,
        EVEN_M=num_merged_patches % fc2_block_m == 0,
        num_warps=fc2_warps,
        num_stages=fc2_stages,
    )

    return output