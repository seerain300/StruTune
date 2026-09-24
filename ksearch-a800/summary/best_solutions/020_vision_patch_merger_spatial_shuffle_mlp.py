# task: 020_vision_patch_merger_spatial_shuffle_mlp
# bench: SOL-L1 | batch: ablation_strict_nolib_20260917（强化守卫消融解，替代 formal 版展示）
# final eval (official evaluator, full workloads): valid=True pass=15/15 geomean=1.464x
# 替代说明: formal_20260914 解调 F.linear×2（C·混合贡献，2.160x，库贡献约 32%）；本版为
#   强化守卫重跑的全 Triton 解（零库调用），51/100 轮时中止，终评 15/15 valid、1.464x（-32%）。
#   formal 版源码见同目录 020_vision_patch_merger_spatial_shuffle_mlp@formal_linear.py
# tokens: formal 批 2,255,698 | 消融批 948,286（75 次调用，中止前）

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_shuffle_kernel(
    hidden_ptr,
    grid_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    shuffled_ptr,
    eps,
    N_GRIDS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    merged_idx = tl.program_id(0)
    quadrant = tl.program_id(1)

    patch_start = tl.zeros((), dtype=tl.int32)
    merged_start = tl.zeros((), dtype=tl.int32)
    source_base = tl.zeros((), dtype=tl.int32)
    source_width = tl.zeros((), dtype=tl.int32)

    for grid_idx in tl.static_range(0, N_GRIDS):
        t = tl.load(grid_ptr + grid_idx * 3).to(tl.int32)
        h = tl.load(grid_ptr + grid_idx * 3 + 1).to(tl.int32)
        w = tl.load(grid_ptr + grid_idx * 3 + 2).to(tl.int32)

        patch_count = t * h * w
        merged_count = patch_count // 4
        local_merged_idx = merged_idx - merged_start
        belongs = (
            (local_merged_idx >= 0)
            & (local_merged_idx < merged_count)
        )

        merged_h = h // 2
        merged_w = w // 2
        merged_hw = merged_h * merged_w

        time_idx = local_merged_idx // merged_hw
        spatial_idx = local_merged_idx % merged_hw
        merged_y = spatial_idx // merged_w
        merged_x = spatial_idx % merged_w

        local_source_base = (
            time_idx * h * w
            + (merged_y * 2) * w
            + merged_x * 2
        )

        source_base = tl.where(
            belongs,
            patch_start + local_source_base,
            source_base,
        )
        source_width = tl.where(belongs, w, source_width)

        patch_start += patch_count
        merged_start += merged_count

    source_row = (
        source_base
        + (quadrant // 2) * source_width
        + quadrant % 2
    )

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE

    x = tl.load(
        hidden_ptr + source_row * HIDDEN_SIZE + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    mean = tl.sum(x, axis=0) * (1.0 / HIDDEN_SIZE)
    centered = tl.where(mask, x - mean, 0.0)
    variance = (
        tl.sum(centered * centered, axis=0)
        * (1.0 / HIDDEN_SIZE)
    )
    normalized = centered * tl.rsqrt(variance + eps)

    weight = tl.load(
        ln_weight_ptr + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        ln_bias_ptr + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    normalized = normalized * weight + bias

    output_offset = (
        merged_idx * (4 * HIDDEN_SIZE)
        + quadrant * HIDDEN_SIZE
        + cols
    )
    tl.store(
        shuffled_ptr + output_offset,
        normalized,
        mask=mask,
    )


@triton.jit
def _fc1_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(
        num_pid_m - first_pid_m,
        GROUP_M,
    )
    pid_in_group = pid % num_pid_in_group

    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    input_ptrs = (
        input_ptr
        + offs_m[:, None] * K
        + offs_k[None, :]
    )
    weight_ptrs = (
        weight_ptr
        + offs_n[None, :] * K
        + offs_k[:, None]
    )
    row_mask = offs_m[:, None] < M

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    for _ in range(0, K, BLOCK_K):
        input_tile = tl.load(
            input_ptrs,
            mask=row_mask,
            other=0.0,
        )
        weight_tile = tl.load(weight_ptrs)
        accumulator += tl.dot(input_tile, weight_tile)
        input_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K

    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    linear = (
        accumulator + bias[None, :]
    ).to(tl.bfloat16).to(tl.float32)

    gelu = 0.5 * linear * (
        1.0 + tl.erf(linear * 0.7071067811865476)
    )

    output_offsets = (
        offs_m[:, None] * N + offs_n[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        gelu,
        mask=row_mask,
    )


@triton.jit
def _fc2_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(
        num_pid_m - first_pid_m,
        GROUP_M,
    )
    pid_in_group = pid % num_pid_in_group

    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    input_ptrs = (
        input_ptr
        + offs_m[:, None] * K
        + offs_k[None, :]
    )
    weight_ptrs = (
        weight_ptr
        + offs_n[None, :] * K
        + offs_k[:, None]
    )
    row_mask = offs_m[:, None] < M

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    for _ in range(0, K, BLOCK_K):
        input_tile = tl.load(
            input_ptrs,
            mask=row_mask,
            other=0.0,
        )
        weight_tile = tl.load(weight_ptrs)
        accumulator += tl.dot(input_tile, weight_tile)
        input_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K

    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    result = accumulator + bias[None, :]

    output_offsets = (
        offs_m[:, None] * N + offs_n[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        result,
        mask=row_mask,
    )


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
    hidden_size = 1536
    expanded_size = 6144
    out_hidden_size = 3584

    num_patches = hidden.shape[0]
    m = num_patches // 4
    num_grids = grid_thw.shape[0]

    shuffled = torch.empty(
        (m, expanded_size),
        device=hidden.device,
        dtype=torch.bfloat16,
    )

    _layer_norm_shuffle_kernel[(m, 4)](
        hidden,
        grid_thw,
        ln_weight,
        ln_bias,
        shuffled,
        eps,
        N_GRIDS=num_grids,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=2048,
        num_warps=8,
    )

    fc1_output = torch.empty(
        (m, expanded_size),
        device=hidden.device,
        dtype=torch.bfloat16,
    )

    if m <= 16:
        fc1_cfg = (16, 128, 64, 1, 4, 4)
    elif m <= 32:
        fc1_cfg = (32, 128, 32, 1, 4, 4)
    elif m <= 64:
        fc1_cfg = (64, 128, 32, 1, 8, 4)
    elif m <= 128:
        fc1_cfg = (64, 128, 32, 2, 8, 4)
    elif m <= 256:
        fc1_cfg = (64, 128, 32, 4, 8, 4)
    elif m <= 512:
        fc1_cfg = (64, 128, 32, 8, 8, 4)
    elif m <= 1024:
        fc1_cfg = (128, 64, 32, 8, 8, 4)
    else:
        fc1_cfg = (128, 64, 32, 16, 8, 4)

    bm, bn, bk, gm, nw, ns = fc1_cfg
    fc1_grid = (
        triton.cdiv(m, bm)
        * triton.cdiv(expanded_size, bn),
    )

    _fc1_gelu_kernel[fc1_grid](
        shuffled,
        fc1_weight,
        fc1_bias,
        fc1_output,
        M=m,
        K=expanded_size,
        N=expanded_size,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        num_warps=nw,
        num_stages=ns,
    )

    output = torch.empty(
        (m, out_hidden_size),
        device=hidden.device,
        dtype=torch.bfloat16,
    )

    if m <= 16:
        fc2_cfg = (16, 128, 64, 1, 4, 4)
    elif m <= 32:
        fc2_cfg = (32, 128, 32, 1, 4, 4)
    elif m <= 64:
        fc2_cfg = (64, 128, 32, 1, 8, 4)
    elif m <= 128:
        fc2_cfg = (64, 64, 64, 2, 4, 4)
    elif m <= 256:
        fc2_cfg = (64, 64, 64, 4, 4, 4)
    elif m <= 512:
        fc2_cfg = (64, 64, 64, 8, 4, 4)
    elif m <= 1024:
        fc2_cfg = (128, 64, 32, 8, 8, 3)
    else:
        fc2_cfg = (128, 64, 32, 16, 8, 3)

    bm, bn, bk, gm, nw, ns = fc2_cfg
    fc2_grid = (
        triton.cdiv(m, bm)
        * triton.cdiv(out_hidden_size, bn),
    )

    _fc2_kernel[fc2_grid](
        fc1_output,
        fc2_weight,
        fc2_bias,
        output,
        M=m,
        K=expanded_size,
        N=out_hidden_size,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        num_warps=nw,
        num_stages=ns,
    )

    return output