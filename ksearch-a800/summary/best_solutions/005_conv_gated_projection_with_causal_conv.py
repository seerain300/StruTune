# task: 005_conv_gated_projection_with_causal_conv
# bench: SOL-L1 | batch: ablation_strict_nolib_20260917（强化守卫消融解，替代 formal 版展示）
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=1.420x
# 替代说明: formal_20260914 解调 F.linear×2（B·自研为主，1.565x）；本版为强化守卫重跑的
#   全 Triton 解（零库调用），43/100 轮时中止，终评 16/16 valid、1.420x（-10%）。
#   formal 版源码见同目录 005_conv_gated_projection_with_causal_conv@formal_linear.py
# tokens: formal 批 1,311,758 | 消融批 713,455（59 次调用，中止前）

import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 2048


def _projection_configs():
    return [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
    ]


_INPUT_CONFIGS = _projection_configs() + [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
]
_OUTPUT_CONFIGS = _projection_configs()


@triton.autotune(configs=_INPUT_CONFIGS, key=["M"])
@triton.jit
def _input_projection_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    projected_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    N: tl.constexpr = 3 * H

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * H + offs_k[None, :]
    weight_ptrs = weight_ptr + offs_n[None, :] * H + offs_k[:, None]
    mask_m = offs_m[:, None] < M

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, H, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m, other=0.0)
        weight = tl.load(weight_ptrs)
        accumulator += tl.dot(x, weight)
        x_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K

    bias = tl.load(bias_ptr + offs_n)
    accumulator += bias[None, :]

    stream = offs_n // H
    channel = offs_n - stream * H
    output_offsets = (
        stream[None, :] * M * H
        + offs_m[:, None] * H
        + channel[None, :]
    )
    tl.store(projected_ptr + output_offsets, accumulator, mask=mask_m)


@triton.jit
def _causal_gate_kernel(
    projected_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    y_ptr,
    M: tl.constexpr,
    seq_len: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    positions = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    channels = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    position_grid = positions[:, None]
    channel_grid = channels[None, :]
    valid_current = position_grid < seq_len

    rows = pid_b * seq_len + position_grid
    current_offsets = rows * H + channel_grid

    b0 = tl.load(
        projected_ptr + current_offsets,
        mask=valid_current,
        other=0.0,
    )
    x0 = tl.load(
        projected_ptr + 2 * M * H + current_offsets,
        mask=valid_current,
        other=0.0,
    )
    gated0 = (b0 * x0).to(tl.bfloat16)
    weight0 = tl.load(conv_weight_ptr + channel_grid * 4 + 3)
    accumulator = gated0.to(tl.float32) * weight0.to(tl.float32)

    for lag in tl.static_range(1, 4):
        source_offsets = current_offsets - lag * H
        valid_source = valid_current & (position_grid >= lag)

        b = tl.load(
            projected_ptr + source_offsets,
            mask=valid_source,
            other=0.0,
        )
        x_proj = tl.load(
            projected_ptr + 2 * M * H + source_offsets,
            mask=valid_source,
            other=0.0,
        )
        gated_input = (b * x_proj).to(tl.bfloat16)
        conv_weight = tl.load(
            conv_weight_ptr + channel_grid * 4 + (3 - lag)
        )
        accumulator += (
            gated_input.to(tl.float32) * conv_weight.to(tl.float32)
        )

    conv_bias = tl.load(conv_bias_ptr + channel_grid)
    conv_output = (accumulator + conv_bias).to(tl.bfloat16)

    c = tl.load(
        projected_ptr + M * H + current_offsets,
        mask=valid_current,
        other=0.0,
    )
    y = (c * conv_output).to(tl.bfloat16)

    tl.store(y_ptr + current_offsets, y, mask=valid_current)


@triton.autotune(configs=_OUTPUT_CONFIGS, key=["M"])
@triton.jit
def _output_projection_kernel(
    y_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(H, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    y_ptrs = y_ptr + offs_m[:, None] * H + offs_k[None, :]
    weight_ptrs = weight_ptr + offs_n[None, :] * H + offs_k[:, None]
    mask_m = offs_m[:, None] < M

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, H, BLOCK_K):
        y = tl.load(y_ptrs, mask=mask_m, other=0.0)
        weight = tl.load(weight_ptrs)
        accumulator += tl.dot(y, weight)
        y_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K

    bias = tl.load(bias_ptr + offs_n)
    accumulator += bias[None, :]

    output_offsets = offs_m[:, None] * H + offs_n[None, :]
    tl.store(output_ptr + output_offsets, accumulator, mask=mask_m)


@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    batch_size, seq_len, hidden_size = x.shape
    token_count = batch_size * seq_len

    projected = torch.empty(
        (3, token_count, hidden_size),
        device=x.device,
        dtype=x.dtype,
    )
    y = projected[1]
    output = torch.empty(
        (batch_size, seq_len, hidden_size),
        device=x.device,
        dtype=x.dtype,
    )

    if seq_len <= 2:
        causal_block_m = 2
        causal_block_h = 256
        causal_num_warps = 4
    elif seq_len <= 4:
        causal_block_m = 4
        causal_block_h = 256
        causal_num_warps = 4
    elif seq_len <= 64:
        causal_block_m = 8
        causal_block_h = 256
        causal_num_warps = 8
    else:
        causal_block_m = 16
        causal_block_h = 128
        causal_num_warps = 8

    with torch.cuda.device(x.device):
        input_grid = lambda meta: (
            triton.cdiv(token_count, meta["BLOCK_M"])
            * triton.cdiv(3 * hidden_size, meta["BLOCK_N"]),
        )
        _input_projection_kernel[input_grid](
            x,
            in_proj_weight,
            in_proj_bias,
            projected,
            M=token_count,
            H=_HIDDEN_SIZE,
        )

        _causal_gate_kernel[
            (
                triton.cdiv(seq_len, causal_block_m),
                triton.cdiv(hidden_size, causal_block_h),
                batch_size,
            )
        ](
            projected,
            conv_weight,
            conv_bias,
            y,
            M=token_count,
            seq_len=seq_len,
            H=_HIDDEN_SIZE,
            BLOCK_M=causal_block_m,
            BLOCK_H=causal_block_h,
            num_warps=causal_num_warps,
        )

        output_grid = lambda meta: (
            triton.cdiv(token_count, meta["BLOCK_M"])
            * triton.cdiv(hidden_size, meta["BLOCK_N"]),
        )
        _output_projection_kernel[output_grid](
            y,
            out_proj_weight,
            out_proj_bias,
            output,
            M=token_count,
            H=_HIDDEN_SIZE,
        )

    return output