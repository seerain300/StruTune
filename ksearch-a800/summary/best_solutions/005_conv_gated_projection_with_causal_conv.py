# task: 005_conv_gated_projection_with_causal_conv
# bench: SOL-L1 | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=1.565x
# feedback best (5-workload sample during search): 1.559x
# torch fallback audit: B·自研为主 (linear×2)
# tokens: 1,311,758

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _packed_conv_gate_kernel(
    projected_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    y_ptr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TIME_TILE: tl.constexpr,
):
    col_block = tl.program_id(0)
    time_block = tl.program_id(1)

    blocks_per_sequence = tl.cdiv(seq_len, TIME_TILE)
    batch_index = time_block // blocks_per_sequence
    tile_index = time_block % blocks_per_sequence

    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base_time = tile_index * TIME_TILE
    projected_stride = 3 * hidden_size
    batch_row = batch_index * seq_len

    w_base = conv_weight_ptr + cols * 4
    w_0 = tl.load(w_base).to(tl.float32)
    w_1 = tl.load(w_base + 1).to(tl.float32)
    w_2 = tl.load(w_base + 2).to(tl.float32)
    w_3 = tl.load(w_base + 3).to(tl.float32)

    time_0 = base_time - 3
    time_1 = base_time - 2
    time_2 = base_time - 1
    time_3 = base_time
    time_4 = base_time + 1
    time_5 = base_time + 2
    time_6 = base_time + 3

    row_0 = batch_row + time_0
    row_1 = batch_row + time_1
    row_2 = batch_row + time_2
    row_3 = batch_row + time_3
    row_4 = batch_row + time_4
    row_5 = batch_row + time_5
    row_6 = batch_row + time_6

    valid_0 = time_0 >= 0
    valid_1 = time_1 >= 0
    valid_2 = time_2 >= 0
    valid_3 = time_3 < seq_len
    valid_4 = time_4 < seq_len
    valid_5 = time_5 < seq_len
    valid_6 = time_6 < seq_len

    p_0 = projected_ptr + row_0 * projected_stride + cols
    p_1 = projected_ptr + row_1 * projected_stride + cols
    p_2 = projected_ptr + row_2 * projected_stride + cols
    p_3 = projected_ptr + row_3 * projected_stride + cols
    p_4 = projected_ptr + row_4 * projected_stride + cols
    p_5 = projected_ptr + row_5 * projected_stride + cols
    p_6 = projected_ptr + row_6 * projected_stride + cols

    b_0 = tl.load(p_0, mask=valid_0, other=0.0)
    x_0 = tl.load(p_0 + 2 * hidden_size, mask=valid_0, other=0.0)
    bx_0 = (b_0 * x_0).to(tl.bfloat16).to(tl.float32)

    b_1 = tl.load(p_1, mask=valid_1, other=0.0)
    x_1 = tl.load(p_1 + 2 * hidden_size, mask=valid_1, other=0.0)
    bx_1 = (b_1 * x_1).to(tl.bfloat16).to(tl.float32)

    b_2 = tl.load(p_2, mask=valid_2, other=0.0)
    x_2 = tl.load(p_2 + 2 * hidden_size, mask=valid_2, other=0.0)
    bx_2 = (b_2 * x_2).to(tl.bfloat16).to(tl.float32)

    b_3 = tl.load(p_3, mask=valid_3, other=0.0)
    x_3 = tl.load(p_3 + 2 * hidden_size, mask=valid_3, other=0.0)
    bx_3 = (b_3 * x_3).to(tl.bfloat16).to(tl.float32)

    b_4 = tl.load(p_4, mask=valid_4, other=0.0)
    x_4 = tl.load(p_4 + 2 * hidden_size, mask=valid_4, other=0.0)
    bx_4 = (b_4 * x_4).to(tl.bfloat16).to(tl.float32)

    b_5 = tl.load(p_5, mask=valid_5, other=0.0)
    x_5 = tl.load(p_5 + 2 * hidden_size, mask=valid_5, other=0.0)
    bx_5 = (b_5 * x_5).to(tl.bfloat16).to(tl.float32)

    b_6 = tl.load(p_6, mask=valid_6, other=0.0)
    x_6 = tl.load(p_6 + 2 * hidden_size, mask=valid_6, other=0.0)
    bx_6 = (b_6 * x_6).to(tl.bfloat16).to(tl.float32)

    acc_0 = bx_0 * w_0
    acc_0 += bx_1 * w_1
    acc_0 += bx_2 * w_2
    acc_0 += bx_3 * w_3

    acc_1 = bx_1 * w_0
    acc_1 += bx_2 * w_1
    acc_1 += bx_3 * w_2
    acc_1 += bx_4 * w_3

    acc_2 = bx_2 * w_0
    acc_2 += bx_3 * w_1
    acc_2 += bx_4 * w_2
    acc_2 += bx_5 * w_3

    acc_3 = bx_3 * w_0
    acc_3 += bx_4 * w_1
    acc_3 += bx_5 * w_2
    acc_3 += bx_6 * w_3

    bias = tl.load(conv_bias_ptr + cols).to(tl.float32)

    conv_0 = (acc_0 + bias).to(tl.bfloat16)
    conv_1 = (acc_1 + bias).to(tl.bfloat16)
    conv_2 = (acc_2 + bias).to(tl.bfloat16)
    conv_3 = (acc_3 + bias).to(tl.bfloat16)

    c_0 = tl.load(p_3 + hidden_size, mask=valid_3, other=0.0)
    c_1 = tl.load(p_4 + hidden_size, mask=valid_4, other=0.0)
    c_2 = tl.load(p_5 + hidden_size, mask=valid_5, other=0.0)
    c_3 = tl.load(p_6 + hidden_size, mask=valid_6, other=0.0)

    tl.store(
        y_ptr + row_3 * hidden_size + cols,
        (c_0 * conv_0).to(tl.bfloat16),
        mask=valid_3,
    )
    tl.store(
        y_ptr + row_4 * hidden_size + cols,
        (c_1 * conv_1).to(tl.bfloat16),
        mask=valid_4,
    )
    tl.store(
        y_ptr + row_5 * hidden_size + cols,
        (c_2 * conv_2).to(tl.bfloat16),
        mask=valid_5,
    )
    tl.store(
        y_ptr + row_6 * hidden_size + cols,
        (c_3 * conv_3).to(tl.bfloat16),
        mask=valid_6,
    )


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
    num_rows = batch_size * seq_len

    projected = F.linear(
        x.reshape(num_rows, hidden_size),
        in_proj_weight,
        in_proj_bias,
    )

    y = torch.empty(
        (num_rows, hidden_size),
        device=x.device,
        dtype=torch.bfloat16,
    )

    block_size = 256
    time_tile = 4
    grid = (
        triton.cdiv(hidden_size, block_size),
        batch_size * triton.cdiv(seq_len, time_tile),
    )

    _packed_conv_gate_kernel[grid](
        projected,
        conv_weight,
        conv_bias,
        y,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=block_size,
        TIME_TILE=time_tile,
        num_warps=4,
        num_stages=1,
    )

    output = F.linear(y, out_proj_weight, out_proj_bias)
    return output.reshape(batch_size, seq_len, hidden_size)