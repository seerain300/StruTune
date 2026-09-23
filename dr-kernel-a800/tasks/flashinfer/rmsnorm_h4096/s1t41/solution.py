import torch
import triton
import triton.language as tl


@triton.jit
def _compute_inv_rms_rows_kernel(
    x_ptr,            # *pointer to hidden_states (B,H)
    inv_rms_ptr,      # *pointer to output inv_rms (B,)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    BLOCK_SIZE: tl.constexpr,  # columns per chunk
    VEC_ROWS: tl.constexpr,    # number of rows processed per program
    NUM_ITERS: tl.constexpr,   # ceil(H / BLOCK_SIZE)
):
    # Each program handles VEC_ROWS rows
    pid = tl.program_id(0)
    rows = pid * VEC_ROWS + tl.arange(0, VEC_ROWS)
    mask_rows = rows < B

    # Accumulator for sum of squares per row
    sum_sq = tl.zeros([VEC_ROWS], dtype=tl.float32)

    # Loop over columns in chunks
    for it in tl.static_range(NUM_ITERS):
        cols = it * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
        mask_cols = cols < H
        # Build 2D offsets: [VEC_ROWS, BLOCK_SIZE]
        offs = rows[:, None] * stride_x_row + cols[None, :] * stride_x_col
        mask = mask_rows[:, None] & mask_cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=1)

    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + rows, inv_rms, mask=mask_rows)


@triton.jit
def _scale_output_rows_kernel(
    x_ptr,            # *pointer to hidden_states (B,H)
    weight_ptr,       # *pointer to weight (H,)
    inv_rms_ptr,      # *pointer to inv_rms (B,)
    out_ptr,          # *pointer to output (B,H)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    stride_out_row,   # stride for row in out
    stride_out_col,   # stride for col in out
    OUT_DTYPE: tl.constexpr,    # output dtype (tl.float16 or tl.bfloat16)
    BLOCK_SIZE: tl.constexpr,   # columns per chunk
    VEC_ROWS: tl.constexpr,     # number of rows processed per program
    NUM_ITERS: tl.constexpr,    # ceil(H / BLOCK_SIZE)
):
    pid = tl.program_id(0)
    rows = pid * VEC_ROWS + tl.arange(0, VEC_ROWS)
    mask_rows = rows < B
    inv_rms = tl.load(inv_rms_ptr + rows, mask=mask_rows, other=0.0).to(tl.float32)

    for it in tl.static_range(NUM_ITERS):
        cols = it * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < H

        # Load x chunk
        offs_x = rows[:, None] * stride_x_row + cols[None, :] * stride_x_col
        mask_x = mask_rows[:, None] & mask_cols[None, :]
        x = tl.load(x_ptr + offs_x, mask=mask_x, other=0.0).to(tl.float32)

        # Load weight chunk and broadcast inv_rms
        offs_w = cols  # weight is 1D
        w = tl.load(weight_ptr + offs_w, mask=mask_cols, other=0.0).to(tl.float32)
        y = x * inv_rms[:, None] * w[None, :]

        # Store output in original dtype
        offs_out = rows[:, None] * stride_out_row + cols[None, :] * stride_out_col
        tl.store(out_ptr + offs_out, y.to(OUT_DTYPE), mask=mask_x)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguous layout
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()
        B, H = hidden_states.shape

        # Prepare output tensor
        out = torch.empty_like(hidden_states)

        # Tiling parameters: process all columns in one iteration for H=4096
        BLOCK_SIZE = 4096
        VEC_ROWS = 8
        NUM_ITERS = (H + BLOCK_SIZE - 1) // BLOCK_SIZE  # typically 1 for H=4096

        # Launch grid: one program per group of VEC_ROWS rows
        grid = (triton.cdiv(B, VEC_ROWS),)

        # Kernel 1: compute per-row inv_rms
        inv_rms = torch.empty(B, dtype=torch.float32, device=hidden_states.device)
        _compute_inv_rms_rows_kernel[grid](
            hidden_states,
            inv_rms,
            B, H, 1e-5,
            hidden_states.stride(0), hidden_states.stride(1),
            BLOCK_SIZE=BLOCK_SIZE, VEC_ROWS=VEC_ROWS, NUM_ITERS=NUM_ITERS,
            num_warps=8, num_stages=2,
        )

        # Kernel 2: scale and write output
        _scale_output_rows_kernel[grid](
            hidden_states, weight, inv_rms, out,
            B, H,
            hidden_states.stride(0), hidden_states.stride(1),
            out.stride(0), out.stride(1),
            OUT_DTYPE=tl.float16 if out.dtype == torch.float16 else tl.bfloat16,
            BLOCK_SIZE=BLOCK_SIZE, VEC_ROWS=VEC_ROWS, NUM_ITERS=NUM_ITERS,
            num_warps=8, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
