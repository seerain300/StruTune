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
    BLOCK_SIZE: tl.constexpr,  # columns processed per iteration
    VEC_ROWS: tl.constexpr,    # number of rows processed per program
    NUM_ITERS: tl.constexpr,   # number of iterations to cover H
):
    # One program processes VEC_ROWS rows
    pid = tl.program_id(0)
    row_start = pid * VEC_ROWS
    rows = row_start + tl.arange(0, VEC_ROWS)
    row_mask = rows < B

    # Accumulator for sum of squares per row
    sumsq = tl.zeros([VEC_ROWS], dtype=tl.float32)

    # Loop over columns in chunks
    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < H
        offs = rows[:, None] * stride_x_row + cols[None, :] * stride_x_col
        mask2d = row_mask[:, None] & col_mask[None, :]
        x_chunk = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
        x_chunk = x_chunk.to(tl.float32)
        sumsq += tl.sum(x_chunk * x_chunk, axis=1)

    # Compute mean and inv_rms
    H_f32 = tl.full([1], H, dtype=tl.float32)
    mean = sumsq / H_f32
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + rows, inv_rms, mask=row_mask)


@triton.jit
def _scale_rows_with_weight_kernel(
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
    BLOCK_SIZE: tl.constexpr,  # columns processed per iteration
    VEC_ROWS: tl.constexpr,    # number of rows processed per program
    NUM_ITERS: tl.constexpr,   # number of iterations to cover H
    OUT_DTYPE: tl.constexpr,   # output dtype (tl.float16 or tl.bfloat16)
):
    # One program processes VEC_ROWS rows
    pid = tl.program_id(0)
    row_start = pid * VEC_ROWS
    rows = row_start + tl.arange(0, VEC_ROWS)
    row_mask = rows < B

    # Load inv_rms for these rows (float32)
    inv_rms = tl.load(inv_rms_ptr + rows, mask=row_mask, other=1.0)

    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < H

        # Load weight chunk as float32
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        # For each row in this program, compute and store outputs
        for r in range(VEC_ROWS):
            row = row_start + r
            rm = row < B
            if rm:
                offs_x = row * stride_x_row + cols * stride_x_col
                x = tl.load(x_ptr + offs_x, mask=col_mask, other=0.0).to(tl.float32)
                y = x * inv_rms[r] * w
                y = y.to(OUT_DTYPE)
                offs_out = row * stride_out_row + cols * stride_out_col
                tl.store(out_ptr + offs_out, y, mask=col_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are contiguous and on CUDA
        assert hidden_states.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors"
        x = hidden_states.contiguous()
        w = weight.contiguous()
        B, H = x.shape

        # Output tensor (empty; no torch ops on host)
        out = torch.empty((B, H), device=x.device)

        # Tiling parameters
        BLOCK_SIZE = 2048
        VEC_ROWS = 8
        NUM_ITERS = (H + BLOCK_SIZE - 1) // BLOCK_SIZE  # e.g., 2 for H=4096

        # Compute inv_rms (B,)
        inv_rms = torch.empty(B, dtype=torch.float32, device=x.device)

        stride_x_row = x.stride(0)
        stride_x_col = x.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Launch norm kernel: one program handles VEC_ROWS rows
        grid_norm = (triton.cdiv(B, VEC_ROWS),)
        _compute_inv_rms_rows_kernel[grid_norm](
            x, inv_rms, B, H, 1e-5,
            stride_x_row, stride_x_col,
            BLOCK_SIZE=BLOCK_SIZE, VEC_ROWS=VEC_ROWS, NUM_ITERS=NUM_ITERS,
            num_warps=8, num_stages=2,
        )

        # Launch output kernel
        grid_out = (triton.cdiv(B, VEC_ROWS),)
        out_dtype = tl.float16 if x.dtype == torch.float16 else tl.bfloat16
        _scale_rows_with_weight_kernel[grid_out](
            x, w, inv_rms, out, B, H,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            BLOCK_SIZE=BLOCK_SIZE, VEC_ROWS=VEC_ROWS, NUM_ITERS=NUM_ITERS, OUT_DTYPE=out_dtype,
            num_warps=8, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
