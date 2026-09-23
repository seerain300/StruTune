import torch
import triton
import triton.language as tl


@triton.jit
def _fused_norm_scale_row_kernel(
    x_ptr,            # *pointer to hidden_states
    weight_ptr,       # *pointer to weight
    out_ptr,          # *pointer to output
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    stride_out_row,   # stride for row in out
    stride_out_col,   # stride for col in out
    OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 or tl.bfloat16)
    BLOCK_SIZE: tl.constexpr, # base chunk size along columns
    VEC: tl.constexpr,        # columns processed per iteration
    NUM_ITERS: tl.constexpr,  # number of iterations to cover H
):
    # One program per row
    row_id = tl.program_id(0)

    # First pass: compute sum of squares for this row and derive inv_rms
    sumsq = 0.0
    for it in tl.static_range(NUM_ITERS):
        col_offsets = it * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]  # shape (BLOCK_SIZE, 1)
        col_vec = tl.arange(0, VEC)[None, :]                                # shape (1, VEC)
        offs = col_offsets * VEC + col_vec                                  # broadcast to (BLOCK_SIZE, VEC)
        mask = offs < H

        x_idx = row_id * stride_x_row + offs * stride_x_col
        out_idx = row_id * stride_out_row + offs * stride_out_col

        x_vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x_vals * x_vals, axis=0)

    mean = sumsq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)  # scalar per row

    # Second pass: compute and store outputs
    for it in tl.static_range(NUM_ITERS):
        col_offsets = it * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
        col_vec = tl.arange(0, VEC)[None, :]
        offs = col_offsets * VEC + col_vec
        mask = offs < H

        x_idx = row_id * stride_x_row + offs * stride_x_col
        out_idx = row_id * stride_out_row + offs * stride_out_col

        x_vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        y_vals = x_vals * inv_rms * w_vals  # float32 compute

        # Cast to output dtype
        if OUT_DTYPE == tl.float16:
            y_cast = y_vals.to(tl.float16)
        elif OUT_DTYPE == tl.bfloat16:
            y_cast = y_vals.to(tl.bfloat16)
        else:
            y_cast = y_vals  # default float32

        tl.store(out_ptr + out_idx, y_cast, mask=mask)


def _dtype_code(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


def _grid_1d(B: int):
    return (B,)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # CPU fallback (kept minimal; Triton is used on CUDA)
        if not hidden_states.is_cuda or not weight.is_cuda:
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure contiguous for predictable strides and coalesced access
        x = hidden_states.contiguous()
        w = weight.contiguous()
        B, H = x.shape
        assert w.numel() == H, "Weight must have length equal to hidden size."

        out = torch.empty_like(x)

        stride_x_row = x.stride(0)
        stride_x_col = x.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Tile configuration: cover 4096 columns in a single iteration for the common case
        BLOCK_SIZE = 512   # columns per base chunk
        VEC = 8             # vectors per iteration
        NUM_ITERS = (H + BLOCK_SIZE * VEC - 1) // (BLOCK_SIZE * VEC)  # typically 1 for H=4096

        grid = _grid_1d(B)
        OUT_DTYPE = _dtype_code(x.dtype)

        _fused_norm_scale_row_kernel[grid](
            x, w, out,
            B, H, 1e-5,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            OUT_DTYPE=OUT_DTYPE,
            BLOCK_SIZE=BLOCK_SIZE,
            VEC=VEC,
            NUM_ITERS=NUM_ITERS,
            num_warps=8,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
