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
    NUM_ITERS: tl.constexpr,  # number of iterations over columns
):
    # Each program handles one row
    row_id = tl.program_id(0)
    row_x_ptr = x_ptr + row_id * stride_x_row
    row_out_ptr = out_ptr + row_id * stride_out_row

    # 1) First pass: compute sum of squares across all columns in this row
    sumsq = 0.0
    for it in tl.static_range(NUM_ITERS):
        offs = it * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(row_x_ptr + offs * stride_x_col, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sumsq += tl.sum(x32 * x32, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)  # scalar per row

    # 2) Second pass: compute output y = x * inv_rms * weight and store
    for it in tl.static_range(NUM_ITERS):
        cols = it * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < H

        # Load x for this row chunk
        x = tl.load(row_x_ptr + cols * stride_x_col, mask=col_mask, other=0.0).to(tl.float32)

        # Load weight for this chunk (broadcast across columns)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        # Compute output
        y = x * inv_rms * w

        # Cast to requested output dtype
        if OUT_DTYPE == tl.bfloat16:
            y = y.to(tl.bfloat16)
        elif OUT_DTYPE == tl.float16:
            y = y.to(tl.float16)

        # Store output
        tl.store(row_out_ptr + cols * stride_out_col, y, mask=col_mask)


def _choose_params(H):
    # Heuristic: large tile to minimize loop iterations while keeping good occupancy.
    # For H >= 4096, use 1024 x 64. For smaller H, fall back to 512 x 64.
    if H >= 4096:
        BLOCK_SIZE = 1024
        VEC = 64
    else:
        BLOCK_SIZE = 512
        VEC = 64
    NUM_ITERS = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    return BLOCK_SIZE, VEC, NUM_ITERS


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Expect 2D hidden_states [B, H] and 1D weight [H]
        assert hidden_states.dim() == 2, "hidden_states must be 2D [B, H]"
        assert weight.dim() == 1, "weight must be 1D [H]"
        B, H = hidden_states.shape
        assert H == 4096, "This optimized Triton kernel expects hidden_size == 4096"

        # Ensure contiguous tensors for coalesced memory access
        x = hidden_states.contiguous()
        w = weight.contiguous()

        # Output tensor in the same dtype/shape as hidden_states
        out = torch.empty_like(x)

        # Compute strides (in elements)
        stride_x_row = x.stride(0)
        stride_x_col = x.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Select dtype for output
        OUT_DTYPE = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
        BLOCK_SIZE, VEC, NUM_ITERS = _choose_params(H)

        # Launch one program per row
        grid = (B,)
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
