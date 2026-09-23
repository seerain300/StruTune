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
    # Base pointers for this row
    row_x_ptr = x_ptr + row_id * stride_x_row
    row_out_ptr = out_ptr + row_id * stride_out_row

    # 1) First pass: compute sum of squares across all columns in this row
    sumsq = 0.0  # float32 accumulator
    for it in tl.static_range(NUM_ITERS):
        col_start = it * VEC
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Load x row segment
        x = tl.load(row_x_ptr + offs * stride_x_col, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sumsq += tl.sum(x_f32 * x_f32)

    # Compute inv_rms for this row
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)  # 1 / sqrt(mean + EPS)

    # 2) Second pass: compute and store y = x * inv_rms * weight
    for it in tl.static_range(NUM_ITERS):
        col_start = it * VEC
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Load x segment (row) and weight segment
        x = tl.load(row_x_ptr + offs * stride_x_col, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        y = x_f32 * inv_rms * w  # float32 compute

        # Cast to desired output dtype and store
        if OUT_DTYPE == tl.float16:
            y = y.to(tl.float16)
        else:
            # tl.bfloat16
            y = y.to(tl.bfloat16)
        tl.store(row_out_ptr + offs * stride_out_col, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Expect hidden_states: [B, H], weight: [H]
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        # Work in FP32 for compute
        x = hidden_states.to(torch.float32)
        w = weight.to(torch.float32)
        B, H = x.shape
        out = torch.empty_like(x)  # FP32 output buffer

        # Choose tiling: per-iteration work 32768 cols to keep loops minimal for typical H
        BLOCK_SIZE = 2048
        VEC = 16  # columns per iteration: 2048 * 16 = 32768
        NUM_ITERS = (H + VEC - 1) // VEC  # compile-time constant via meta

        # Launch one program per row
        grid = (B,)
        _fused_norm_scale_row_kernel[grid](
            x, w, out,
            B, H, 1e-5,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            OUT_DTYPE=tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16,
            BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS,
            num_warps=8, num_stages=2
        )

        # Cast back to original dtype if necessary
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
