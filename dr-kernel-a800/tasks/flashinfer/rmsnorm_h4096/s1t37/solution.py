import torch
import triton
import triton.language as tl


@triton.jit
def _fused_norm_scale_row_kernel(
    x_ptr,            # *pointer to hidden_states (fp32)
    weight_ptr,       # *pointer to weight (fp32)
    out_ptr,          # *pointer to output (original dtype: fp16 or bf16)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x (elements)
    stride_x_col,     # stride for col in x (elements)
    stride_out_row,   # stride for row in out (elements)
    stride_out_col,   # stride for col in out (elements)
    OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 or tl.bfloat16)
    BLOCK_SIZE: tl.constexpr, # base chunk size along columns
    VEC: tl.constexpr,        # columns processed per iteration
    NUM_ITERS: tl.constexpr,  # number of iterations = ceil_div(H, BLOCK_SIZE * VEC)
):
    # One program per row
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    # First pass: compute sum of squares across columns for this row
    sum_sq = tl.zeros((), dtype=tl.float32)
    for it in tl.static_range(NUM_ITERS):
        base = it * (BLOCK_SIZE * VEC)
        cols = base + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H

        x_row_ptr = x_ptr + row_id * stride_x_row + cols * stride_x_col
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)  # already fp32
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean_sq = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean_sq + EPS)  # scalar per row

    # Second pass: compute outputs and store in original dtype
    for it in tl.static_range(NUM_ITERS):
        base = it * (BLOCK_SIZE * VEC)
        cols = base + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H

        x_row_ptr = x_ptr + row_id * stride_x_row + cols * stride_x_col
        w_ptr = weight_ptr + cols  # weight is 1D contiguous (fp32)

        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)  # fp32
        w_vals = tl.load(w_ptr, mask=mask, other=0.0)      # fp32
        out_vals = x_vals * inv_rms * w_vals              # fp32 compute

        out_row_ptr = out_ptr + row_id * stride_out_row + cols * stride_out_col
        if OUT_DTYPE == tl.float16:
            tl.store(out_row_ptr, out_vals.to(tl.float16), mask=mask)
        else:
            tl.store(out_row_ptr, out_vals.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguous
        if not hidden_states.is_cuda or not weight.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensors for hidden_states and weight.")
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape
        # Promote to fp32 for computation
        x = hidden_states.to(torch.float32)
        w = weight.to(torch.float32)

        # Output in original dtype
        out = torch.empty_like(hidden_states)

        # Strides (in elements)
        stride_x_row = x.stride(0)
        stride_x_col = x.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Tile parameters known to perform well
        BLOCK_SIZE = 512
        VEC = 16
        NUM_ITERS = (H + (BLOCK_SIZE * VEC) - 1) // (BLOCK_SIZE * VEC)

        OUT_DTYPE = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16

        grid = (B,)
        _fused_norm_scale_row_kernel[grid](
            x, w, out,
            B, H, 1e-5,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            OUT_DTYPE,
            BLOCK_SIZE=BLOCK_SIZE,
            VEC=VEC,
            NUM_ITERS=NUM_ITERS,
            num_warps=8,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
