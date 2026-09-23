import torch
import triton
import triton.language as tl


@triton.jit
def _fused_norm_scale_row_kernel(
    x_ptr,            # *pointer to hidden_states (float32)
    weight_ptr,       # *pointer to weight (float32)
    out_ptr,          # *pointer to output (float32)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    stride_out_row,   # stride for row in out
    stride_out_col,   # stride for col in out
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
    for it in range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        x_chunk = tl.load(row_x_ptr + cols * stride_x_col, mask=mask, other=0.0)
        sumsq += tl.sum(x_chunk * x_chunk, axis=0)

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)  # 1 / sqrt(mean + EPS)

    # 2) Second pass: compute y = x * inv_rms * weight and store
    for it in range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        x_chunk = tl.load(row_x_ptr + cols * stride_x_col, mask=mask, other=0.0)
        w_chunk = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        out_chunk = x_chunk * inv_rms * w_chunk
        tl.store(row_out_ptr + cols * stride_out_col, out_chunk, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # If tensors are not CUDA, fall back to PyTorch implementation to avoid crashes
        if not (hidden_states.is_cuda and weight.is_cuda):
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure contiguous tensors and compute in float32
        x = hidden_states.to(torch.float32).contiguous()
        weight = weight.to(torch.float32).contiguous()

        B, H = x.shape
        out = torch.empty_like(x)  # compute in FP32

        # Tiling parameters known to be performant and correct
        BLOCK_SIZE = 512
        VEC = 16
        NUM_ITERS = (H + (BLOCK_SIZE * VEC) - 1) // (BLOCK_SIZE * VEC)  # e.g., 1 for H=4096

        grid = (B,)

        _fused_norm_scale_row_kernel[grid](
            x, weight, out,
            B, H, 1e-5,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            VEC=VEC,
            NUM_ITERS=NUM_ITERS,
            num_warps=8, num_stages=2,
        )

        # Cast back to original dtype to match the original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
