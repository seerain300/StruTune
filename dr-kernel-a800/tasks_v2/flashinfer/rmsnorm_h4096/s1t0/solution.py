import torch
import triton
import triton.language as tl


@triton.jit
def _compute_inv_rms_kernel(
    x_ptr,         # *pointer to hidden_states
    inv_rms_ptr,   # *pointer to per-row inv_rms (float32)
    B,             # batch size (rows)
    H,             # hidden size (columns)
    EPS,           # epsilon (float32)
    stride_x_row,  # stride for row in x
    stride_x_col,  # stride for col in x
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)

    sumsq = 0.0
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE

    mean = sumsq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)


@triton.jit
def _apply_scale_kernel(
    x_ptr,           # *pointer to hidden_states
    weight_ptr,      # *pointer to weight
    inv_rms_ptr,     # *pointer to per-row inv_rms (float32)
    out_ptr,         # *pointer to output
    B,               # batch size
    H,               # hidden size
    stride_x_row,    # stride for row in x
    stride_x_col,    # stride for col in x
    stride_out_row,  # stride for row in out
    stride_out_col,  # stride for col in out
    OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 or tl.bfloat16)
    BLOCK_SIZE: tl.constexpr, # chunk size along columns
):
    # One program per row
    row = tl.program_id(0)
    # Load inv_rms for this row (float32)
    inv_rms = tl.load(inv_rms_ptr + row)

    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        w = w.to(tl.float32)
        y = x * inv_rms * w
        # Cast to desired output dtype before storing
        if OUT_DTYPE == tl.float16:
            y = y.to(tl.float16)
        elif OUT_DTYPE == tl.bfloat16:
            y = y.to(tl.bfloat16)
        tl.store(out_ptr + row * stride_out_row + offs * stride_out_col, y, mask=mask)
        start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if hidden_states.device.type != "cuda":
            raise RuntimeError("ModelNew expects CUDA tensors for Triton kernel.")
        if weight.device.type != "cuda":
            raise RuntimeError("ModelNew expects CUDA tensors for Triton kernel.")

        # Ensure shapes
        if hidden_states.dim() != 2:
            raise RuntimeError("hidden_states must be 2D (B, H).")
        if weight.dim() != 1 or weight.numel() != hidden_states.shape[1]:
            raise RuntimeError("weight must be 1D with length equal to hidden_states.shape[1].")

        # Make inputs contiguous
        x = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = x.shape

        # Output tensor same shape and dtype as input
        out = torch.empty_like(x)

        # Strides (in elements)
        stride_x_row = x.stride(0)
        stride_x_col = x.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Choose dtype mapping for output (compute in fp32, cast on store)
        if x.dtype == torch.float16:
            out_dtype = tl.float16
        elif x.dtype == torch.bfloat16:
            out_dtype = tl.bfloat16
        else:
            # Default to bfloat16; adjust if other dtypes are expected
            out_dtype = tl.bfloat16


def run(*args):
    return ModelNew()(*args)
