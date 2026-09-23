import torch
import triton
import triton.language as tl

# Kernel 1: per-row reduction of sum of squares
@triton.jit
def reduce_row_sumsq_kernel(
    x_ptr,             # *fp32, shape [B, H]
    inv_rms_ptr,       # *fp32, shape [B]
    B: tl.constexpr,   # number of rows
    H: tl.constexpr,   # hidden size (columns)
    EPS: tl.constexpr, # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    sumsq = 0.0
    # For H == BLOCK_SIZE (4096), this loop runs once; for general H it tiles over columns.
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_row_ptr = x_ptr + row * H + cols
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
        sumsq += tl.sum(x_vals * x_vals, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)


# Kernel 2: elementwise scaling: y[row, col] = x[row, col] * inv_rms[row] * weight[col]
@triton.jit
def scale_row_elements_kernel(
    x_ptr,              # *fp32, shape [B, H]
    weight_ptr,         # *fp32, shape [H]
    inv_rms_ptr,        # *fp32, shape [B]
    out_ptr,            # *fp32, shape [B, H]
    B: tl.constexpr,    # number of rows
    H: tl.constexpr,    # hidden size
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return
    inv_r = tl.load(inv_rms_ptr + row)  # scalar per row
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_row_ptr = x_ptr + row * H + cols
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
        w_vals = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        f = inv_r * w_vals  # per-column factor
        y_vals = x_vals * f
        out_row_ptr = out_ptr + row * H + cols
        tl.store(out_row_ptr, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Expect 2D hidden states and 1D weight
        assert hidden_states.dim() == 2, "hidden_states must be 2D [B, H]"
        assert weight.dim() == 1, "weight must be 1D [H]"
        B, H = hidden_states.shape
        # Optimized path assumes hidden_size == 4096 (matches provided get_inputs)
        assert H == 4096, "This optimized Triton path assumes hidden_size == 4096"

        # Ensure contiguous and compute in fp32
        x = hidden_states.contiguous().to(torch.float32)  # [B, H]
        weight_fp32 = weight.contiguous().to(torch.float32)  # [H]

        # Allocate output buffer
        out = torch.empty((B, H), dtype=torch.float32, device=x.device)
        # Buffer for per-row inv_rms
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

        EPS = 1e-5

        # Launch reduction kernel: one program per row
        grid = (B,)
        reduce_row_sumsq_kernel[grid](
            x,
            inv_rms,
            B=B,
            H=H,
            EPS=EPS,
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=4,
        )

        # Launch scaling kernel: one program per row
        scale_row_elements_kernel[grid](
            x,
            weight_fp32,
            inv_rms,
            out,
            B=B,
            H=H,
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=4,
        )

        # Cast back to original dtype of hidden_states
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
