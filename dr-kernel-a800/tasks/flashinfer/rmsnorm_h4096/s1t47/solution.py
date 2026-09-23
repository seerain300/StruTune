import torch
import triton
import triton.language as tl


@triton.jit
def _compute_inv_rms_rowwise_kernel(
    x_ptr,            # *pointer to hidden_states (float32, contiguous, row-major, shape [B, H] flattened)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    inv_rms_ptr,      # *pointer to output inv_rms (float32, shape [B])
    BLOCK_SIZE: tl.constexpr,  # chunk size along columns
    VEC: tl.constexpr,         # iterations per pass (columns per iteration = BLOCK_SIZE * VEC)
    NUM_ITERS: tl.constexpr,   # iterations = ceil_div(H, BLOCK_SIZE * VEC)
):
    # One program per row
    row_id = tl.program_id(0)
    # Accumulator for sum of squares for this row
    sum_sq = 0.0
    # Loop over column tiles; x is laid out as contiguous [B, H] flattened:
    # base offset for row: row_id * H
    for it in range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        # Base pointer for this row
        x_row_base = x_ptr + row_id * H
        x_vals = tl.load(x_row_base + cols, mask=mask, other=0.0)
        # Square and reduce
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    # Compute mean and inv_rms
    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + EPS)
    # Store inv_rms for this row
    tl.store(inv_rms_ptr + row_id, inv_rms)


@triton.jit
def _apply_scale_rowwise_kernel(
    x_ptr,            # *pointer to hidden_states (float32, contiguous, row-major, shape [B, H] flattened)
    weight_ptr,       # *pointer to weight (float32, contiguous, 1D of length H)
    out_ptr,          # *pointer to output (float32, contiguous, shape [B, H] flattened)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    inv_rms_ptr,      # *pointer to inv_rms (float32, shape [B])
    BLOCK_SIZE: tl.constexpr,  # chunk size along columns
    VEC: tl.constexpr,         # iterations per pass (columns per iteration = BLOCK_SIZE * VEC)
    NUM_ITERS: tl.constexpr,   # iterations = ceil_div(H, BLOCK_SIZE * VEC)
):
    # One program per row
    row_id = tl.program_id(0)
    inv_rms = tl.load(inv_rms_ptr + row_id)  # float32 scalar
    # Output is laid out as contiguous [B, H] flattened
    out_row_base = out_ptr + row_id * H
    # Process columns in tiles
    for it in range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        # Load x[row, cols] and weight[cols]
        x_row_base = x_ptr + row_id * H
        x_vals = tl.load(x_row_base + cols, mask=mask, other=0.0)
        w_vals = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        # Compute y = x * inv_rms * w
        y_vals = x_vals * inv_rms * w_vals
        # Store to out[row, cols]
        tl.store(out_row_base + cols, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are CUDA tensors and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        w = weight.contiguous()

        # Compute in float32 for numerical stability
        x = hidden.to(torch.float32)  # shape [B, H], contiguous
        w_f32 = w.to(torch.float32)   # shape [H], contiguous

        B, H = x.shape
        EPS = 1e-5

        # Prepare output (float32) and inv_rms buffer (float32)
        out = torch.empty((B, H), dtype=torch.float32, device=x.device)  # contiguous
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

        # Tile configuration that previously delivered best speed:
        # process 8192 columns per iteration for H=4096 => NUM_ITERS=2
        BLOCK_SIZE = 512
        VEC = 16
        NUM_ITERS = (H + BLOCK_SIZE * VEC - 1) // (BLOCK_SIZE * VEC)

        # Launch kernel to compute per-row inv_rms
        grid = (B,)
        _compute_inv_rms_rowwise_kernel[grid](
            x, B, H, EPS, inv_rms,
            BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS,
            num_warps=8, num_stages=2
        )

        # Launch kernel to apply scaling and write output
        _apply_scale_rowwise_kernel[grid](
            x, w_f32, out, B, H, inv_rms,
            BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS,
            num_warps=8, num_stages=2
        )

        # Cast back to original dtype for the output
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
