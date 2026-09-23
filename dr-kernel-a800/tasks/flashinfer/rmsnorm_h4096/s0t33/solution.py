import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None


# Triton kernels for reduction and scaling
if triton is not None:

    @triton.jit
    def reduce_row_sumsq_kernel(
        x_ptr,          # *fp32
        inv_rms_ptr,    # *fp32
        B,              # int
        H,              # int
        EPS,            # fp32
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row
        row_id = tl.program_id(axis=0)
        # Guard in case grid > B
        if row_id >= B:
            return

        # Accumulator for sum of squares (fp32)
        sumsq = 0.0

        # Iterate over columns in tiles
        for col_start in range(0, H, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < H

            # Row-major offset: row_id * H + cols
            x_row_ptr = x_ptr + row_id * H + cols
            x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
            x_sq = x_vals * x_vals
            # Reduce this tile into a scalar
            tile_sum = tl.sum(x_sq, axis=0)
            sumsq += tile_sum

        # Compute inv_rms[row] = rsqrt(sumsq / H + EPS)
        mean = sumsq / H
        inv_rms = tl.rsqrt(mean + EPS)

        # Store per-row result
        tl.store(inv_rms_ptr + row_id, inv_rms)


    @triton.jit
    def scale_row_elements_kernel(
        x_ptr,          # *fp32
        weight_ptr,     # *fp32
        y_ptr,          # *fp32
        inv_rms_ptr,    # *fp32
        B,              # int
        H,              # int
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row
        row_id = tl.program_id(axis=0)
        if row_id >= B:
            return

        # Load inv_rms[row]
        inv_rms = tl.load(inv_rms_ptr + row_id)

        # Iterate over columns in tiles
        for col_start in range(0, H, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < H

            x_row_ptr = x_ptr + row_id * H + cols
            w_ptr = weight_ptr + cols

            x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
            w_vals = tl.load(w_ptr, mask=mask, other=0.0)

            y_vals = x_vals * inv_rms * w_vals
            tl.store(y_ptr + row_id * H + cols, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        # Cast to fp32 for compute
        x = hidden_states.to(torch.float32)
        weight_f32 = weight.to(torch.float32)

        batch_size, hidden_size = x.shape
        assert hidden_size == 4096, "This optimized implementation expects hidden_size == 4096."

        # Output buffer (fp32 for compute)
        y = torch.empty_like(x)

        # Per-row inverse RMS (fp32)
        inv_rms = torch.empty(batch_size, device=x.device, dtype=torch.float32)

        # Launch reduction kernel: one program per row
        # BLOCK_SIZE tuned for H=4096; 2048 -> 2 iterations; adjust warps/stages for throughput
        BLOCK_SIZE = 2048
        grid = (batch_size,)
        reduce_row_sumsq_kernel[grid](
            x, inv_rms, batch_size, hidden_size, 1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )

        # Launch scaling kernel: one program per row
        scale_row_elements_kernel[grid](
            x, weight_f32, y, inv_rms, batch_size, hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )

        # Cast back to original dtype
        return y.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
