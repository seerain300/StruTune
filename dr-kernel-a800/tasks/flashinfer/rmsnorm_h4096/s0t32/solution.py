import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None


# Triton kernels for reduction and scaling
if triton is not None:

    @triton.jit
    def reduce_row_sumsq_kernel(x_ptr, inv_rms_ptr, H, EPS, NUM_TILES: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        # One program per row
        row = tl.program_id(0)
        # Accumulator for sum of squares (fp32)
        acc = 0.0
        # Loop over tiles
        for t in range(NUM_TILES):
            col_start = t * BLOCK_SIZE
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < H
            # Load x[row, cols] as fp32
            x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
            sq = x * x
            # Masked reduction: sum sq over valid columns
            acc += tl.sum(sq, axis=0)
        # Compute inv_rms = 1 / sqrt(mean + EPS) where mean = acc / H
        mean = acc / H
        inv_rms = tl.rsqrt(mean + EPS)
        # Store per-row inv_rms (fp32)
        tl.store(inv_rms_ptr + row, inv_rms)

    @triton.jit
    def scale_row_elements_kernel(x_ptr, weight_ptr, inv_rms_ptr, y_ptr,
                                   B, H, NUM_TILES: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        # One program per row
        row = tl.program_id(0)
        # Load inv_rms[row]
        inv_rms = tl.load(inv_rms_ptr + row)
        # Iterate over columns in tiles
        for t in range(NUM_TILES):
            col_start = t * BLOCK_SIZE
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < H
            # Load x[row, cols] as fp32
            x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
            # Load weight[cols] as fp32
            w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            # Compute y = x * inv_rms * w
            y = x * inv_rms * w
            tl.store(y_ptr + row * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels"
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape
        EPS = 1e-5

        # Compute in fp32 for numerical stability
        x_fp32 = hidden_states.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)

        # Allocate outputs as fp32, then cast back to original dtype at the end
        y_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)

        # Prepare per-row inv_rms buffer (fp32), one value per row
        inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden_states.device)

        # Choose tile size; for H=4096, 2048 gives two tiles; use masked loops for generality
        BLOCK_SIZE = 2048
        NUM_TILES = (H + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Launch reduction kernel: one program per row
        grid = (B,)
        reduce_row_sumsq_kernel[grid](
            x_fp32, inv_rms, H, EPS, NUM_TILES, BLOCK_SIZE,
            num_warps=8, num_stages=2
        )

        # Launch scaling kernel: one program per row
        scale_row_elements_kernel[grid](
            x_fp32, weight_fp32, inv_rms, y_fp32,
            B, H, NUM_TILES, BLOCK_SIZE,
            num_warps=8, num_stages=2
        )

        # Cast back to original dtype of hidden_states
        return y_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
