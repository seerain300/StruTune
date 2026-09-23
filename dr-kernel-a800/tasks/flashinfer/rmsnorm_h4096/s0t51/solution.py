import torch
import triton
import triton.language as tl

# Fused kernel: per-row reduction and scaling in a single pass over tiles.
# One Triton program handles one row. It loops over the row in tiles of BLOCK_SIZE.
# It first accumulates sum of squares to compute inv_rms, then re-reads the row
# and writes the scaled output.
@triton.jit
def row_norm_and_scale(
    x_ptr,            # *f32, shape [B, H]
    weight_ptr,       # *f32, shape [H]
    out_ptr,          # *f32, shape [B, H]
    B: tl.constexpr,  # batch size (not used directly, but can be used for grid)
    H: tl.constexpr,  # hidden size
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)  # one program per row
    row_offset = row_id * H

    # Accumulator for sum of squares
    acc = 0.0

    # First pass: compute sumsq over the row
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)

    mean = acc / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Second pass: scale and write output y = x * inv_rms * weight
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(out_ptr + row_offset + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are 2D and on same device
        assert hidden_states.dim() == 2, "hidden_states must be [batch, hidden_size]"
        B, H = hidden_states.shape

        # Cast to float32 for numeric stability; ensure contiguous
        x = hidden_states.contiguous().to(torch.float32)  # [B, H]
        weight_fp32 = weight.contiguous().to(torch.float32)  # [H]

        # Output buffer in fp32
        out = torch.empty((B, H), dtype=torch.float32, device=x.device)

        # Tuned parameters: use BLOCK_SIZE=1024 to balance occupancy and register usage
        BLOCK_SIZE = 1024
        grid = (B,)
        EPS = 1e-5

        row_norm_and_scale[grid](
            x,
            weight_fp32,
            out,
            B=B,
            H=H,
            EPS=EPS,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,   # reasonable default for 1024-wide tiles
            num_stages=3,  # pipeline stages for better throughput
        )

        # Return in original dtype
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
