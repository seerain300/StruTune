import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Fused Triton kernel: per-row normalization then elementwise scaling by weight.
# Each program handles one row (batch element). It loops over columns in tiles of BLOCK_SIZE:
# - First pass: compute sum of squares for the row.
# - Compute inv_rms = rsqrt(mean + EPS).
# - Second pass: scale and write results y = x * inv_rms * w.
@triton.jit
def fused_row_norm_scale(x_ptr, w_ptr, out_ptr, H: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    r = tl.program_id(0)  # row index
    sumsq = 0.0
    # First pass: reduction over the row
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_ptr + r * H + offs, mask=mask, other=0.0)
        x2 = x * x
        sumsq += tl.sum(x2, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)  # scalar for this row

    # Second pass: scale and write
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_ptr + r * H + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(out_ptr + r * H + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Fallback to original PyTorch if Triton or CUDA is not available
        if (not TRITON_AVAILABLE) or (hidden_states.device.type != "cuda"):
            batch_size, hidden_size = hidden_states.shape
            # Keep the original assert for safety in fallback path
            assert hidden_size == 4096, "hidden_size must be 4096 in this implementation"
            EPS = 1e-5
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure contiguity and dtype, compute in fp32
        x = hidden_states.contiguous()
        w = weight.contiguous()

        # Cast to float32 for compute; weight is 1D [H]
        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)

        B, H = x_fp32.shape
        # Sanity check: weight length must match hidden dimension
        if w_fp32.numel() != H:
            raise ValueError(f"weight length {w_fp32.numel()} must match hidden_size {H}")

        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=x_fp32.device)

        # Kernel launch configuration:
        # For H=4096, using BLOCK_SIZE=2048 reduces loop iterations to 2 in both passes.
        # Use num_warps=8 for better throughput with larger tiles.
        BLOCK_SIZE = 2048
        num_warps = 8

        # Launch one program per row
        fused_row_norm_scale[(B,)](
            x_fp32, w_fp32, out_fp32,
            H=H, EPS=1e-5, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps, num_stages=2
        )

        # Cast back to original dtype
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
