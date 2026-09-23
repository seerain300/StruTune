import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _normalize_scale_row_kernel(hidden_ptr, weight_ptr, out_ptr,
                                B, H,
                                BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)

    # First pass: compute sum of squares across the row in chunks
    sumsq = 0.0
    for c in range(0, H, BLOCK_SIZE):
        offs = c + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        row_base = hidden_ptr + row * H
        x = tl.load(row_base + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sumsq += tl.sum(x32 * x32, axis=0)

    # Compute inv_rms: rsqrt(mean + EPS)
    mean_sq = sumsq / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Second pass: compute output for the row in chunks
    for c in range(0, H, BLOCK_SIZE):
        offs = c + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        row_base = hidden_ptr + row * H
        x = tl.load(row_base + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x32 * inv_rms * w
        tl.store(out_ptr + row * H + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == HIDDEN_SIZE, f"Expected hidden size {HIDDEN_SIZE}, got {H}"

        # Output as FP32 in kernel, cast later
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _normalize_scale_row_kernel[grid](
            hidden, weight, out_fp32,
            B, H,
            BLOCK_SIZE=128,  # compile-time chunk size; loops over 0..31 steps
            num_warps=4,     # reasonable default; kernel is simple per row
        )

        # Cast back to original dtype to match PyTorch behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
