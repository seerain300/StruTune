import torch
import triton
import triton.language as tl

# Fixed hidden size per the original assertion
HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _normalize_scale_row_kernel(hidden_ptr, weight_ptr, out_ptr,
                                B: tl.constexpr, H: tl.constexpr,
                                BLOCK_SIZE: tl.constexpr):
    """
    One Triton program per row. Processes a single row of length H.

    - hidden_ptr: *ptr to [B, H] (row-major contiguous)
    - weight_ptr: *ptr to [H]
    - out_ptr: *ptr to [B, H] (float32 output buffer)
    """
    row = tl.program_id(0)
    # Base offset for this row in a row-major [B, H] tensor
    base = row * H

    # Vectorized offsets across the row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # mask for safety, since BLOCK_SIZE==HIDDEN_SIZE, it's true for all lanes

    # Load entire row (masked), cast to float32
    x = tl.load(hidden_ptr + base + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)

    # Ensure masked lanes do not contribute to sum
    x32 = tl.where(mask, x32, 0.0)

    # Compute sum of squares in FP32
    sumsq = tl.sum(x32 * x32, axis=0)

    # Compute inverse RMS per row: rsqrt(mean + EPS)
    inv_rms = tl.rsqrt(sumsq / H + EPS)

    # Load weight vector, cast to FP32
    w = tl.load(weight_ptr + offs, mask=mask, other=0).to(tl.float32)

    # Compute output: y = (x * inv_rms) * w
    y32 = x32 * inv_rms * w

    # Store FP32 output for this row
    tl.store(out_ptr + row * H + offs, y32, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        # Output buffer in FP32 for numerical stability; cast back after kernel
        out = torch.empty((B, H), device=hidden.device, dtype=torch.float32)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _normalize_scale_row_kernel[grid](
            hidden, weight, out,
            B=B, H=H,                      # constexpr meta-parameters
            BLOCK_SIZE=HIDDEN_SIZE,       # constexpr inside the kernel
            num_warps=4,
            num_stages=2,
        )

        # Cast to original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
