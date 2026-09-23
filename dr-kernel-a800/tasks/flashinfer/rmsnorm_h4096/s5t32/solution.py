import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_and_scale_row(hidden_ptr, weight_ptr, out_ptr,
                            stride_hs, stride_out,
                            BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    base_hs = hidden_ptr + row * stride_hs
    base_out = out_ptr + row * stride_out

    # Vectorized column offsets (compile-time constant 4096)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < BLOCK_SIZE  # always true for BLOCK_SIZE == HIDDEN_SIZE, but keep for safety

    # Load entire row (masked), cast to float32
    x = tl.load(base_hs + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    # Ensure masked lanes do not contribute to sum
    x32 = tl.where(mask, x32, 0.0)

    # Compute sum of squares in FP32
    sumsq = tl.sum(x32 * x32, axis=0)

    # Compute inverse RMS per row: rsqrt(mean + EPS)
    inv_rms = tl.rsqrt(sumsq / BLOCK_SIZE + EPS)

    # Load weight vector (masked), cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute output: y = x * inv_rms * w (FP32)
    y = x32 * inv_rms * w

    # Store FP32 output with mask
    tl.store(base_out + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        hidden = hidden_states.contiguous()
        w = weight.contiguous()
        B, H = hidden.shape
        assert H == HIDDEN_SIZE, f"Expected hidden size {HIDDEN_SIZE}, got {H}"

        # Output buffer in FP32 for computation
        out_fp32 = torch.empty((B, HIDDEN_SIZE), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row
        normalize_and_scale_row[(B,)](
            hidden, w, out_fp32,
            hidden.stride(0), out_fp32.stride(0),
            BLOCK_SIZE=HIDDEN_SIZE,
            num_warps=4, num_stages=2
        )

        # Cast to original dtype to match PyTorch behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
