import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_and_scale_row(hidden_ptr, weight_ptr, out_ptr,
                            stride_hs, stride_out):
    # One program per row
    row = tl.program_id(0)
    base_hs = hidden_ptr + row * stride_hs
    base_out = out_ptr + row * stride_out

    # Column offsets (compile-time constant 4096)
    offs = tl.arange(0, HIDDEN_SIZE)
    mask = offs < HIDDEN_SIZE

    # Load row (masked), cast to float32
    x = tl.load(base_hs + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    # Ensure masked lanes do not contribute to sum
    x32 = tl.where(mask, x32, 0.0)

    # Compute sum of squares in FP32
    sumsq = tl.sum(x32 * x32, axis=0)

    # Compute inverse RMS per row: rsqrt(mean + EPS)
    inv_rms = tl.rsqrt(sumsq / HIDDEN_SIZE + EPS)

    # Load weight vector (masked), cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute output: y = x32 * inv_rms * w (FP32)
    y32 = x32 * inv_rms * w

    # Store result (masked)
    tl.store(base_out + offs, y32, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous inputs
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        # Allocate FP32 output
        B, H = hidden.shape
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        normalize_and_scale_row[grid](
            hidden, weight, out_fp32,
            hidden.stride(0), out_fp32.stride(0),
            num_warps=4, num_stages=2
        )

        # Cast to original dtype (match original behavior)
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
