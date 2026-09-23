import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_scale_row(hidden_ptr, weight_ptr, out_ptr,
                        B, H, stride_hs, stride_out, EPS, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)  # compile-time constant vector for columns
    mask = offs < H  # safety mask; with BLOCK_SIZE=H, all True

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row * stride_hs
    out_row_ptr = out_ptr + row * stride_out

    # Load the entire row and cast to float32
    x = tl.load(hidden_row_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x32 * x32, axis=0)  # scalar float32

    # Compute inverse RMS
    inv_rms = tl.rsqrt(sumsq / H + EPS)  # scalar

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    w32 = w.to(tl.float32)

    # Compute final output: y = x * inv_rms * w
    y32 = x32 * inv_rms * w32

    # Store result
    tl.store(out_row_ptr + offs, y32, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if not on CUDA
        if not hidden_states.is_cuda:
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure inputs are contiguous
        hidden = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = hidden.shape
        assert H == HIDDEN_SIZE, "hidden_size must be 4096"

        # Allocate FP32 output buffer
        out_fp32 = torch.empty((B, H), device=hidden.device, dtype=torch.float32)

        # Compute strides in elements
        stride_hs = hidden.stride(0)
        stride_out = out_fp32.stride(0)

        # Launch Triton kernel: one program per row
        grid = (B,)
        normalize_scale_row[grid](
            hidden, w, out_fp32,
            B, H, stride_hs, stride_out, EPS,
            BLOCK_SIZE=HIDDEN_SIZE,  # tl.constexpr meta-parameter
            num_warps=8,              # suitable for 4096-wide vector
            num_stages=1
        )

        # Cast to original dtype to match original behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
