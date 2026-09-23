import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_and_scale_row(hidden_ptr, weight_ptr, out_ptr,
                            stride_hs, stride_out,
                            B, H):
    # One program per row
    row = tl.program_id(0)
    base_hs = hidden_ptr + row * stride_hs
    base_out = out_ptr + row * stride_out

    # Vectorized column offsets (compile-time constant 4096)
    offs = tl.arange(0, HIDDEN_SIZE)
    mask = offs < H  # robust masking (H == 4096 in our case)

    # Load hidden row, cast to float32
    x = tl.load(base_hs + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)

    # Zero-out masked lanes to ensure they don't contribute
    x32 = tl.where(mask, x32, 0.0)

    # Compute sum of squares in FP32
    sumsq = tl.sum(x32 * x32, axis=0)

    # Compute inverse RMS per row: rsqrt(mean + EPS)
    inv_rms = tl.rsqrt(sumsq / H + EPS)

    # Load weight vector, cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0).to(tl.float32)

    # Compute output: y = x32 * inv_rms * w
    y32 = x32 * inv_rms * w

    # Store FP32 result
    tl.store(base_out + offs, y32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous inputs
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device"
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape
        assert H == 4096, "Expected hidden_size == 4096"

        # Output buffer in FP32 for computation
        out = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        normalize_and_scale_row[grid](
            hidden_states, weight, out,
            hidden_states.stride(0), out.stride(0),
            B, H,
            num_warps=4,
            num_stages=2,
        )

        # Cast back to original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
