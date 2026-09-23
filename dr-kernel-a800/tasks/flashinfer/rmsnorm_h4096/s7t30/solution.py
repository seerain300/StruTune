import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden_size (4096).
# Assumes hidden_size == 4096. Two vector passes: accumulate sum of squares, then scale/store.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_ptr,
                             hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_offset = pid * hidden_size

    # Vector of column offsets (0..BLOCK_SIZE-1), with BLOCK_SIZE == hidden_size
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size  # mask is always true when BLOCK_SIZE == hidden_size

    # Pass 1: accumulate sum of squares across the row in float32
    x = tl.load(hidden_ptr + row_offset + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    sum_sq = tl.sum(x32 * x32, axis=0)

    # Compute inv_rms as a scalar per row
    inv_rms = tl.rsqrt(sum_sq / hidden_size + EPS)

    # Pass 2: scale and store
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x32 * inv_rms * w
    tl.store(out_ptr + row_offset + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton kernel expects CUDA tensors. Original model uses bfloat16; we convert to float32 for compute.
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        assert hidden_states.shape[1] == 4096, "hidden_size must be 4096."

        # Compute in float32 inside the kernel; output cast back to original dtype to match behavior.
        hidden = hidden_states.contiguous().to(torch.float32)
        weight = weight.contiguous().to(torch.float32)

        B, H = hidden.shape
        out = torch.empty_like(hidden)

        # Launch configuration: one program per row
        grid = (B,)

        # Tuned launch params for robust performance on memory-bound kernel
        _row_fused_scale_kernel[grid](
            hidden, weight, out,
            hidden_size=H, EPS=1e-5,
            BLOCK_SIZE=H,
            num_warps=8,
            num_stages=1,
        )

        # Cast back to original dtype to match the original model's behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
