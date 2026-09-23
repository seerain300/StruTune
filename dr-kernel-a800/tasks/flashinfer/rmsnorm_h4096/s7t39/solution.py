import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, hidden_size = 4096 (constexpr), no masks.
# Optimization: avoid re-reading hidden states in a second pass.
# During the first pass, store x (float32) into a temp buffer (out_fp32_ptr),
# then immediately scale using x and weight, and write the final result to out_ptr.
@triton.jit
def _row_fused_scale_kernel(hidden_ptr, weight_ptr, out_fp32_ptr, out_ptr,
                             hidden_size: tl.constexpr, EPS: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    # Each program handles one row
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)  # 0..BLOCK_SIZE-1; BLOCK_SIZE == hidden_size (4096)

    # Row base index
    idx = pid * hidden_size + offs

    # Load hidden states once, cast to float32, compute sum of squares
    x = tl.load(hidden_ptr + idx)               # no mask (hidden_size is exact)
    x = x.to(tl.float32)
    sumsq = tl.sum(x * x)

    mean = sumsq / hidden_size
    inv_rms = tl.rsqrt(mean + EPS)

    # Store x (float32) to temp output buffer
    tl.store(out_fp32_ptr + idx, x)

    # Scale using x (from temp) and weight, compute in fp32
    w = tl.load(weight_ptr + offs).to(tl.float32)
    y = out_fp32_ptr[idx] * inv_rms * w
    tl.store(out_ptr + idx, y)                  # Triton will cast to out_ptr's dtype as needed


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
        # Ensure contiguous for simple pointer arithmetic
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        batch_size, hidden_size = hidden.shape
        assert hidden_size == 4096, "This Triton implementation expects hidden_size == 4096"
        assert weight.numel() == hidden_size, "weight must have length equal to hidden_size"

        # Output tensors:
        # - out_fp32: float32 temp to hold x (float32), used for scaling after computing inv_rms
        # - out: final output with same dtype as hidden
        out_fp32 = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=hidden.device)
        out = torch.empty_like(hidden)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _row_fused_scale_kernel[grid](
            hidden, weight, out_fp32, out,
            hidden_size, 1e-5,
            BLOCK_SIZE=hidden_size,
            num_warps=16,
            num_stages=1
        )
        return out


def run(*args):
    return ModelNew()(*args)
