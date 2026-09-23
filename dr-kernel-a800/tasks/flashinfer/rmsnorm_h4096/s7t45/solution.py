import torch
import triton
import triton.language as tl

# Triton kernel: one program per row, vectorized across the full hidden size (4096).
# Two-pass approach:
# - Pass 1: accumulate sum of squares across the row in float32, compute inv_rms.
# - Pass 2: scale and store output, multiplying by weight.
@triton.jit
def _row_fused_scale_kernel(
    hidden_ptr,      # *float32 or *half/bf16, we'll cast to float32 inside
    weight_ptr,      # *float32 or *half/bf16, we'll cast to float32 inside
    out_ptr,         # *same dtype as input row (we'll cast here)
    batch_size,      # int32
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # e.g., 4096
):
    pid = tl.program_id(axis=0)  # program id: row index
    # Safety in case grid is larger than batch_size (shouldn't happen if we set grid=batch_size)
    if pid >= batch_size:
        return

    # Compute base offsets
    row_hidden_offset = pid * BLOCK_SIZE

    # Pass 1: compute sum of squares across the row in float32
    sumsq = 0.0
    # Simple linear loop over the row. BLOCK_SIZE is constexpr and equals 4096 in our setup.
    for col in range(0, BLOCK_SIZE):
        # Load x element, cast to float32 for accumulation
        x_val = tl.load(hidden_ptr + row_hidden_offset + col)
        x_f32 = x_val.to(tl.float32)
        sumsq += x_f32 * x_f32

    # Compute inv_rms: rsqrt(mean + EPS)
    mean = sumsq / BLOCK_SIZE
    inv_rms = tl.math.rsqrt(mean + EPS)

    # Pass 2: scale and store
    for col in range(0, BLOCK_SIZE):
        x_val = tl.load(hidden_ptr + row_hidden_offset + col)
        x_f32 = x_val.to(tl.float32)
        w_val = tl.load(weight_ptr + col).to(tl.float32)
        y_f32 = x_f32 * inv_rms * w_val
        # Store to output as float32 (we will cast in host if needed)
        tl.store(out_ptr + pid * BLOCK_SIZE + col, y_f32)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure device and contiguity
        assert hidden_states.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors."
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        batch_size = hidden.shape[0]
        hidden_size = hidden.shape[1]
        assert hidden_size == 4096, "This optimized kernel expects hidden_size == 4096."

        # Output in float32 for numeric stability, then cast back to input dtype at the end.
        out_fp32 = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        # Use a configuration that has shown strong speedups on memory-bound kernels.
        _row_fused_scale_kernel[grid](
            hidden, weight, out_fp32,
            batch_size,
            EPS=1e-5,
            BLOCK_SIZE=4096,
            num_warps=16,
            num_stages=1,
        )

        # Cast back to the original dtype to match the original behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
