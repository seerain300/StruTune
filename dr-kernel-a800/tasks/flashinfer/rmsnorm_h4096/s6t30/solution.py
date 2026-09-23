import torch
import triton
import triton.language as tl

# Triton kernel: per-row RMS scaling with weight.
# One program per row. We read x twice: once to compute sum of squares, once to write output.
@triton.jit
def _rms_scale_kernel(
    hidden_ptr,        # *float32, pointer to hidden_states (cast to float32)
    weight_ptr,        # *float32, pointer to weight (cast to float32)
    out_ptr,           # *T_out, pointer to output (will store casted result)
    batch_size,        # int32
    hidden_size,       # int32
    eps,               # float32
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row_id = tl.program_id(0)  # one program per row

    # First pass: accumulate sum of squares over the row.
    sum_sq = 0.0
    num_tiles = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile in range(0, num_tiles):
        offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)

    # Compute inv_rms = 1 / sqrt(mean(x^2) + eps) where mean = sum_sq / hidden_size
    mean = sum_sq / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Second pass: scale and write output
    for tile in range(0, num_tiles):
        offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row_id * hidden_size + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        y = x * inv_rms * w  # float32 compute
        # Cast to desired output dtype
        if out_dtype_code == 0:
            out_val = y  # fp32
        elif out_dtype_code == 1:
            out_val = y.to(tl.float16)  # fp16
        else:
            out_val = y.to(tl.bfloat16)  # bf16
        tl.store(out_ptr + row_id * hidden_size + offs, out_val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure 2D input [batch, hidden_size] and 1D weight [hidden_size]
        assert hidden_states.dim() == 2, "hidden_states must be 2D [batch, hidden_size]"
        assert weight.dim() == 1, "weight must be 1D [hidden_size]"
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "This benchmark expects hidden_size == 4096"
        assert hidden_states.device == weight.device, "hidden_states and weight must be on the same device"
        assert hidden_states.device.type == "cuda", "This Triton implementation requires CUDA device"

        # Cast to float32 for compute; Triton will write back to desired dtype
        hidden = hidden_states.to(torch.float32).contiguous()
        weight = weight.to(torch.float32).contiguous()

        # Output tensor in the same dtype as hidden_states (PyTorch op did .to(hidden_states.dtype))
        out = torch.empty((batch_size, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)

        # Choose dtype code
        if out.dtype == torch.float32:
            out_dtype_code = 0
        elif out.dtype == torch.float16:
            out_dtype_code = 1
        elif out.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            raise RuntimeError(f"Unsupported output dtype: {out.dtype}")

        # Grid: one program per row
        grid = (batch_size,)

        # Launch Triton kernel
        _rms_scale_kernel[grid](
            hidden, weight, out,
            batch_size, hidden_size, 1e-5, out_dtype_code,
            BLOCK_SIZE=1024, num_warps=8, num_stages=3
        )
        return out


def run(*args):
    return ModelNew()(*args)
