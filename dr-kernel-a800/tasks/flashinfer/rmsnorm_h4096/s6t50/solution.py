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

    # Accumulate sum of squares in fp32
    sum_sq = 0.0
    offs = 0
    while offs < hidden_size:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < hidden_size
        x = tl.load(hidden_ptr + idx, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE

    mean_sq = sum_sq / hidden_size
    inv_rms = tl.rsqrt(mean_sq + eps)

    # Second pass: scale and store
    offs = 0
    while offs < hidden_size:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < hidden_size
        x = tl.load(hidden_ptr + idx, mask=mask, other=0.0)
        w = tl.load(weight_ptr + idx, mask=mask, other=0.0)
        y = x * inv_rms * w

        # Cast at store based on out_dtype_code
        if out_dtype_code == 0:
            out_val = y
        elif out_dtype_code == 1:
            out_val = y.to(tl.float16)
        else:
            out_val = y.to(tl.bfloat16)

        tl.store(out_ptr + idx, out_val, mask=mask)
        offs += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure 2D input with last dim == 4096
        assert hidden_states.dim() == 2, "hidden_states must be 2D [batch_size, hidden_size]"
        batch_size, hidden_size = hidden_states.shape
        assert hidden_size == 4096, "hidden_size must be 4096 for this implementation"
        assert weight.dim() == 1 and weight.shape[0] == hidden_size, "weight must be [hidden_size]"

        # Compute in float32; cast only at store
        hidden_in = hidden_states.to(torch.float32).contiguous()
        weight_in = weight.to(torch.float32).contiguous()

        # Determine output dtype code
        if hidden_states.dtype == torch.float32:
            out_dtype_code = 0
        elif hidden_states.dtype == torch.float16:
            out_dtype_code = 1
        elif hidden_states.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            raise ValueError("Unsupported output dtype; expected fp32/fp16/bf16")

        out = torch.empty_like(hidden_states, device=hidden_states.device)

        # Launch Triton kernel: one program per row
        grid = (batch_size,)
        _rms_scale_kernel[grid](
            hidden_in, weight_in, out,
            batch_size, hidden_size, 1e-5, out_dtype_code,
            BLOCK_SIZE=1024,
            num_warps=8, num_stages=3
        )
        return out


def run(*args):
    return ModelNew()(*args)
