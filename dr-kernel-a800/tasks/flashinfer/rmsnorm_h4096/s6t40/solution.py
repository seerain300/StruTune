import torch
import triton
import triton.language as tl

# Kernel 1: compute per-row sum of squares in float32.
# Output: out_ptr[row] = sum(x[row, :]^2)
@triton.jit
def _sum_sq_kernel(
    hidden_ptr,   # *T_in, pointer to input hidden_states
    out_ptr,      # *float32, pointer to output per-row sum
    batch_size,   # int32
    hidden_size,  # int32
    BLOCK_SIZE: tl.constexpr   # e.g., 1024
):
    row = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    # Iterate over columns in tiles
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row * hidden_size + offs, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        acc += tl.sum(x_fp32 * x_fp32, axis=0)
    tl.store(out_ptr + row, acc)


# Kernel 2: compute inv_rms per row: inv_rms[row] = 1 / sqrt(sum_sq[row] / hidden_size + eps)
@triton.jit
def _inv_rms_kernel(
    sum_ptr,      # *float32, per-row sum of squares
    out_ptr,      # *float32, per-row inv_rms
    hidden_size,  # int32
    eps,          # float32
    BLOCK_SIZE: tl.constexpr   # 1 (scalar per row)
):
    row = tl.program_id(0)
    s = tl.load(sum_ptr + row)  # float32
    mean_sq = s / hidden_size
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
    tl.store(out_ptr + row, inv_rms)


# Kernel 3: scale using per-row inv_rms and weight, write output.
# Output: out[row, col] = hidden[row, col] * inv_rms[row] * weight[col]
@triton.jit
def _scale_kernel(
    hidden_ptr,   # *T_in, pointer to input hidden_states
    weight_ptr,   # *float32, pointer to weight (cast to float32)
    out_ptr,      # *T_out, pointer to output (casted)
    batch_size,   # int32
    hidden_size,  # int32
    inv_rms_ptr,  # *float32, pointer to per-row inv_rms
    out_dtype_code: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_SIZE: tl.constexpr        # e.g., 1024
):
    row = tl.program_id(0)
    # Load per-row inv_rms
    inv_rms = tl.load(inv_rms_ptr + row)  # float32 scalar
    # Loop over columns in tiles: multiply hidden by inv_rms and weight, store
    for col_start in range(0, hidden_size, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row * hidden_size + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x.to(tl.float32) * inv_rms * w
        # Cast to desired output dtype
        if out_dtype_code == 0:
            y_cast = y  # fp32
        elif out_dtype_code == 1:
            y_cast = y.to(tl.float16)
        else:
            y_cast = y.to(tl.bfloat16)
        tl.store(out_ptr + row * hidden_size + offs, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs
        assert hidden_states.dim() == 2, "hidden_states must be [batch_size, hidden_size]"
        assert weight.dim() == 1 and weight.shape[0] == hidden_states.shape[1], "weight must be [hidden_size]"
        batch_size, hidden_size = hidden_states.shape
        # Output tensor
        out = torch.empty_like(hidden_states)

        # Choose block size and warps
        BLOCK_SIZE = 1024
        num_warps = 8
        num_stages = 3

        # 1) Triton kernel to compute per-row sum of squares (float32)
        sum_out = torch.empty(batch_size, dtype=torch.float32, device=hidden_states.device)
        grid_sum = (batch_size,)
        _sum_sq_kernel[grid_sum](
            hidden_states, sum_out,
            batch_size, hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages
        )

        # 2) Triton kernel to compute inv_rms per row: inv_rms = 1 / sqrt(mean + eps)
        eps = 1e-5
        inv_rms = torch.empty_like(sum_out)  # float32 per row
        _inv_rms_kernel[(batch_size,)](
            sum_out, inv_rms, hidden_size, eps,
            BLOCK_SIZE=1,
            num_warps=1,
            num_stages=1
        )

        # 3) Triton kernel to scale and write output
        # Determine output dtype code: 0=fp32, 1=fp16, 2=bf16
        if out.dtype == torch.float32:
            out_dtype_code = 0
        elif out.dtype == torch.float16:
            out_dtype_code = 1
        elif out.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            raise RuntimeError(f"Unsupported output dtype: {out.dtype}")

        _scale_kernel[(batch_size,)](
            hidden_states, weight.to(torch.float32), out,
            batch_size, hidden_size,
            inv_rms, out_dtype_code,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages
        )

        return out


def run(*args):
    return ModelNew()(*args)
