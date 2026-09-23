import torch
import triton
import triton.language as tl

# Kernel 1: compute inv_rms per row (sum of squares reduction over columns)
@triton.jit
def _compute_inv_rms_kernel(
    x_ptr,              # *f32, pointer to input hidden_states (float32)
    inv_rms_ptr,        # *f32, pointer to output vector [B]
    B: tl.constexpr,    # batch size
    H: tl.constexpr,    # hidden size
    EPS: tl.constexpr,  # epsilon
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)  # one program per row
    # Accumulate sum of squares for this row in float32
    sum_sq = 0.0
    start = 0
    while start < H:
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        # Load row slice as float32
        x = tl.load(x_ptr + pid * H + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE
    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + pid, inv_rms)

# Kernel 2: elementwise compute y = (x * inv_rms) * weight
@triton.jit
def _apply_weight_kernel(
    x_ptr,           # *f32, pointer to input hidden_states (float32)
    inv_rms_ptr,     # *f32, pointer to per-row inv_rms [B]
    weight_ptr,      # *f32, pointer to weight [H]
    out_ptr,         # *T, pointer to output (casted to target dtype)
    B: tl.constexpr, # batch size
    H: tl.constexpr, # hidden size
    OUT_DTYPE: tl.constexpr,  # output dtype (e.g., tl.bfloat16)
    BLOCK_SIZE: tl.constexpr  # columns per block
):
    pid_row = tl.program_id(0)  # row id
    pid_col_block = tl.program_id(1)  # block id along columns
    cols = pid_col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    # Load x slice for this row
    x = tl.load(x_ptr + pid_row * H + cols, mask=mask, other=0.0)

    # Load per-row inv_rms
    inv_rms = tl.load(inv_rms_ptr + pid_row)

    # Load weight slice
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)

    # Compute y in float32 then cast to OUT_DTYPE
    y = x * inv_rms * w
    y_cast = y.to(OUT_DTYPE)

    # Store output
    tl.store(out_ptr + pid_row * H + cols, y_cast, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        # Shapes
        batch_size, hidden_size = hidden.shape
        # hidden_size is assumed to be 4096 in the provided harness
        # We'll handle general H, but BLOCK_SIZE below assumes H is multiple of 1024 or at least looped.

        # Compute in float32
        x_f32 = hidden.to(torch.float32)

        # Allocate inv_rms vector [B] (float32)
        inv_rms = torch.empty(batch_size, dtype=torch.float32, device=hidden.device)

        # Launch reduction kernel: one program per row
        BLOCK_SIZE = 1024
        grid_reduce = (batch_size,)
        _compute_inv_rms_kernel[grid_reduce](
            x_f32, inv_rms, batch_size, hidden_size, self.epsilon,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2
        )

        # Prepare output tensor in the same dtype as hidden (e.g., bfloat16)
        out = torch.empty((batch_size, hidden_size), dtype=hidden.dtype, device=hidden.device)

        # Determine Triton OUT_DTYPE based on hidden dtype
        if hidden.dtype == torch.bfloat16:
            OUT_DTYPE = tl.bfloat16
        elif hidden.dtype == torch.float16:
            OUT_DTYPE = tl.float16
        elif hidden.dtype == torch.float32:
            OUT_DTYPE = tl.float32
        else:
            raise RuntimeError(f"Unsupported dtype: {hidden.dtype}")

        # Launch elementwise kernel: 2D grid over rows and column blocks
        grid_apply = (batch_size, triton.cdiv(hidden_size, BLOCK_SIZE))
        _apply_weight_kernel[grid_apply](
            x_f32, inv_rms, weight.to(torch.float32), out,
            batch_size, hidden_size, OUT_DTYPE,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2
        )

        return out