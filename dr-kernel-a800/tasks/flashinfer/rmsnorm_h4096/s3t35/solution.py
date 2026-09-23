import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel A: compute per-row sum of squares in float32
# x_ptr: float32 [B, H], row-major contiguous
# sums_ptr: float32 [B]
@triton.jit
def reduce_sum_sq_kernel(x_ptr, sums_ptr, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    if row_id >= sums_ptr.shape[0]:
        return

    sumsq = 0.0
    # Iterate over columns in tiles
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
        x2 = x * x
        # Reduce this tile into a scalar
        sumsq += tl.sum(x2, axis=0)
    # Store per-row sum of squares
    tl.store(sums_ptr + row_id, sumsq)


# Triton kernel B: compute inv_rms from sums per row
# sums_ptr: float32 [B]
# inv_rms_ptr: float32 [B]
@triton.jit
def compute_inv_rms_kernel(sums_ptr, inv_rms_ptr, H, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    if row_id >= inv_rms_ptr.shape[0]:
        return

    sumsq = tl.load(sums_ptr + row_id)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv)


# Triton kernel C: produce final output in float32: y[row, col] = x[row, col] * inv_rms[row] * weight[col]
# x_ptr: float32 [B, H]
# w_ptr: float32 [H]
# out_ptr: float32 [B, H]
@triton.jit
def output_kernel(x_ptr, w_ptr, inv_rms_ptr, out_ptr, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    if row_id >= out_ptr.shape[0]:
        return

    inv = tl.load(inv_rms_ptr + row_id)
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = x * inv * w
        tl.store(out_ptr + row_id * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on CUDA and float32 for compute; make contiguous
        x = hidden_states.to(torch.float32).contiguous()
        w = weight.to(torch.float32).contiguous()
        B, H = x.shape

        # Prepare output buffer in float32
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=x.device)

        # Allocate intermediate buffers
        sums = torch.empty((B,), dtype=torch.float32, device=x.device)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

        # Launch Triton kernels
        BLOCK_SIZE = 1024  # tuned for H=4096
        grid = (B,)

        # Kernel A: per-row sum of squares
        reduce_sum_sq_kernel[grid](x, sums, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Kernel B: compute inv_rms
        compute_inv_rms_kernel[grid](sums, inv_rms, H, EPS=1e-5, BLOCK_SIZE=1, num_warps=1, num_stages=1)

        # Kernel C: final output
        output_kernel[grid](x, w, inv_rms, out_f32, H, BLOCK_SIZE=1024, num_warps=8, num_stages=4)

        # Match original behavior: cast back to original dtype of hidden_states
        out = out_f32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
