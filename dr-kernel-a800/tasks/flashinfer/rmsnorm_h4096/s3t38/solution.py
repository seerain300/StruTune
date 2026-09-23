import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel 1: compute per-row sum of squares (float32)
# x_ptr: float32 [B, H], row-major contiguous
# sums_ptr: float32 [B]
# H: number of columns
@triton.jit
def reduce_sum_sq_kernel(x_ptr, sums_ptr, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    row_base = row_id * H
    sumsq = 0.0
    # Iterate over columns in tiles
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        ptrs = row_base + cols
        x_tile = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
        sumsq += tl.sum(x_tile * x_tile, axis=0)
    tl.store(sums_ptr + row_id, sumsq)


# Triton kernel 2: compute per-row inv_rms from sums
# sums_ptr: float32 [B], inv_rms_ptr: float32 [B], H (int), EPS (float)
@triton.jit
def compute_inv_rms_kernel(sums_ptr, inv_rms_ptr, H, EPS: tl.constexpr):
    row_id = tl.program_id(axis=0)
    sumsq = tl.load(sums_ptr + row_id)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv)


# Triton kernel 3: produce final output y = x * inv_rms[row] * weight
# x_ptr: float32 [B, H], w_ptr: float32 [H], inv_rms_ptr: float32 [B], out_ptr: float32 [B, H]
@triton.jit
def output_kernel(x_ptr, w_ptr, inv_rms_ptr, out_ptr, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    inv = tl.load(inv_rms_ptr + row_id)
    row_base = row_id * H
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_ptrs = row_base + cols
        w_ptrs = cols
        x_tile = tl.load(x_ptr + x_ptrs, mask=mask, other=0.0)
        w_tile = tl.load(w_ptr + w_ptrs, mask=mask, other=0.0)
        y_tile = x_tile * inv * w_tile
        tl.store(out_ptr + x_ptrs, y_tile, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Fallback to PyTorch if Triton or CUDA not available
        if (not TRITON_AVAILABLE) or (hidden_states.device.type != "cuda"):
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure contiguous and float32
        x = hidden_states.to(torch.float32).contiguous()      # [B, H]
        w = weight.to(torch.float32).contiguous()             # [H]
        B, H = x.shape
        device = x.device

        # Allocate intermediates
        sums = torch.empty((B,), dtype=torch.float32, device=device)      # per-row sum of squares
        inv_rms = torch.empty((B,), dtype=torch.float32, device=device)    # per-row normalization

        # Launch reduction kernel: one program per row
        BLOCK_SIZE = 1024
        reduce_sum_sq_kernel[(B,)](x, sums, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Launch normalization kernel
        compute_inv_rms_kernel[(B,)](sums, inv_rms, H, EPS=1e-5, num_warps=1, num_stages=1)

        # Allocate output tensor (float32)
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch output kernel: one program per row
        output_kernel[(B,)](x, w, inv_rms, out_f32, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2)

        # Cast to original dtype and return
        out = out_f32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
