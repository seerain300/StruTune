import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute per-row sum of squares in float32
# hidden_ptr: float32 [B, H], row-major contiguous
# sums_ptr: float32 [B]
@triton.jit
def reduce_sum_sq_kernel(hidden_ptr, sums_ptr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = row_id * H
    acc = 0.0  # scalar accumulator for this row
    # Iterate over columns in tiles
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row_start + cols, mask=mask, other=0.0)
        x2 = x * x
        acc += tl.sum(x2, axis=0)
    tl.store(sums_ptr + row_id, acc)


# Triton kernel: compute per-row inverse RMS from sums and H
# sums_ptr: float32 [B]
# inv_ptr: float32 [B]
@triton.jit
def compute_inv_rms_kernel(sums_ptr, inv_ptr, H: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    sumsq = tl.load(sums_ptr + row_id)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_ptr + row_id, inv)


# Triton kernel: elementwise output in float32, 2D tiled over rows and columns
# hidden_ptr: float32 [B, H]
# w_ptr: float32 [H]
# inv_ptr: float32 [B]
# out_ptr: float32 [B, H]
@triton.jit
def output_kernel(hidden_ptr, w_ptr, inv_ptr, out_ptr, B, H, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_rows = row_offsets < B
    mask_cols = col_offsets < H

    # Load per-row inv_rms
    inv_row = tl.load(inv_ptr + row_offsets, mask=mask_rows, other=1.0)  # shape [BLOCK_M]

    # Broadcast masks
    mask_rc = mask_rows[:, None] & mask_cols[None, :]

    # Base pointers for each row
    base_rows = row_offsets * H  # shape [BLOCK_M]

    # Load x tile: shape [BLOCK_M, BLOCK_N]
    x_ptrs = hidden_ptr + base_rows[:, None] + col_offsets[None, :]
    x = tl.load(x_ptrs, mask=mask_rc, other=0.0)  # float32

    # Load w tile: shape [BLOCK_N]
    w = tl.load(w_ptr + col_offsets, mask=mask_cols, other=0.0)  # float32

    # Compute y = x * inv_row[:, None] * w[None, :]
    y = x * inv_row[:, None] * w[None, :]

    # Store
    out_ptrs = out_ptr + base_rows[:, None] + col_offsets[None, :]
    tl.store(out_ptrs, y, mask=mask_rc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Fallback to original PyTorch if Triton not available or tensors not on CUDA
        if (not TRITON_AVAILABLE) or (not hidden_states.is_cuda) or (not weight.is_cuda):
            batch_size, hidden_size = hidden_states.shape
            assert hidden_size == 4096
            EPS = 1e-5
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure dtype float32 and contiguous (host-side minimal setup)
        hidden = hidden_states.to(torch.float32).contiguous()  # [B, H]
        w = weight.to(torch.float32).contiguous()             # [H]
        B, H = hidden.shape

        # 1) Compute per-row sum of squares (kernel)
        sums = torch.empty((B,), dtype=torch.float32, device=hidden.device)
        BLOCK_SIZE = 256
        grid_reduce = (B,)
        reduce_sum_sq_kernel[grid_reduce](hidden, sums, H, BLOCK_SIZE, num_warps=4, num_stages=2)

        # 2) Compute inv_rms per row (kernel)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)
        EPS = 1e-5
        grid_inv = (B,)
        compute_inv_rms_kernel[grid_inv](sums, inv_rms, H, EPS, BLOCK_SIZE, num_warps=4, num_stages=2)

        # 3) Output: elementwise y = hidden * inv_rms[:, None] * w[None, :]
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
        # Use 2D tiling over rows and columns to improve throughput, better occupancy for large B
        BLOCK_M = 32    # rows per program
        BLOCK_N = 1024  # columns per program (H=4096 => 4 tiles)
        grid_m = triton.cdiv(B, BLOCK_M)
        grid_n = triton.cdiv(H, BLOCK_N)
        output_kernel[(grid_m, grid_n)](hidden, w, inv_rms, out_f32, B, H, BLOCK_M, BLOCK_N, num_warps=8, num_stages=3)

        # Cast to original dtype and return on original device
        out = out_f32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
