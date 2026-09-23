import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: compute per-row sum of squares (float32)
@triton.jit
def reduce_sum_sq_kernel(hidden_ptr, sums_ptr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    # Accumulator for this row
    sumsq = tl.zeros((), dtype=tl.float32)
    # Iterate over columns in tiles
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        # Load row tile
        x = tl.load(hidden_ptr + row_id * H + cols, mask=mask, other=0.0)
        x2 = x * x
        # Reduce within the tile
        tile_sum = tl.sum(x2, axis=0)
        sumsq += tile_sum
    # Store per-row sum of squares
    tl.store(sums_ptr + row_id, sumsq)


# Kernel 2: compute inv_rms per row in float32
@triton.jit
def compute_inv_rms_kernel(sums_ptr, inv_ptr, H: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    sumsq = tl.load(sums_ptr + row_id)  # scalar float32
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_ptr + row_id, inv)


# Kernel 3: 2D-tiled elementwise output: y[row, col] = hidden[row, col] * inv[row] * weight[col]
@triton.jit
def output_kernel(hidden_ptr, w_ptr, inv_ptr, out_ptr,
                   B: tl.constexpr, H: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Each program handles BLOCK_M rows and BLOCK_N columns
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    row_mask = rows < B
    # Load per-row inv_rms and broadcast across columns
    inv_row = tl.load(inv_ptr + rows, mask=row_mask, other=1.0)  # [BLOCK_M]
    inv_row = inv_row[:, None]  # [BLOCK_M, 1] for broadcasting

    for col_start in range(0, H, BLOCK_N):
        col_idxs = col_start + cols  # [BLOCK_N]
        col_mask = col_idxs < H
        # 2D mask for [rows, cols]
        mask = row_mask[:, None] & col_mask[None, :]

        # Load hidden tile [BLOCK_M, BLOCK_N]
        h_ptr = hidden_ptr + rows[:, None] * H + col_idxs[None, :]
        x = tl.load(h_ptr, mask=mask, other=0.0)

        # Load weight tile [BLOCK_N]
        w = tl.load(w_ptr + col_idxs, mask=col_mask, other=0.0)  # [BLOCK_N]
        w = w[None, :]  # broadcast over rows

        # Compute y = x * inv_row * w
        y = x * inv_row * w

        # Store to output
        out_ptr_tile = out_ptr + rows[:, None] * H + col_idxs[None, :]
        tl.store(out_ptr_tile, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # If Triton not available or tensors not on CUDA, fallback to original PyTorch implementation
        if (not TRITON_AVAILABLE) or (not hidden_states.is_cuda) or (not weight.is_cuda):
            batch_size, hidden_size = hidden_states.shape
            assert hidden_size == 4096
            EPS = 1e-5
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure dtype float32 and contiguous
        hidden = hidden_states.to(torch.float32).contiguous()  # [B, H]
        w = weight.to(torch.float32).contiguous()             # [H]
        B, H = hidden.shape

        # Allocate buffers
        sums = torch.empty((B,), dtype=torch.float32, device=hidden.device)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Kernel 1: per-row sum of squares
        BLOCK_SIZE = 256
        grid_reduce = (B,)
        reduce_sum_sq_kernel[grid_reduce](hidden, sums, H, BLOCK_SIZE, num_warps=4, num_stages=2)

        # Kernel 2: inv_rms per row
        EPS = 1e-5
        grid_inv = (B,)
        compute_inv_rms_kernel[grid_inv](sums, inv_rms, H, EPS, BLOCK_SIZE, num_warps=4, num_stages=2)

        # Kernel 3: 2D-tiled output
        BLOCK_M = 32   # rows per program
        BLOCK_N = 1024 # columns per program (H=4096 -> 4 iterations)
        grid_out = (triton.cdiv(B, BLOCK_M),)
        output_kernel[grid_out](hidden, w, inv_rms, out_f32, B, H, BLOCK_M, BLOCK_N, num_warps=4, num_stages=2)

        # Cast to original dtype and return on original device
        out = out_f32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
