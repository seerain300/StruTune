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
def reduce_sum_sq_kernel(hidden_ptr, sums_ptr, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    # bounds check
    if row_id >= B:
        return
    row_base = row_id * H
    sumsq = 0.0
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # load x row slice
        x = tl.load(hidden_ptr + row_base + offs, mask=mask, other=0.0)
        # accumulate sum of squares
        sumsq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE
    tl.store(sums_ptr + row_id, sumsq)


# Triton kernel: compute per-row inv_rms from sums
# sums_ptr: float32 [B], H: int, EPS: float
@triton.jit
def compute_inv_rms_kernel(sums_ptr, inv_rms_ptr, H, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    if row_id >= B:
        return
    sumsq = tl.load(sums_ptr + row_id)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv)


# Triton kernel: produce final output y = x * inv_rms[row] * weight[col]
# x_ptr: float32 [B, H], w_ptr: float32 [H], inv_rms_ptr: float32 [B], out_ptr: float32 [B, H]
@triton.jit
def output_kernel(x_ptr, w_ptr, inv_rms_ptr, out_ptr, B, H,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)  # programs over rows tiles
    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    mask_rows = rows < B

    # prepare column offsets
    col_start = 0
    while col_start < H:
        cols = col_start + tl.arange(0, BLOCK_N)
        mask_cols = cols < H

        # broadcast masks for 2D tile
        mask = mask_rows[:, None] & mask_cols[None, :]

        # load per-row inv_rms (masked for rows)
        inv = tl.load(inv_rms_ptr + rows, mask=mask_rows, other=1.0)  # [BLOCK_M]
        inv = inv[:, None]  # broadcast across columns

        # load x tile: [BLOCK_M, BLOCK_N]
        x_offs = rows[:, None] * H + cols[None, :]
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

        # load w tile: [BLOCK_N]
        w = tl.load(w_ptr + cols, mask=mask_cols, other=0.0)

        # compute y = x * inv * w (broadcast w across rows)
        y = x * inv * w[None, :]

        # store y
        out_offs = rows[:, None] * H + cols[None, :]
        tl.store(out_ptr + out_offs, y, mask=mask)

        col_start += BLOCK_N


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

        # Ensure dtype float32 and contiguous
        hidden = hidden_states.to(torch.float32).contiguous()  # [B, H]
        w = weight.to(torch.float32).contiguous()             # [H]
        B, H = hidden.shape

        # 1) Compute per-row sum of squares
        sums = torch.empty((B,), dtype=torch.float32, device=hidden.device)
        reduce_sum_sq_kernel[(B,)](hidden, sums, H, BLOCK_SIZE=256, num_warps=4, num_stages=2)

        # 2) Compute inv_rms per row
        inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)
        compute_inv_rms_kernel[(B,)](sums, inv_rms, H, EPS=1e-5, BLOCK_SIZE=256, num_warps=4, num_stages=2)

        # 3) Produce output in float32 using 2D-tiled elementwise kernel
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
        BLOCK_M = 16    # rows per program (tile)
        BLOCK_N = 1024  # columns per program (tile)
        grid_out = (triton.cdiv(B, BLOCK_M),)
        output_kernel[grid_out](hidden, w, inv_rms, out_f32, B, H, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8, num_stages=2)

        # Cast to original dtype of hidden_states and return on original device
        out = out_f32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
