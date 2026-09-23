import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute per-row sum of squares in float32
# hidden_states_ptr: float32 [B, H], row-major contiguous
# sums_ptr: float32 [B]
@triton.jit
def reduce_sum_sq_kernel(hidden_states_ptr, sums_ptr, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    # Each program handles one row
    row_ptr = hidden_states_ptr + row_id * H
    sumsq = 0.0
    for col_start in range(0, H, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(row_ptr + offs, mask=mask, other=0.0)
        sq = x * x
        sumsq += tl.sum(sq, axis=0)
    tl.store(sums_ptr + row_id, sumsq)


# Triton kernel: compute per-row inv_rms from sums
# sums_ptr: float32 [B], inv_rms_ptr: float32 [B], H is passed for shape info
@triton.jit
def compute_inv_rms_kernel(sums_ptr, inv_rms_ptr, H, EPS, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    sumsq = tl.load(sums_ptr + row_id)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv)


# Triton kernel: produce output y[row, col] = hidden[row, col] * inv_rms[row] * weight[col]
# 2D tiling over rows and columns
@triton.jit
def output_kernel(hidden_ptr, weight_ptr, inv_rms_ptr, out_ptr,
                   B, H, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_mask = row_ids < B

    # Preload per-row inv_rms (vector of size BLOCK_M)
    inv = tl.load(inv_rms_ptr + row_ids, mask=rows_mask, other=1.0)  # [BLOCK_M]

    for col_start in range(0, H, BLOCK_N):
        col_ids = col_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        cols_mask = col_ids < H

        # Build 2D mask for the tile
        mask = rows_mask[:, None] & cols_mask[None, :]

        # Pointers for the hidden tile: shape [BLOCK_M, BLOCK_N]
        ptrs = hidden_ptr + row_ids[:, None] * H + col_ids[None, :]

        # Load hidden tile and weight vector, compute result
        x = tl.load(ptrs, mask=mask, other=0.0)  # [BLOCK_M, BLOCK_N]
        w = tl.load(weight_ptr + col_ids, mask=cols_mask, other=0.0)  # [BLOCK_N]

        # Broadcast per-row inv to [BLOCK_M, 1] and multiply
        y = x * inv[:, None] * w[None, :]

        # Store to output
        out_ptrs = out_ptr + row_ids[:, None] * H + col_ids[None, :]
        tl.store(out_ptrs, y, mask=mask)


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
        BLOCK_SIZE = 256
        grid_reduce = (B,)
        reduce_sum_sq_kernel[grid_reduce](hidden, sums, H, BLOCK_SIZE, num_warps=4, num_stages=2)

        # 2) Compute inv_rms per row
        inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)
        compute_inv_rms_kernel[grid_reduce](sums, inv_rms, H, 1e-5, BLOCK_SIZE, num_warps=4, num_stages=2)

        # 3) Produce output in float32 using 2D-tiled elementwise kernel
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
        BLOCK_M = 32   # rows per program
        BLOCK_N = 1024 # columns per program (covers H=4096)
        grid_out = (triton.cdiv(B, BLOCK_M),)
        output_kernel[grid_out](hidden, w, inv_rms, out_f32, B, H, BLOCK_M, BLOCK_N, num_warps=8, num_stages=2)

        # Cast to original dtype of hidden_states and return
        out = out_f32.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
