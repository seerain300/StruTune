import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: compute per-row sum of squares (float32)
# x_ptr: [B, H] float32, row-major contiguous
# sums_ptr: [B] float32, output per-row sum of squares
@triton.jit
def reduce_sum_sq_kernel(x_ptr, sums_ptr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    acc = 0.0
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        idx = row_id * H + cols
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)
    tl.store(sums_ptr + row_id, acc)


# Kernel 2: compute per-row inv_rms from sums
# sums_ptr: [B] float32
# inv_rms_ptr: [B] float32
@triton.jit
def compute_inv_rms_kernel(sums_ptr, inv_rms_ptr, H: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    sumsq = tl.load(sums_ptr + row_id)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv)


# Kernel 3: precompute per-row scaled weight vector: w_scaled[j] = w[j] * inv_rms[row]
# w_ptr: [H] float32
# inv_rms_ptr: [B] float32
# w_scaled_ptr: [B, H] float32
@triton.jit
def precompute_row_scaled_weight(w_ptr, inv_rms_ptr, w_scaled_ptr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    inv = tl.load(inv_rms_ptr + row_id)
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        w_scaled = w * inv
        # Store to [row_id, cols]
        tl.store(w_scaled_ptr + row_id * H + cols, w_scaled, mask=mask)


# Kernel 4: produce output y = x * w_scaled, 2D tiling over rows and columns
# x_ptr: [B, H] float32
# w_scaled_ptr: [B, H] float32
# out_ptr: [B, H] float32
@triton.jit
def output_kernel(x_ptr, w_scaled_ptr, out_ptr, B: tl.constexpr, H: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    mask_rows = rows < B

    for col_start in range(0, H, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask_cols = cols < H

        idx = rows[:, None] * H + cols[None, :]  # [BLOCK_M, BLOCK_N]
        mask = (mask_rows[:, None]) & (mask_cols[None, :])

        x = tl.load(x_ptr + idx, mask=mask, other=0.0)       # [BLOCK_M, BLOCK_N]
        w_scaled = tl.load(w_scaled_ptr + idx, mask=mask, other=0.0)  # [BLOCK_M, BLOCK_N]
        y = x * w_scaled
        tl.store(out_ptr + idx, y, mask=mask)


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

        # 3) Precompute per-row scaled weight vector: w_scaled[row, :]
        w_scaled = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
        # One program per row; tile columns for efficiency
        grid_rows = (B,)
        precompute_row_scaled_weight[grid_rows](w, inv_rms, w_scaled, H, BLOCK_SIZE, num_warps=4, num_stages=2)

        # 4) Produce output in float32 using 2D-tiled kernel over rows and columns
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
        BLOCK_M = 16  # more rows per program
        BLOCK_N = 512 # larger column tile for H=4096
        grid_out = (triton.cdiv(B, BLOCK_M),)
        output_kernel[grid_out](hidden, w_scaled, out_f32, B, H, BLOCK_M, BLOCK_N, num_warps=8, num_stages=2)

        # Cast to original dtype of hidden_states and return
        out = out_f32.to(hidden_states.dtype)
        # Return on original device
        if hidden_states.device.type != "cuda":
            out = out.to(hidden_states.device)
        return out


def run(*args):
    return ModelNew()(*args)
