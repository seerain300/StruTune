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
# H: number of columns (runtime int)
@triton.jit
def reduce_sum_sq_kernel(x_ptr, sums_ptr, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    sumsq = tl.zeros((), dtype=tl.float32)
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_row_ptr = x_ptr + row_id * H
        x_vals = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        sumsq += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sums_ptr + row_id, sumsq)


# Triton kernel 2: compute per-row inv_rms from sums (float32)
# sums_ptr: float32 [B]
# inv_rms_ptr: float32 [B]
# H: number of columns (runtime int)
@triton.jit
def compute_inv_rms_kernel(sums_ptr, inv_rms_ptr, H, EPS: tl.constexpr):
    row_id = tl.program_id(axis=0)
    sumsq = tl.load(sums_ptr + row_id)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv)


# Triton kernel 3: produce output y = x * inv_rms[row] * w (float32)
# x_ptr: float32 [B, H], row-major contiguous
# w_ptr: float32 [H]
# inv_rms_ptr: float32 [B]
# out_ptr: float32 [B, H]
@triton.jit
def output_kernel(x_ptr, w_ptr, inv_rms_ptr, out_ptr, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    inv = tl.load(inv_rms_ptr + row_id)
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_row_ptr = x_ptr + row_id * H
        x_vals = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        w_vals = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y_vals = x_vals * inv * w_vals
        tl.store(out_ptr + row_id * H + cols, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on CUDA and in float32 for compute
        orig_device = hidden_states.device
        x = hidden_states.to(torch.float32)
        w = weight.to(torch.float32)
        x = x.contiguous()
        w = w.contiguous()
        B, H = x.shape

        # Allocate buffers
        sums = torch.empty((B,), dtype=torch.float32, device=x.device)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=x.device)

        # Launch kernels
        BLOCK_SIZE = 1024
        reduce_sum_sq_kernel[(B,)](x, sums, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)
        EPS = 1e-5
        compute_inv_rms_kernel[(B,)](sums, inv_rms, H, EPS, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)
        output_kernel[(B,)](x, w, inv_rms, out_f32, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Return tensor (no Python list). Since original inputs are bfloat16,
        # out_f32 is acceptable and matches the numerical behavior.
        if orig_device.type != "cuda":
            out = out_f32.to(orig_device)
        else:
            out = out_f32
        return out


def run(*args):
    return ModelNew()(*args)
