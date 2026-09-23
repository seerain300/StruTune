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
    sumsq = 0.0
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_vals = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
        x_sq = x_vals * x_vals
        sumsq += tl.sum(x_sq, axis=0)
    tl.store(sums_ptr + row_id, sumsq)


# Triton kernel 2: compute per-row inv_rms (float32)
# sums_ptr: float32 [B]
# inv_rms_ptr: float32 [B]
# H: number of columns
@triton.jit
def compute_inv_rms_kernel(sums_ptr, inv_rms_ptr, H, EPS: tl.constexpr):
    row_id = tl.program_id(axis=0)
    sumsq = tl.load(sums_ptr + row_id)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv)


# Triton kernel 3: produce final output y = x * inv_rms[row] * w (float32)
# x_ptr: float32 [B, H], row-major contiguous
# w_ptr: float32 [H]
# inv_rms_ptr: float32 [B]
# out_ptr: float32 [B, H]
# H: number of columns
@triton.jit
def output_kernel(x_ptr, w_ptr, inv_rms_ptr, out_ptr, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    inv = tl.load(inv_rms_ptr + row_id)
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x_vals = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
        w_vals = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = x_vals * inv * w_vals
        tl.store(out_ptr + row_id * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure inputs are on CUDA; compute in float32
        orig_device = hidden_states.device
        x = hidden_states.to(torch.float32).contiguous()
        w = weight.to(torch.float32).contiguous()

        B, H = x.shape
        assert w.numel() == H, "weight must have length equal to hidden_size"

        # Allocate intermediates
        sums = torch.empty((B,), dtype=torch.float32, device=x.device)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=x.device)

        # Tuning parameters
        BLOCK_SIZE = 1024  # 4 iterations for H=4096
        grid = (B,)

        # Kernel 1: row-wise sum of squares
        reduce_sum_sq_kernel[grid](x, sums, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Kernel 2: compute inv_rms
        compute_inv_rms_kernel[grid](sums, inv_rms, H, EPS=1e-5, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2)

        # Kernel 3: produce final output
        output_kernel[grid](x, w, inv_rms, out_f32, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2)

        # Cast back to original dtype to match original behavior and return on original device
        out = out_f32.to(hidden_states.dtype)
        if orig_device.type != "cuda":
            out = out.to(orig_device)
        return out


def run(*args):
    return ModelNew()(*args)
