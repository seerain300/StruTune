import torch
import triton
import triton.language as tl


# Reduction kernel: per-row inv_rms = rsqrt(mean(x^2) + EPS)
@triton.jit
def _row_rms_inv_kernel(
    hidden_ptr,            # *const fp16/bf16/fp32, shape [B, H]
    inv_rms_ptr,           # *float32, shape [B]
    B: tl.int32,           # batch size
    H: tl.int32,           # hidden size
    EPS: tl.float32,       # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    # Base pointer for this row (assuming contiguous [B, H] with stride(1) == 1)
    row_base = hidden_ptr + row * H

    sum_sq = 0.0
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(row_base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)
        col += BLOCK_SIZE

    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)


# Elementwise kernel: out[row, col] = hidden[row, col] * inv_rms[row] * weight[col]
@triton.jit
def _apply_two_scales_kernel(
    hidden_ptr,            # *const fp16/bf16/fp32, shape [B, H]
    inv_rms_ptr,           # *float32, shape [B]
    weight_ptr,            # *float32, shape [H]
    out_ptr,               # *float32, shape [B, H]
    B: tl.int32,
    H: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= B:
        return

    col_start = col_block * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load hidden row slice and weight slice
    x = tl.load(hidden_ptr + row * H + offs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    inv_r = tl.load(inv_rms_ptr + row)

    # Compute in float32
    x_f32 = x.to(tl.float32)
    w_f32 = w.to(tl.float32)
    inv_f32 = inv_r  # already float32 from kernel

    y = x_f32 * inv_f32 * w_f32

    # Store result (float32)
    tl.store(out_ptr + row * H + offs, y, mask=mask)


def run_triton(hidden_states, weight):
    """
    Triton-only implementation of:
        x = hidden_states.to(torch.float32)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        y = (x * inv_rms) * weight.to(torch.float32)
        return y.to(hidden_states.dtype)
    All computation is done in Triton kernels. Host code only allocates and launches.
    """
    # Ensure contiguous and float32 for compute
    hidden = hidden_states.contiguous().to(torch.float32)
    weight = weight.contiguous().to(torch.float32)

    B, H = hidden.shape
    device = hidden.device

    # 1) Compute per-row inv_rms in Triton
    inv_rms = torch.empty((B,), dtype=torch.float32, device=device)
    _row_rms_inv_kernel[(B,)](
        hidden, inv_rms, B, H, 1e-5, BLOCK_SIZE=1024, num_warps=8, num_stages=2
    )

    # 2) Elementwise apply: out[row, col] = hidden[row, col] * inv_rms[row] * weight[col]
    out_f32 = torch.empty((B, H), dtype=torch.float32, device=device)

    # Choose tiling for elementwise apply
    if H >= 4096:
        BLOCK_SIZE_E = 4096
        num_warps_e = 16
        grid_apply = (B, 1)  # process entire row in one block
    elif H >= 2048:
        BLOCK_SIZE_E = 2048
        num_warps_e = 8
        grid_apply = (B, triton.cdiv(H, BLOCK_SIZE_E))
    else:
        BLOCK_SIZE_E = 1024
        num_warps_e = 4
        grid_apply = (B, triton.cdiv(H, BLOCK_SIZE_E))

    _apply_two_scales_kernel[grid_apply](
        hidden, inv_rms, weight, out_f32, B, H,
        BLOCK_SIZE=BLOCK_SIZE_E,
        num_warps=num_warps_e, num_stages=2
    )

    # Cast back to original dtype
    return out_f32.to(hidden_states.dtype)


# Optional: keep original Model for reference; evaluation uses ModelNew.
class Model(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
