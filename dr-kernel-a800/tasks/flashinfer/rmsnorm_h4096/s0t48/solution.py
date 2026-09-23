import torch
import triton
import triton.language as tl

# Kernel 1: compute per-row sum of squares and inv_rms
@triton.jit
def reduce_row_sumsq_kernel(
    x_ptr,            # *f32, [B, H], contiguous
    inv_rms_ptr,      # *f32, [B]
    B: tl.constexpr,  # batch size (used in grid)
    H: tl.constexpr,  # hidden size (compile-time specialization helps)
    EPS: tl.constexpr,          # epsilon
    BLOCK_SIZE: tl.constexpr,   # tile size along H
):
    r = tl.program_id(axis=0)  # one program per row
    sumsq = 0.0
    # Loop over tiles of size BLOCK_SIZE
    for offs in range(0, H, BLOCK_SIZE):
        col = offs + tl.arange(0, BLOCK_SIZE)
        mask = col < H
        x = tl.load(x_ptr + r * H + col, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + r, inv_rms)

# Kernel 2: scale row elements by inv_rms[row] and weight[j]
@triton.jit
def scale_row_elements_kernel(
    x_ptr,            # *f32, [B, H]
    weight_ptr,       # *f32, [H]
    inv_rms_ptr,      # *f32, [B]
    out_ptr,          # *f32, [B, H]
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # hidden size
    BLOCK_SIZE: tl.constexpr,  # tile size along H
):
    r = tl.program_id(axis=0)  # one program per row
    f = tl.load(inv_rms_ptr + r)  # per-row factor
    for offs in range(0, H, BLOCK_SIZE):
        col = offs + tl.arange(0, BLOCK_SIZE)
        mask = col < H
        x = tl.load(x_ptr + r * H + col, mask=mask, other=0.0)
        w = tl.load(weight_ptr + col, mask=mask, other=0.0)
        y = x * f * w
        tl.store(out_ptr + r * H + col, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguity; compute in fp32
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        B, H = hidden_states.shape
        # Specialized fast path for H == 4096
        if H == 4096:
            x = hidden_states.contiguous().to(torch.float32)  # [B, H]
            weight_fp32 = weight.contiguous().to(torch.float32)  # [H]
            out = torch.empty((B, H), dtype=torch.float32, device=x.device)
            inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)
            EPS = 1e-5
            # Launch reduction kernel: one program per row
            grid = (B,)
            reduce_row_sumsq_kernel[grid](
                x, inv_rms, B, H, EPS,
                BLOCK_SIZE=4096,
                num_warps=8,
                num_stages=4,
            )
            # Launch scaling kernel: one program per row
            scale_row_elements_kernel[grid](
                x, weight_fp32, inv_rms, out, B, H,
                BLOCK_SIZE=4096,
                num_warps=8,
                num_stages=4,
            )
            return out.to(hidden_states.dtype)
        else:
            # General fallback path for arbitrary H
            x = hidden_states.contiguous().to(torch.float32)  # [B, H]
            weight_fp32 = weight.contiguous().to(torch.float32)  # [H]
            out = torch.empty((B, H), dtype=torch.float32, device=x.device)
            inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)
            EPS = 1e-5
            # Launch reduction kernel: one program per row
            grid = (B,)
            reduce_row_sumsq_kernel[grid](
                x, inv_rms, B, H, EPS,
                BLOCK_SIZE=1024,
                num_warps=4,
                num_stages=3,
            )
            # Launch scaling kernel: one program per row
            scale_row_elements_kernel[grid](
                x, weight_fp32, inv_rms, out, B, H,
                BLOCK_SIZE=1024,
                num_warps=4,
                num_stages=3,
            )
            return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
