import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _normalize_scale_kernel(hidden_ptr, weight_ptr, out_ptr, sumsq_ptr, B, H, stride_hs, BLOCK_SIZE: tl.constexpr):
    """
    One program per row. Two passes:
    1) Accumulate sum of squares in FP32 and compute inv_rms = rsqrt(mean + EPS).
    2) Re-read row, multiply by inv_rms and weight, write FP32 output.
    """
    row = tl.program_id(0)
    # If grid exactly B, no out-of-bounds, but keep a guard (safe when grid=B)
    # Pass 1: compute sum of squares
    sumsq = 0.0
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(hidden_ptr + row * stride_hs + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        # Ensure masked lanes don't contribute
        x32 = tl.where(mask, x32, 0.0)
        sumsq += tl.sum(x32 * x32, axis=0)
    inv_rms = tl.rsqrt(sumsq / H + EPS)
    tl.store(sumsq_ptr + row, inv_rms)

    # Pass 2: compute final output y = x * inv_rms * weight
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(hidden_ptr + row * stride_hs + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_ptr + row * H + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous
        if not hidden_states.is_cuda:
            hidden_states = hidden_states.cuda(non_blocking=True)
        if not weight.is_cuda:
            weight = weight.cuda(non_blocking=True)
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape
        # Output will be FP32 in kernel; cast to original dtype after
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)

        # Per-row inv_rms storage (FP32)
        sumsq = torch.empty((B,), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        # Choose BLOCK_SIZE to vectorize over columns; 1024 works well for H=4096
        _normalize_scale_kernel[grid](
            hidden_states, weight, out_fp32, sumsq,
            B, H, hidden_states.stride(0),
            BLOCK_SIZE=1024,
            num_warps=4,  # reasonable default for such a small row
            num_stages=2,
        )

        # Cast to original dtype to match original behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
