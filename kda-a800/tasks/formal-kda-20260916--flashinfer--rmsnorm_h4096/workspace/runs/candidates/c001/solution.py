import torch
import triton
import triton.language as tl

# rmsnorm_h4096 — fused one-pass RMSNorm (design A: one program per row).
# Numerics mirror the reference exactly:
#   ms  = mean_j (fp32(x)^2)              (fp32 accumulate, denom H=4096)
#   inv = rsqrt(ms + 1e-5)                (eps on mean-of-squares)
#   y   = (fp32(x) * inv) * fp32(w)       (no centering; order (x*inv)*w)
#   out = bf16(y)                         (single final downcast, RNE)

H = 4096
EPS = 1e-5


@triton.jit
def _rmsnorm_kernel(
    x_ptr,          # *bf16  [B, H]
    w_ptr,          # *bf16  [H]
    out_ptr,        # *bf16  [B, H]
    x_row_stride,   # row stride of x (elements)
    out_row_stride, # row stride of out (elements)
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # == H (power of two) -> no reduction mask
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + row * x_row_stride + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)

    sumsq = tl.sum(x * x, axis=0)          # fp32 reduction
    inv = tl.rsqrt(sumsq / H + EPS)        # fp32
    y = (x * inv) * w                      # fp32 throughout

    tl.store(out_ptr + row * out_row_stride + cols, y.to(tl.bfloat16))


def run(hidden_states, weight):
    # PyTorch used for metadata / launch plumbing only (no compute fallback).
    assert hidden_states.shape[1] == H

    # Contiguity safeguard (no-op / no copy when inputs are already contiguous).
    hidden_states = hidden_states.contiguous()
    weight = weight.contiguous()

    B = hidden_states.shape[0]
    out = torch.empty_like(hidden_states)

    grid = (B,)
    _rmsnorm_kernel[grid](
        hidden_states,
        weight,
        out,
        hidden_states.stride(0),
        out.stride(0),
        H=H,
        EPS=EPS,
        BLOCK_SIZE=H,
        num_warps=8,
    )
    return out
