import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,          # *bf16  [batch, N]  input hidden_states
    w_ptr,          # *bf16  [N]         weight
    y_ptr,          # *bf16  [batch, N]  output
    N: tl.constexpr,        # hidden_size (== 4096)
    EPS: tl.constexpr,      # 1e-5
    BLOCK_H: tl.constexpr,  # == N, power of two -> no masking
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)

    # Single global load of the row; reused for the output (no second load of x).
    x = tl.load(x_ptr + row * N + offs).to(tl.float32)

    # fp32 accumulation of sum of squares -> mean -> +eps -> rsqrt (exact ref order).
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_sq / N + EPS)

    # Weight promoted to fp32 before the multiply.
    w = tl.load(w_ptr + offs).to(tl.float32)

    y = (x * inv_rms) * w

    # Single round-to-nearest-even cast back to bf16 at store.
    tl.store(y_ptr + row * N + offs, y.to(tl.bfloat16))


def run(hidden_states, weight):
    batch, hidden = hidden_states.shape
    out = torch.empty_like(hidden_states)
    grid = (batch,)
    _rmsnorm_kernel[grid](
        hidden_states,
        weight,
        out,
        N=hidden,
        EPS=1e-5,
        BLOCK_H=hidden,
        num_warps=8,
    )
    return out
