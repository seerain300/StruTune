import torch
import torch.nn.functional as F

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: pad 1D tensor along last dimension by adding pad_size zeros
# Input: X: [S] (1D contiguous), Output: Y: [S + pad_size] contiguous
@triton.jit
def pad_1d_kernel(X_ptr, Y_ptr, S, pad_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    out_idx = pid
    total = S + pad_size
    if out_idx < total:
        if out_idx < S:
            tl.store(Y_ptr + out_idx, tl.load(X_ptr + out_idx))
        else:
            tl.store(Y_ptr + out_idx, 0.0)
    # Process BLOCK_SIZE elements per program
    offs = tl.arange(0, BLOCK_SIZE)
    base = pid * BLOCK_SIZE
    idx = base + offs
    mask = idx < total
    tl.store(Y_ptr + idx, tl.load(X_ptr + idx, mask=idx < S, other=0.0), mask=mask)


# Triton kernel: elementwise multiply D[h, d] * X[b, s, h, d] -> Y[b, s, h, d] in bfloat16
@triton.jit
def d_residual_mul_kernel(D_ptr, X_ptr, Y_ptr,
                          BATCH, S_tot, HEADS, HEAD_DIM,
                          BLOCK_SIZE: tl.constexpr):
    # Each program handles BLOCK_SIZE elements over flattened index
    pid = tl.program_id(axis=0)
    base = pid * BLOCK_SIZE
    offs = base + tl.arange(0, BLOCK_SIZE)
    total = BATCH * S_tot * HEADS * HEAD_DIM
    mask = offs < total

    # Map linear index to (b, s, h, d)
    dh = HEADS * HEAD_DIM
    b = offs // (S_tot * dh)
    rem = offs % (S_tot * dh)
    h = rem // (S_tot)
    rem2 = rem % (S_tot)
    s = rem2 // HEAD_DIM
    d = rem2 % HEAD_DIM

    x_index = b * S_tot * dh + s * dh + h * D + d
    # D is [HEADS, HEAD_DIM] contiguous; index = h * D + d
    d_index = h * HEAD_DIM + d
    y_index = x_index

    x_val = tl.load(X_ptr + x_index, mask=mask, other=0.0)
    d_val = tl.load(D_ptr + d_index, mask=mask, other=0.0)

    y_val = x_val * d_val
    y_val = y_val.to(tl.bfloat16)
    tl.store(Y_ptr + y_index, y_val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        # Constants from original
        BATCH = hidden_states.shape


def run(*args):
    return ModelNew()(*args)
