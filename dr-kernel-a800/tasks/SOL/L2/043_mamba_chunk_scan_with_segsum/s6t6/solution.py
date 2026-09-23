import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: Pad along seq_len by appending zeros on the right.
# Input: hidden_states [B, L], Output: hidden_padded [B, L_out], L_out = L + pad_right.
@triton.jit
def pad_seq_kernel(
    in_ptr,            # *float32, input [B, L]
    out_ptr,           # *float32, output [B, L_out]
    L: tl.constexpr,   # original seq_len
    L_out: tl.constexpr,  # padded seq_len
    pad_right: tl.constexpr  # number of zeros to append on the right
):
    b = tl.program_id(0)
    pos = tl.program_id(1)
    if pos < L:
        val = tl.load(in_ptr + b * L + pos)
        tl.store(out_ptr + b * L_out + pos, val)
    else:
        tl.store(out_ptr + b * L_out + pos, 0.0)


# Triton kernel: Create lower-triangular mask [I, I] with diagonal=-1:
# out[i, j] = 1.0 if i >= j - 1 else 0.0. I = chunk_size (256).
@triton.jit
def lower_tri_mask_kernel(out_ptr, I: tl.constexpr):
    rows = tl.arange(0, I)
    cols = tl.arange(0, I)
    row_idx = rows[:, None]  # [I, 1]
    col_idx = cols[None, :]  # [1, I]
    cond = row_idx >= (col_idx - 1)  # diagonal=-1 => i >= j-1
    out_val = tl.where(cond, 1.0, 0.0)
    offsets = row_idx * I + col_idx
    tl.store(out_ptr + offsets, out_val)


# Triton kernel: Inclusive per-row cumsum on each row of a [I, I] matrix.
# Input and Output pointers point to the same [I, I] contiguous buffer.
@triton.jit
def per_row_cumsum_kernel(in_ptr, out_ptr, I: tl.constexpr):
    r = tl.program_id(0)  # row index
    cols = tl.arange(0, I)
    in_row = in_ptr + r * I + cols
    out_row = out_ptr + r * I + cols
    acc = 0.0
    for j in range(0, I):
        val = tl.load(in_row + j)
        acc = acc + val
        tl.store(out_row + j, acc)


# Triton kernel: Compute diagonal output term Y_diag:
# Y[b, nc, i, h, d] = sum_j M[b, nc, i, j, h] * V[b, nc, j, h, d]
# Grid: (B, N, H, D). Each program handles one (b, nc, h, d) and loops over i, j.
@triton.jit
def y_diag_triton_kernel(
    M_ptr,  # *float32, [B, N, I, H, D] contiguous
    V_ptr,  # *float32, [B, N, I, H, D] contiguous
    Y_ptr,  # *float32, [B, N, I, H, D] contiguous
    B: tl.constexpr, N: tl.constexpr, I: tl.constexpr, H: tl.constexpr, D: tl.constexpr
):
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)
    d = tl.program_id(3)

    total_stride = I * H * D
    for i in range(0, I):
        acc = 0.0
        for j in range(0, I):
            base = ((b * N) + nc) * total_stride + (h * D) + d
            m_offset = base + j * (H * D) + i * (H * D)  # i*(H*D) term accounts for row index
            v_offset = base + j * (H * D)
            M_val = tl.load(M_ptr + m_offset)
            V_val = tl.load(V_ptr + v_offset)
            acc = acc + M_val * V_val
        y_offset = base + i * (H * D)
        tl.store(Y_ptr + y_offset, acc)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_size = 256
        self.state_size = 256
        self.n_groups = 1
        self.num_heads = 16
        self.head


def run(*args):
    return ModelNew()(*args)
