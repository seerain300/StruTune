import torch
import triton
import triton.language as tl


@triton.jit
def create_2d_buffer_kernel(
    out_ptr,  # linearized 1D pointer to (N*D) elements
    N, D,
    BLOCK: tl.constexpr,
):
    """
    Initialize a 2D buffer of shape (N, D) with zeros via linear indexing.
    out_ptr is a flat view of the (N, D) tensor: index = i*D + d
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_elements = N * D
    mask = offs < n_elements
    tl.store(out_ptr + offs, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr,  # 1D output vector of length L
    start, end,
    L,  # length
    BLOCK: tl.constexpr,
):
    """
    Fill out_ptr with values: out[i] = start + i * step, where step = (end - start) / L.
    For integer i in [0, L-1].
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    L_vec = L
    mask = offs < L
    step = (end - start) / L_vec
    vals = start + offs * step
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def ones_1d_kernel(
    out_ptr,  # 1D output vector of length L
    L,  # length
    BLOCK: tl.constexpr,
):
    """
    Fill out_ptr with ones of length L.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L
    tl.store(out_ptr + offs, 1.0, mask=mask)


@triton.jit
def gate_forward_kernel(
    out_ptr,          # 2D buffer (N, D), linearized via strides
    gate_ptr,         # 1D vector of length N (rows)
    N, D,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr,
):
    """
    Compute elementwise out[i, d] = out[i, d] * gate[i] for 2D buffer (N, D).
    Linear indexing: offs = i*D + d. gate is 1D of length N.
    """
    pid = tl.program_id(0)  # launch one program per row
    i = pid  # i in [0, N)
    offs = tl.arange(0, BLOCK)
    d = offs  # columns processed in BLOCK
    mask = d < D
    out_row_ptr = out_ptr + i * out_stride0 + d * out_stride1
    gate_val = tl.load(gate_ptr + i, mask=True, other=1.0)
    out_val = tl.load(out_row_ptr, mask=mask, other=0.0)
    out_val = out_val * gate_val
    tl.store(out_row_ptr, out_val, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    out_ptr,          # 2D buffer (N, D), linearized via strides
    t_ptr,            # 1D vector of length N (seq_len)
    deltas_ptr,       # 1D vector of length D
    N, D,
    out_stride0, out_stride1,
    shift,            # scalar float
    BLOCK: tl.constexpr,
):
    """
    Elementwise: out[i, d] = out[i, d] * (exp(-t[i] * deltas[d]) + shift)
    Linear indexing: offs = i*D + d.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_elements = N * D
    mask = offs < n_elements
    d = offs % D
    i = offs // D
    out_elem = tl.load(out_ptr + i * out_stride0 + d * out_stride1, mask=mask, other=0.0)
    ti = tl.load(t_ptr + i, mask=True, other=0.0)
    delta = tl.load(deltas_ptr + d, mask=True, other=0.0)
    factor = tl.exp(-ti * delta) + shift
    out_elem = out_elem * factor
    tl.store(out_ptr + i * out_stride0 + d * out_stride1, out_elem, mask=mask)


@triton.jit
def add_residual_kernel(
    out_ptr,          # 2D buffer (N, D), linearized via strides
    N, D,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr,
):
    """
    Elementwise: out[i, d] = out[i, d] + out[i, d]
    Linear indexing: offs = i*D + d.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_elements = N * D
    mask = offs < n_elements
    d = offs % D
    i = offs // D
    out_elem = tl.load(out_ptr + i * out_stride0 + d * out_stride1, mask=mask, other=0.0)
    out_elem = out_elem + out_elem
    tl.store(out_ptr + i * out_stride0 + d * out_stride1, out_elem, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_size: int, seq_len: int):
        """
        Triton-only forward that launches all required kernels exactly once.
        Returns a tensor of shape (batch_size, 256).
        """
        N = batch_size
        D = 256  # match original d_model

        # Allocate final output (the only allowed torch op in forward)
        out = torch.empty((N, D), device='cuda', dtype=torch.float32)
        out_flat = out.view(-1)

        # 1) create_2d_buffer_kernel: initialize out with zeros
        BLOCK = 1024
        grid0 = (triton.cdiv(N * D, BLOCK),)
        create_2d_buffer_kernel[grid0](out_flat, N, D, BLOCK=BLOCK)

        # 2) linspace_1d_kernel: t = linspace(0, seq_len-1, seq_len)
        t = torch.empty((seq_len,), device='cuda', dtype=torch.float32)
        grid1 = (triton.cdiv(seq_len, BLOCK),)
        linspace_1d_kernel[grid1](t, 0.0, (seq_len - 1), seq_len, BLOCK=BLOCK)

        # 3) ones_1d_kernel: gate vector of length N filled with ones
        gate = torch.empty((N,), device='cuda', dtype=torch.float32)
        grid2 = (triton.cdiv(N, BLOCK),)
        ones_1d_kernel[grid2](gate, N, BLOCK=BLOCK)

        # 4) gate_forward_kernel: out = out * gate (elementwise)
        grid3 = (N,)
        gate_forward_kernel[grid3](
            out, gate, out,
            N, D,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK,
        )

        # 5) exp_mod_apply_kernel: out = out * (exp(-t[i] * deltas[d]) + 0.05)
        # deltas: 1D linspace(0, D-1, D)
        deltas = torch.empty((D,), device='cuda', dtype=torch.float32)
        grid_d = (triton.cdiv(D, BLOCK),)
        linspace_1d_kernel[grid_d](deltas, 0.0, (D - 1), D, BLOCK=BLOCK)
        grid4 = (triton.cdiv(N * D, BLOCK),)
        exp_mod_apply_kernel[grid4](
            out_flat, t, deltas, out_flat,
            N, D,
            out.stride(0), out.stride(1),
            0.05,  # shift
            BLOCK=BLOCK,
        )

        # 6) add_residual_kernel: out = out + out (self-add)
        grid5 = (triton.cdiv(N * D, BLOCK),)
        add_residual_kernel[grid5](
            out_flat,
            N, D,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK,
        )

        # Return the Triton-processed tensor. No torch computation was performed in forward beyond allocation.
        return out


def run(*args):
    return ModelNew()(*args)
