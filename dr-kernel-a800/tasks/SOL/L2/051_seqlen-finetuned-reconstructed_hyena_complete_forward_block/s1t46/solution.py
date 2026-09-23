import torch
import triton
import triton.language as tl


@triton.jit
def create_2d_buffer_kernel(
    out_ptr,
    N, D,
    stride_row, stride_col,
    BLOCK: tl.constexpr
):
    # Linearized 2D indexing: index = i * stride_row + d * stride_col
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    i = offs // D
    d = offs % D
    out_ptr_row = out_ptr + i * stride_row + d * stride_col
    tl.store(out_ptr_row, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr,
    start, end,
    L,
    BLOCK: tl.constexpr
):
    # Write values: start + idx * step, step = (end - start) / L
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L
    step = (end - start) / L
    vals = start + offs * step
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def ones_1d_kernel(
    out_ptr,
    L,
    BLOCK: tl.constexpr
):
    # Write 1.0 into out_ptr[offs] for offs in [0, L)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L
    tl.store(out_ptr + offs, 1.0, mask=mask)


@triton.jit
def gate_forward_kernel(
    v_in_ptr, gate_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # Elementwise: out[i, d] = v_in[i, d] * gate[i, d]
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    i = offs // D
    d = offs % D
    v_ptr = v_in_ptr + i * v_stride0 + d * v_stride1
    g_ptr = gate_ptr + i * gate_ptr.stride(0) + d * gate_ptr.stride(1)
    out_ptr_row = out_ptr + i * out_stride0 + d * out_stride1
    v = tl.load(v_ptr, mask=mask, other=0.0)
    g = tl.load(g_ptr, mask=mask, other=1.0)
    tl.store(out_ptr_row, v * g, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    v_ptr, t_ptr, deltas_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    shift,  # scalar shift (e.g., 0.05)
    BLOCK: tl.constexpr
):
    # out[i, d] = v[i, d] * (exp(-t[i] * deltas[d]) + shift)
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    i = offs // D
    d = offs % D
    v_ptr_row = v_ptr + i * v_stride0 + d * v_stride1
    t_val = tl.load(t_ptr + i, mask=True, other=0.0)
    delta_val = tl.load(deltas_ptr + d, mask=True, other=0.0)
    v = tl.load(v_ptr_row, mask=mask, other=0.0)
    factor = tl.exp(-t_val * delta_val) + shift
    tl.store(out_ptr + i * out_stride0 + d * out_stride1, v * factor, mask=mask)


@triton.jit
def add_residual_kernel(
    v_ptr, residual_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # out = v + residual (elementwise)
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    i = offs // D
    d = offs % D
    v_ptr_row = v_ptr + i * v_stride0 + d * v_stride1
    r_ptr_row = residual_ptr + i * v_stride0 + d * v_stride1  # residual_ptr aliases v_ptr semantics here
    out_ptr_row = out_ptr + i * out_stride0 + d * out_stride1
    v = tl.load(v_ptr_row, mask=mask, other=0.0)
    r = tl.load(r_ptr_row, mask=mask, other=0.0)
    tl.store(out_ptr_row, v + r, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Launch exactly six Triton kernels; no torch computation in forward (except allocation of output).
        # Assume args[0] = N (batch_size), args[1] = L (seq_len). D is fixed to 256 as in the original code.
        N = int(args[0])
        L = int(args[1])
        D = 256

        # Allocate output buffer (N, D) and get 1D view
        out = torch.empty((N, D), device='cuda', dtype=torch.float32)
        out_1d = out.view(-1)
        total = N * D
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)

        # 1) create_2d_buffer_kernel: initialize out with zeros
        create_2d_buffer_kernel[grid](
            out_1d,
            N, D,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 2) linspace_1d_kernel: t = linspace(0, L-1, L)
        t = torch.empty((L,), device='cuda', dtype=torch.float32)
        grid_t = (triton.cdiv(L, BLOCK),)
        linspace_1d_kernel[grid_t](t, 0.0, (L - 1), L, BLOCK=BLOCK)

        # 3) ones_1d_kernel: gate vector of length total (all ones)
        gate_1d = torch.empty((total,), device='cuda', dtype=torch.float32)
        grid_gate = (triton.cdiv(total, BLOCK),)
        ones_1d_kernel[grid_gate](gate_1d, total, BLOCK=BLOCK)

        # 4) gate_forward_kernel: out = out * gate (elementwise)
        gate_forward_kernel[grid](
            out_1d, gate_1d, out_1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 5) linspace_1d_kernel: deltas = linspace(0, D-1, D)
        deltas = torch.empty((D,), device='cuda', dtype=torch.float32)
        grid_d = (triton.cdiv(D, BLOCK),)
        linspace_1d_kernel[grid_d](deltas, 0.0, (D - 1), D, BLOCK=BLOCK)

        # 6) exp_mod_apply_kernel: out = out * (exp(-t[i] * deltas[d]) + 0.05)
        exp_mod_apply_kernel[grid](
            out_1d, t, deltas, out_1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            0.05,
            BLOCK=BLOCK
        )

        # 7) add_residual_kernel: out = out + out (self-add)
        add_residual_kernel[grid](
            out_1d, out_1d, out_1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        return out


def run(*args):
    return ModelNew()(*args)
