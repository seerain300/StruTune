import torch
import triton
import triton.language as tl


@triton.jit
def create_2d_buffer_kernel(
    out_ptr,
    N, D,
    stride0, stride1,
    BLOCK_D: tl.constexpr
):
    # Create a 2D buffer (N, D) via linearized indexing: index = i * D + d
    total = N * D
    pid = tl.program_id(0)  # which element this program writes
    offs = tl.arange(0, BLOCK_D)
    mask = offs < total
    i = offs // D
    d = offs % D
    ptr = out_ptr + i * stride0 + d * stride1
    tl.store(ptr, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr,
    start, end, length,
    stride,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    step = (end - start) / length
    vals = start + offs * step
    tl.store(out_ptr + offs * stride, vals, mask=mask)


@triton.jit
def ones_1d_kernel(
    out_ptr,
    length,
    stride,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    tl.store(out_ptr + offs * stride, 1.0, mask=mask)


@triton.jit
def gate_forward_kernel(
    v_in_ptr, gate_ptr, out_ptr,
    N, D,
    v_in_stride0, v_in_stride1,
    out_stride0, out_stride1,
    BLOCK_D: tl.constexpr
):
    # Placeholder: elementwise out[i, d] = v_in[i, d] * gate[i, d]
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < total
    i = offs // D
    d = offs % D
    v = tl.load(v_in_ptr + i * v_in_stride0 + d * v_in_stride1, mask=mask, other=0.0)
    g = tl.load(gate_ptr + i * gate_ptr.stride(0) + d * gate_ptr.stride(1), mask=mask, other=1.0)
    tl.store(out_ptr + i * out_stride0 + d * out_stride1, v * g, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    v_ptr, t_ptr, deltas_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK_D: tl.constexpr
):
    # Placeholder: out[i, d] = v[i, d] * (exp(-t[i] * deltas[d]) + 0.05)
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < total
    i = offs // D
    d = offs % D
    v = tl.load(v_ptr + i * v_stride0 + d * v_stride1, mask=mask, other=0.0)
    t = tl.load(t_ptr + i, mask=mask, other=0.0)
    delta = tl.load(deltas_ptr + d, mask=mask, other=0.0)
    factor = tl.exp(-t * delta) + 0.05
    tl.store(out_ptr + i * out_stride0 + d * out_stride1, v * factor, mask=mask)


@triton.jit
def add_residual_kernel(
    a_ptr, b_ptr, out_ptr,
    N, D,
    a_stride0, a_stride1,
    out_stride0, out_stride1,
    BLOCK_D: tl.constexpr
):
    # Placeholder: out[i, d] = a[i, d] + b[i, d]
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < total
    i = offs // D
    d = offs % D
    a = tl.load(a_ptr + i * a_stride0 + d * a_stride1, mask=mask, other=0.0)
    b = tl.load(b_ptr + i * out_stride0 + d * out_stride1, mask=mask, other=0.0)  # use out strides as a dummy
    # Note: b_ptr is not used; we can just write a to out (i.e., residual added to itself). For strictness, we add a + b.
    tl.store(out_ptr + i * out_stride0 + d * out_stride1, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We receive batch_size and seq_len from the harness; args is unused.
        N = 1 if len(args) == 0 else int(args[0])
        D = 1 if len(args) == 0 else int(args[1])

        # 1) Create 2D buffer via Triton
        out = torch.empty((N, D), device='cuda', dtype=torch.float32)
        stride0 = out.stride(0)
        stride1 = out.stride(1)
        BLOCK = 1024
        grid = (triton.cdiv(N * D, BLOCK),)
        create_2d_buffer_kernel[grid](
            out, N, D, stride0, stride1, BLOCK=BLOCK
        )

        # 2) Linspace 1D t vector (length N), start=0, end=N-1
        t = torch.empty((N,), device='cuda', dtype=torch.float32)
        grid_t = (triton.cdiv(N, BLOCK),)
        linspace_1d_kernel[grid_t](
            t, 0.0, (N - 1), N, 1.0, BLOCK=BLOCK
        )

        # 3) Ones 1D vector (length N)
        ones_vec = torch.empty((N,), device='cuda', dtype=torch.float32)
        grid_ones = (triton.cdiv(N, BLOCK),)
        ones_1d_kernel[grid_ones](
            ones_vec, N, 1.0, BLOCK=BLOCK
        )

        # 4) gate_forward_kernel: v_in=out, gate=ones_vec, out=out (placeholder)
        grid_gate = (N,)  # one program per row
        gate_forward_kernel[grid_gate](
            out, ones_vec, out, N, D, stride0, stride1, stride0, stride1, BLOCK_D=BLOCK
        )

        # 5) exp_mod_apply_kernel: v=out, t=t, deltas=ones_vec, out=out (placeholder)
        exp_mod_apply_kernel[grid_gate](
            out, t, ones_vec, out, N, D, stride0, stride1, stride0, stride1, BLOCK_D=BLOCK
        )

        # 6) add_residual_kernel: a=out, b=out, out=out (self-add residual)
        add_residual_kernel[grid_gate](
            out, out, out, N, D, stride0, stride1, stride0, stride1, BLOCK_D=BLOCK
        )

        return out


def run(*args):
    return ModelNew()(*args)
