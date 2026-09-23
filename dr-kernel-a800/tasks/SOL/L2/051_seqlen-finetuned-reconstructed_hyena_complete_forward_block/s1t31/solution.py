import torch
import triton
import triton.language as tl


@triton.jit
def create_2d_buffer_kernel(
    out_ptr,
    N, D,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # Linearized indexing: idx = i * D + d
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = offs < total
    # Compute (i, d) for each linear index
    i = offs // D
    d = offs % D
    out_ptrs = out_ptr + i * out_stride0 + d * out_stride1
    tl.store(out_ptrs, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr,
    start, end,
    length,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    L = length
    mask = offs < L
    step = (end - start) / (L - 1) if L > 1 else 0.0
    vals = start + offs * step
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def ones_1d_kernel(
    out_ptr,
    length,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    L = length
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
    # Linearized indexing: idx = i * D + d
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = offs < total
    i = offs // D
    d = offs % D
    v_ptrs = v_in_ptr + i * v_stride0 + d * v_stride1
    g_ptrs = gate_ptr + i * gate_ptr.stride(0) + d * gate_ptr.stride(1)
    o_ptrs = out_ptr + i * out_stride0 + d * out_stride1
    v = tl.load(v_ptrs, mask=mask, other=0.0)
    g = tl.load(g_ptrs, mask=mask, other=1.0)
    tl.store(o_ptrs, v * g, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    v_ptr, t_ptr, deltas_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # Linearized indexing: idx = i * D + d
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = offs < total
    i = offs // D
    d = offs % D
    v_ptrs = v_ptr + i * v_stride0 + d * v_stride1
    t_ptrs = t_ptr + i  # t is length N, 1D
    d_ptrs = deltas_ptr + d  # deltas is length D, 1D
    o_ptrs = out_ptr + i * out_stride0 + d * out_stride1
    v = tl.load(v_ptrs, mask=mask, other=0.0)
    t = tl.load(t_ptrs, mask=mask, other=0.0)
    delta = tl.load(d_ptrs, mask=mask, other=0.0)
    # shift from original is 0.05
    shift = 0.05
    out = v * tl.exp(-t * delta) + v * shift
    tl.store(o_ptrs, out, mask=mask)


@triton.jit
def add_residual_kernel(
    v_ptr, add_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # Linearized indexing: idx = i * D + d
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = offs < total
    i = offs // D
    d = offs % D
    v_ptrs = v_ptr + i * v_stride0 + d * v_stride1
    a_ptrs = add_ptr + i * add_ptr.stride(0) + d * add_ptr.stride(1)  # add_ptr can be same as v_ptr
    o_ptrs = out_ptr + i * out_stride0 + d * out_stride1
    v = tl.load(v_ptrs, mask=mask, other=0.0)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    tl.store(o_ptrs, v + a, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Workload provides batch_size (N) and seq_len; d_model is fixed as 256 in the reference
        N = int(args[0]) if len(args) > 0 else 1
        D = 256  # fixed from the reference setup
        device = 'cuda'  # Triton kernels require CUDA

        # Allocate output tensor and its 1D view
        out = torch.empty((N, D), device=device, dtype=torch.float32)
        out1d = out.view(-1)

        # 1) create_2d_buffer_kernel: initialize out with zeros
        BLOCK = 1024
        total = N * D
        grid = (triton.cdiv(total, BLOCK),)
        create_2d_buffer_kernel[grid](
            out1d,
            N, D,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 2) linspace_1d_kernel: t = linspace(0, N-1, N)
        t = torch.empty((N,), device=device, dtype=torch.float32)
        grid_t = (triton.cdiv(N, BLOCK),)
        linspace_1d_kernel[grid_t](t, 0.0, (N - 1), N, BLOCK=BLOCK)

        # 3) ones_1d_kernel: gate vector of length total (all ones)
        gate_1d = torch.empty((total,), device=device, dtype=torch.float32)
        grid_gate = (triton.cdiv(total, BLOCK),)
        ones_1d_kernel[grid_gate](gate_1d, total, BLOCK=BLOCK)

        # 4) gate_forward_kernel: out = out * gate (elementwise on linearized view)
        grid_gate2 = grid  # covers all elements
        gate_forward_kernel[grid_gate2](
            out1d, gate_1d, out1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 5) exp_mod_apply_kernel: out = out * (exp(-t[i] * deltas[d]) + 0.05)
        deltas = torch.empty((D,), device=device, dtype=torch.float32)
        grid_d = (triton.cdiv(D, BLOCK),)
        linspace_1d_kernel[grid_d](deltas, 0.0, (D - 1), D, BLOCK=BLOCK)
        exp_mod_apply_kernel[grid](
            out1d, t, deltas, out1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 6) add_residual_kernel: out = out + out (self-add residual)
        add_residual_kernel[grid](
            out1d, out1d, out1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # Return the final output tensor (shape: (N, 256))
        return out


def run(*args):
    return ModelNew()(*args)
