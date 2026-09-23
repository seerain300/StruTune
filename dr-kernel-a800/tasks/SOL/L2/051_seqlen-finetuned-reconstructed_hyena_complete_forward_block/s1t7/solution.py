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
    # Linearized 1D indexing into a 2D buffer (N, D)
    grid = tl.num_programs(0)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = offs < total
    i = offs // D
    d = offs % D
    out_ptrs = out_ptr + i * out_stride0 + d * out_stride1
    tl.store(out_ptrs, 0.0, mask=mask)


@triton.jit
def gate_forward_kernel(
    v_in_ptr, gate_ptr, out_ptr,
    N, D,
    v_in_stride0, v_in_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # Elementwise: out[i, d] = v_in[i, d] * gate[i, d]
    grid = tl.num_programs(0)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = offs < total
    i = offs // D
    d = offs % D
    v_ptrs = v_in_ptr + i * v_in_stride0 + d * v_in_stride1
    g_ptrs = gate_ptr + i * gate_ptr.stride(0) + d * gate_ptr.stride(1)
    o_ptrs = out_ptr + i * out_stride0 + d * out_stride1
    v = tl.load(v_ptrs, mask=mask, other=0.0)
    g = tl.load(g_ptrs, mask=mask, other=1.0)
    out = v * g
    tl.store(o_ptrs, out, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    v_ptr, t_ptr, deltas_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # Elementwise over (N, D): out[i, d] = v[i, d] * (exp(-t[i] * deltas[d]) + shift)
    # shift is set to 0.05 as per original code.
    grid = tl.num_programs(0)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = offs < total
    i = offs // D
    d = offs % D
    v_ptrs = v_ptr + i * v_stride0 + d * v_stride1
    o_ptrs = out_ptr + i * out_stride0 + d * out_stride1
    t_i = tl.load(t_ptr + i, mask=mask, other=0.0)          # scalar per row i
    deltas_d = tl.load(deltas_ptr + d, mask=mask, other=0.0)  # per column d
    shift = 0.05
    out = tl.load(v_ptrs, mask=mask, other=0.0) * (tl.exp(-t_i * deltas_d) + shift)
    tl.store(o_ptrs, out, mask=mask)


@triton.jit
def add_residual_kernel(
    v_ptr, residual_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # Elementwise: out[i, d] = v[i, d] + residual[i, d]
    grid = tl.num_programs(0)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = offs < total
    i = offs // D
    d = offs % D
    v_ptrs = v_ptr + i * v_stride0 + d * v_stride1
    res_ptrs = residual_ptr + i * residual_ptr.stride(0) + d * residual_ptr.stride(1)
    o_ptrs = out_ptr + i * out_stride0 + d * out_stride1
    v = tl.load(v_ptrs, mask=mask, other=0.0)
    res = tl.load(res_ptrs, mask=mask, other=0.0)
    tl.store(o_ptrs, v + res, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr, start, end, length,
    BLOCK: tl.constexpr
):
    grid = tl.num_programs(0)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    step = (end - start) / length
    values = start + offs * step
    tl.store(out_ptr + offs, values, mask=mask)


@triton.jit
def ones_1d_kernel(
    out_ptr, length,
    BLOCK: tl.constexpr
):
    grid = tl.num_programs(0)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    ones = tl.full([BLOCK], 1.0, tl.float32)
    tl.store(out_ptr + offs, ones, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, batch_size: int, seq_len: int):
        # Launch the required Triton kernels exactly once; forward must not perform
        # any torch computation except for allocating buffers (to return a tensor).
        N = batch_size
        D = 256  # consistent with the original code
        total = N * D

        # Allocate 2D output buffer (contiguous) and pass its 1D view to kernels
        out = torch.empty((N, D), device='cuda', dtype=torch.float32)
        out1d = out.view(-1)

        # 1) create_2d_buffer_kernel: initialize out with zeros (placeholder)
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        create_2d_buffer_kernel[grid](
            out1d,
            N, D,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 2) linspace_1d_kernel: t = linspace(0, seq_len-1, seq_len)
        L = seq_len
        t = torch.empty((L,), device='cuda', dtype=torch.float32)
        grid_t = (triton.cdiv(L, BLOCK),)
        linspace_1d_kernel[grid_t](t, 0.0, (L - 1), L, BLOCK=BLOCK)

        # 3) ones_1d_kernel: gate vector of length total
        gate_1d = torch.empty((total,), device='cuda', dtype=torch.float32)
        grid_gate = (triton.cdiv(total, BLOCK),)
        ones_1d_kernel[grid_gate](gate_1d, total, BLOCK=BLOCK)

        # 4) gate_forward_kernel: out = out * gate (elementwise)
        gate_forward_kernel[grid](
            out1d, gate_1d, out1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 5) exp_mod_apply_kernel: out = out * (exp(-t[i] * deltas[d]) + 0.05)
        # deltas is 1D linspace(0, D-1, D)
        deltas = torch.empty((D,), device='cuda', dtype=torch.float32)
        grid_d = (triton.cdiv(D, BLOCK),)
        linspace_1d_kernel[grid_d](deltas, 0.0, (D - 1), D, BLOCK=BLOCK)
        exp_mod_apply_kernel[grid](
            out1d, t, deltas, out1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 6) add_residual_kernel: out = out + out (self-add)
        add_residual_kernel[grid](
            out1d, out1d, out1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # Return the final tensor (no torch computation in host)
        return out


def run(*args):
    return ModelNew()(*args)
