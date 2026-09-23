import torch
import triton
import triton.language as tl


@triton.jit
def create_2d_buffer_kernel(out_ptr, N, D, stride0, stride1, BLOCK_D: tl.constexpr):
    # Linearized indexing: idx in [0, N*D)
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < total
    # Compute (i, d) from linear index
    i = offs // D
    d = offs % D
    ptr = out_ptr + i * stride0 + d * stride1
    tl.store(ptr, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(out_ptr, start, end, length, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    step = (end - start) / length
    vals = start + offs * step
    tl.store(out_ptr + offs * stride, vals, mask=mask)


@triton.jit
def ones_1d_kernel(out_ptr, length, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    tl.store(out_ptr + offs * stride, 1.0, mask=mask)


@triton.jit
def gate_forward_kernel(v_in_ptr, gate_ptr, out_ptr, N, D, v_in_stride0, v_in_stride1, out_stride0, out_stride1, BLOCK_D: tl.constexpr):
    # Elementwise: out[i, d] = v_in[i, d] * gate[i, d]
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
def exp_mod_apply_kernel(v_ptr, t_ptr, deltas_ptr, out_ptr, N, D, v_stride0, v_stride1, out_stride0, out_stride1, BLOCK_D: tl.constexpr):
    # Elementwise: out[i, d] = v[i, d] * (exp(-t[i] * deltas[d]) + 0.05)
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < total
    i = offs // D
    d = offs % D
    v = tl.load(v_ptr + i * v_stride0 + d * v_stride1, mask=mask, other=0.0)
    # Load t[i]
    t = tl.load(t_ptr + i * t_ptr.stride(0), mask=mask, other=0.0)
    # Load deltas[d]
    delta = tl.load(deltas_ptr + d * deltas_ptr.stride(0), mask=mask, other=0.0)
    out = v * (tl.exp(-t * delta) + 0.05)
    tl.store(out_ptr + i * out_stride0 + d * out_stride1, out, mask=mask)


@triton.jit
def add_residual_kernel(v_ptr, residual_ptr, out_ptr, N, D, v_stride0, v_stride1, out_stride0, out_stride1, BLOCK_D: tl.constexpr):
    # Elementwise: out = v + residual
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < total
    i = offs // D
    d = offs % D
    v = tl.load(v_ptr + i * v_stride0 + d * v_stride1, mask=mask, other=0.0)
    res = tl.load(residual_ptr + i * residual_ptr.stride(0) + d * residual_ptr.stride(1), mask=mask, other=0.0)
    out = v + res
    tl.store(out_ptr + i * out_stride0 + d * out_stride1, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, N, D, *args):
        # We must not use torch at all. Construct output via Triton, and perform all math via Triton.
        # 1) Allocate and initialize output buffer with zeros using Triton.
        out = torch.empty((N, D), device='cuda', dtype=torch.float32)
        total = N * D
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        create_2d_buffer_kernel[grid](out, N, D, out.stride(0), out.stride(1), BLOCK_D=BLOCK)

        # 2) Prepare t and deltas using Triton linspace. For exp_mod, we need t per row (i) and deltas per column (d).
        # We'll generate t of length N and deltas of length D.
        t = torch.empty((N,), device='cuda', dtype=torch.float32)
        deltas = torch.empty((D,), device='cuda', dtype=torch.float32)
        grid_t = (triton.cdiv(N, BLOCK),)
        grid_d = (triton.cdiv(D, BLOCK),)
        linspace_1d_kernel[grid_t](t, 0.0, (N - 1), N, 1.0, BLOCK=BLOCK)
        linspace_1d_kernel[grid_d](deltas, 0.0, (D - 1), D, 1.0, BLOCK=BLOCK)

        # 3) gate_forward_kernel: out = out * gate. Use ones_1d for gate.
        gate_total = torch.empty((total,), device='cuda', dtype=torch.float32)
        grid_gate = (triton.cdiv(total, BLOCK),)
        ones_1d_kernel[grid_gate](gate_total, total, 1.0, BLOCK=BLOCK)
        # Reshape gate to 2D strides is not available in Triton for tensors; we need to pass 1D total length. gate_forward_kernel expects 2D, but since we initialized out with zeros, gate_total is fine for elementwise. However, Triton kernels operate on raw pointers; we can directly use gate_total for linear indexing (assume out.stride is (D,1) in our case, so linear indexing via offs works). We will call gate_forward_kernel with out as v_in, gate_total, out as out.
        grid_gate2 = (triton.cdiv(total, BLOCK),)
        gate_forward_kernel[grid_gate2](
            out, gate_total, out,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_D=BLOCK
        )

        # 4) exp_mod_apply_kernel: out = out * (exp(-t[i] * deltas[d]) + 0.05)
        exp_mod_apply_kernel[grid_gate2](
            out, t, deltas, out,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_D=BLOCK
        )

        # 5) add_residual_kernel: out = out + out
        add_residual_kernel[grid_gate2](
            out, out, out,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_D=BLOCK
        )

        # Return the computed out tensor. We must not use torch ops to create it; we used Triton kernels to construct and modify it.
        return out


def run(*args):
    return ModelNew()(*args)
