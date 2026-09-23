import torch
import triton
import triton.language as tl


@triton.jit
def create_2d_buffer_kernel(
    out_ptr,  # 1D linearized pointer to (N*256) elements
    N, D,  # D is 256
    out_stride0, out_stride1,  # API placeholders; not needed for linear indexing
    BLOCK: tl.constexpr
):
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    tl.store(out_ptr + offs, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr,  # 1D output pointer
    start, end, length,  # length: number of elements
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    step = (end - start) / (length - 1) if length > 1 else 0.0
    val = start + offs * step
    tl.store(out_ptr + offs, val, mask=mask)


@triton.jit
def ones_1d_kernel(
    out_ptr,
    length,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    tl.store(out_ptr + offs, 1.0, mask=mask)


@triton.jit
def gate_forward_kernel(
    v_in_ptr, gate_ptr, out_ptr,
    N, D,  # D is 256
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    v = tl.load(v_in_ptr + offs, mask=mask, other=0.0)
    g = tl.load(gate_ptr + offs, mask=mask, other=1.0)
    out = v * g
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    v_ptr, t_ptr, deltas_ptr, out_ptr,
    N, D,  # D is 256
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Map linear index to (i, d): i = offs // D, d = offs % D
    i = offs // D
    d = offs - i * D  # equivalent to offs % D
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    t = tl.load(t_ptr + i, mask=mask, other=0.0)           # vector t of length N
    delta = tl.load(deltas_ptr + d, mask=mask, other=0.0)  # vector deltas of length D
    shift = 0.05
    mod = tl.exp(-t * delta) + shift
    out = v * mod
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def add_residual_kernel(
    v_ptr, residual_ptr, out_ptr,
    N, D,  # D is 256
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offs, mask=mask, other=0.0)
    out = v + res
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Extract dynamic dims from args (as provided by evaluator)
        N = int(args[0]["batch_size"])
        L = int(args[0]["seq_len"])
        D = 256  # fixed from original model (d_model)

        # Allocate output buffer (N, D) on CUDA and pass 1D views to Triton kernels
        out = torch.empty((N, D), device='cuda', dtype=torch.float32)
        out1d = out.view(-1)
        total = N * D
        BLOCK = 1024
        grid_total = (triton.cdiv(total, BLOCK),)

        # 1) create_2d_buffer_kernel: initialize out to zeros
        create_2d_buffer_kernel[grid_total](
            out1d,
            N, D,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 2) t = linspace(0, L-1, L)
        t = torch.empty((L,), device='cuda', dtype=torch.float32)
        grid_t = (triton.cdiv(L, BLOCK),)
        linspace_1d_kernel[grid_t](t, 0.0, (L - 1), L, BLOCK=BLOCK)

        # 3) gate vector (ones) of length total
        gate_1d = torch.empty((total,), device='cuda', dtype=torch.float32)
        grid_gate = (triton.cdiv(total, BLOCK),)
        ones_1d_kernel[grid_gate](gate_1d, total, BLOCK=BLOCK)

        # 4) gate_forward: out1d = out1d * gate_1d
        gate_forward_kernel[grid_total](
            out1d, gate_1d, out1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 5) deltas = linspace(0, D-1, D)
        deltas = torch.empty((D,), device='cuda', dtype=torch.float32)
        grid_d = (triton.cdiv(D, BLOCK),)
        linspace_1d_kernel[grid_d](deltas, 0.0, (D - 1), D, BLOCK=BLOCK)

        # 6) exp_mod_apply: out1d = out1d * (exp(-t[i] * deltas[d]) + 0.05)
        exp_mod_apply_kernel[grid_total](
            out1d, t, deltas, out1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 7) add_residual: out1d = out1d + out1d (self-add as residual)
        add_residual_kernel[grid_total](
            out1d, out1d, out1d,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # Return the final output tensor (shape: (N, D))
        return out


def run(*args):
    return ModelNew()(*args)
