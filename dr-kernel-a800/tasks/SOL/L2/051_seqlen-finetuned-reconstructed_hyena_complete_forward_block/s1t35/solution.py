import torch
import triton
import triton.language as tl


@triton.jit
def create_2d_buffer_kernel(
    out_ptr,
    total,
    BLOCK: tl.constexpr
):
    # Fill a 1D output buffer of length 'total' with zeros using Triton.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    tl.store(out_ptr + offs, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr,
    start, end, length,
    BLOCK: tl.constexpr
):
    # Generate a 1D vector of length 'length' from 'start' to 'end'.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    step = (end - start) / (length - 1) if length > 1 else 0.0
    values = start + offs * step
    tl.store(out_ptr + offs, values, mask=mask)


@triton.jit
def ones_1d_kernel(
    out_ptr,
    length,
    BLOCK: tl.constexpr
):
    # Fill a 1D output buffer of length 'length' with ones.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    tl.store(out_ptr + offs, 1.0, mask=mask)


@triton.jit
def gate_forward_kernel(
    out_ptr, gate_ptr, out_out_ptr,
    total,
    BLOCK: tl.constexpr
):
    # Elementwise out_out[i] = out[i] * gate[i]
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    out = tl.load(out_ptr + offs, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offs, mask=mask, other=1.0)
    out_out = out * gate
    tl.store(out_out_ptr + offs, out_out, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    out_ptr, t_ptr, deltas_ptr, out_out_ptr,
    N, D, shift,
    BLOCK: tl.constexpr
):
    # Elementwise: out_out[i] = out[i] * (exp(-t[i // D] * deltas[i % D]) + shift)
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    out = tl.load(out_ptr + offs, mask=mask, other=0.0)
    i = offs
    i_d = i // D
    i_mod = i % D
    t = tl.load(t_ptr + i_d, mask=mask, other=0.0)
    deltas = tl.load(deltas_ptr + i_mod, mask=mask, other=0.0)
    exp_term = tl.exp(-t * deltas)
    scale = exp_term + shift
    out_out = out * scale
    tl.store(out_out_ptr + offs, out_out, mask=mask)


@triton.jit
def add_residual_kernel(
    out_ptr, out_out_ptr,
    total,
    BLOCK: tl.constexpr
):
    # Elementwise: out_out[i] = out[i] + out[i] (self-add)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    out = tl.load(out_ptr + offs, mask=mask, other=0.0)
    out_out = out + out
    tl.store(out_out_ptr + offs, out_out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args are (batch_size, seq_len)
        batch_size = args[0]
        seq_len = args[1]
        N = int(batch_size)
        D = int(seq_len)

        # Output tensor (allocated via torch). Triton kernels will fill it.
        out = torch.empty((N, D), device='cuda', dtype=torch.float32)
        out1d = out.view(-1)
        total = N * D

        # 1) create_2d_buffer_kernel: initialize out with zeros
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        create_2d_buffer_kernel[grid](out1d, total, BLOCK=BLOCK)

        # 2) linspace_1d_kernel: t of length D = seq_len (0..D-1)
        t = torch.empty((D,), device='cuda', dtype=torch.float32)
        grid_t = (triton.cdiv(D, BLOCK),)
        linspace_1d_kernel[grid_t](t, 0.0, (D - 1), D, BLOCK=BLOCK)

        # 3) ones_1d_kernel: gate vector of length total (all ones)
        gate_1d = torch.empty((total,), device='cuda', dtype=torch.float32)
        grid_gate = (triton.cdiv(total, BLOCK),)
        ones_1d_kernel[grid_gate](gate_1d, total, BLOCK=BLOCK)

        # 4) gate_forward_kernel: out = out * gate (elementwise)
        grid_gate2 = grid  # covers all elements
        gate_forward_kernel[grid_gate2](out1d, gate_1d, out1d, total, BLOCK=BLOCK)

        # 5) exp_mod_apply_kernel: out = out * (exp(-t[i // D] * deltas[i % D]) + 0.05)
        deltas = torch.empty((D,), device='cuda', dtype=torch.float32)
        grid_d = (triton.cdiv(D, BLOCK),)
        linspace_1d_kernel[grid_d](deltas, 0.0, (D - 1), D, BLOCK=BLOCK)
        shift = 0.05
        exp_mod_apply_kernel[grid](out1d, t, deltas, out1d, N, D, shift, BLOCK=BLOCK)

        # 6) add_residual_kernel: out = out + out (self-add)
        add_residual_kernel[grid](out1d, out1d, total, BLOCK=BLOCK)

        # Return the output tensor
        return out


def run(*args):
    return ModelNew()(*args)
