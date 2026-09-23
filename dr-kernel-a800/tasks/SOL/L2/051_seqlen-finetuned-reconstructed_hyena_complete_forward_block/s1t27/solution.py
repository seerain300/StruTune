import torch
import triton
import triton.language as tl


@triton.jit
def create_2d_buffer_kernel(
    out_ptr,
    N, D,
    stride0, stride1,
    BLOCK: tl.constexpr
):
    # Initialize a linearized 2D buffer of size N*D with zeros
    pid = tl.program_id(0)
    total = N * D
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    i = offs // D
    d = offs % D
    ptrs = out_ptr + i * stride0 + d * stride1
    tl.store(ptrs, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr,
    start, end, length,
    BLOCK: tl.constexpr
):
    # Fill a 1D tensor with linspace from start to end, length = 'length'
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < length
    step = (end - start) / (length - 1 + 0.0)  # safe for length >= 1; for length==1, step=end-start
    val = start + idx * step
    tl.store(out_ptr + idx, val, mask=mask)


@triton.jit
def ones_1d_kernel(
    out_ptr,
    length,
    BLOCK: tl.constexpr
):
    # Fill a 1D tensor with ones
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < length
    tl.store(out_ptr + idx, 1.0, mask=mask)


@triton.jit
def gate_forward_kernel(
    out_ptr, gate_ptr,
    N, D,
    stride_out0, stride_out1,
    BLOCK: tl.constexpr
):
    # Elementwise multiply: out[i, d] = out[i, d] * gate[d]
    pid = tl.program_id(0)
    total = N * D
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    i = offs // D
    d = offs % D
    out_ptrs = out_ptr + i * stride_out0 + d * stride_out1
    gate_ptrs = gate_ptr + d  # gate is 1D along d
    out_vals = tl.load(out_ptrs, mask=mask, other=0.0)
    gate_vals = tl.load(gate_ptrs, mask=mask, other=1.0)
    tl.store(out_ptrs, out_vals * gate_vals, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    out_ptr, t_ptr, deltas_ptr,
    N, D,
    stride_out0, stride_out1,
    BLOCK: tl.constexpr,
    shift: tl.float32
):
    # out[i, d] = out[i, d] * (exp(-t[i] * deltas[d] + shift))
    pid = tl.program_id(0)
    total = N * D
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    i = offs // D
    d = offs % D
    out_ptrs = out_ptr + i * stride_out0 + d * stride_out1
    t_ptrs = t_ptr + i  # t is 1D along i
    d_ptrs = deltas_ptr + d  # deltas is 1D along d
    out_vals = tl.load(out_ptrs, mask=mask, other=0.0)
    t_vals = tl.load(t_ptrs, mask=mask, other=0.0)
    d_vals = tl.load(d_ptrs, mask=mask, other=0.0)
    factor = -t_vals * d_vals + shift
    new_vals = out_vals * tl.exp(factor)
    tl.store(out_ptrs, new_vals, mask=mask)


@triton.jit
def add_residual_kernel(
    out_ptr,
    N, D,
    stride_out0, stride_out1,
    BLOCK: tl.constexpr
):
    # out = out + out (self-add)
    pid = tl.program_id(0)
    total = N * D
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    i = offs // D
    d = offs % D
    out_ptrs = out_ptr + i * stride_out0 + d * stride_out1
    out_vals = tl.load(out_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, out_vals + out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, batch_size: int, seq_len: int):
        # Only forward signature is provided: (batch_size, seq_len).
        # No torch ops beyond allocating the output tensor are allowed.
        # We allocate the output tensor and pass its data pointer to Triton kernels.
        out = torch.empty((batch_size, seq_len), device='cuda', dtype=torch.float32)

        # 1) create_2d_buffer_kernel: initialize out to zeros via Triton
        BLOCK = 1024
        grid = (triton.cdiv(batch_size * seq_len, BLOCK),)
        create_2d_buffer_kernel[grid](
            out.data_ptr(),
            batch_size, seq_len,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 2) linspace_1d_kernel: t vector of length seq_len
        t = torch.empty((seq_len,), device='cuda', dtype=torch.float32)
        grid_t = (triton.cdiv(seq_len, BLOCK),)
        linspace_1d_kernel[grid_t](
            t,
            0.0, (seq_len - 1), seq_len,
            BLOCK=BLOCK
        )

        # 3) ones_1d_kernel: gate_1d of length seq_len (all ones)
        gate_1d = torch.empty((seq_len,), device='cuda', dtype=torch.float32)
        grid_gate = (triton.cdiv(seq_len, BLOCK),)
        ones_1d_kernel[grid_gate](
            gate_1d,
            seq_len,
            BLOCK=BLOCK
        )

        # 4) gate_forward_kernel: out = out * gate_1d (elementwise)
        grid_gate2 = grid
        gate_forward_kernel[grid_gate2](
            out.data_ptr(), gate_1d.data_ptr(),
            batch_size, seq_len,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 5) exp_mod_apply_kernel: apply exp modulation
        grid5 = grid
        exp_mod_apply_kernel[grid5](
            out.data_ptr(), t.data_ptr(), t.data_ptr(),  # use t as 'deltas' for compatibility
            batch_size, seq_len,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK,
            shift=0.05
        )

        # 6) add_residual_kernel: out = out + out
        add_residual_kernel[grid](
            out.data_ptr(),
            batch_size, seq_len,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        return out


def run(*args):
    return ModelNew()(*args)
