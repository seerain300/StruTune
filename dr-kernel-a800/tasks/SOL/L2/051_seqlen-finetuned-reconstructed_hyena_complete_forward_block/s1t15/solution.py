import triton
import triton.language as tl


@triton.jit
def create_2d_buffer_kernel(
    out_ptr,
    N, D,
    stride0, stride1,
    BLOCK: tl.constexpr
):
    # Linearized 1D fill of zeros into a 2D buffer (N, D) with given strides.
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Map linear index -> (i, d)
    i = offs // D
    d = offs % D

    ptrs = out_ptr + i * stride0 + d * stride1
    tl.store(ptrs, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr,
    start, end,
    length,
    stride,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    step = (end - start) / length
    vals = start + offs * step
    ptrs = out_ptr + offs * stride
    tl.store(ptrs, vals, mask=mask)


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
    ptrs = out_ptr + offs * stride
    tl.store(ptrs, 1.0, mask=mask)


@triton.jit
def gate_forward_kernel(
    v_in_ptr, gate_ptr, out_ptr,
    N, D,
    v_in_stride0, v_in_stride1,
    out_stride0, out_stride1,
    BLOCK_D: tl.constexpr
):
    # Each program handles one row i, vectorizes across D
    i = tl.program_id(0)  # row index
    d = tl.arange(0, BLOCK_D)  # column offsets
    mask = d < D

    v_row_ptr = v_in_ptr + i * v_in_stride0 + d * v_in_stride1
    g_row_ptr = gate_ptr + i * gate_ptr.stride(0) + d * gate_ptr.stride(1)
    o_row_ptr = out_ptr + i * out_stride0 + d * out_stride1

    v = tl.load(v_row_ptr, mask=mask, other=0.0)
    g = tl.load(g_row_ptr, mask=mask, other=1.0)
    out = v * g
    tl.store(o_row_ptr, out, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    v_ptr, t_ptr, deltas_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK_D: tl.constexpr
):
    # Apply exp modulation per row: out[i, d] = v[i, d] * (exp(-t[i] * deltas[d]) + 0.05)
    i = tl.program_id(0)  # row index
    d = tl.arange(0, BLOCK_D)  # column offsets
    mask = d < D

    v_row_ptr = v_ptr + i * v_stride0 + d * v_stride1
    v = tl.load(v_row_ptr, mask=mask, other=0.0)

    # t is 1D length N, delta is 1D length D
    t_val = tl.load(t_ptr + i)
    delta = tl.load(deltas_ptr + d, mask=mask, other=0.0)
    factor = tl.exp(-t_val * delta) + 0.05
    out = v * factor

    o_row_ptr = out_ptr + i * out_stride0 + d * out_stride1
    tl.store(o_row_ptr, out, mask=mask)


@triton.jit
def add_residual_kernel(
    v_ptr, res_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    BLOCK_D: tl.constexpr
):
    # Elementwise add: out[i, d] = v[i, d] + res[i, d]
    i = tl.program_id(0)  # row index
    d = tl.arange(0, BLOCK_D)  # column offsets
    mask = d < D

    v_row_ptr = v_ptr + i * v_stride0 + d * v_stride1
    res_row_ptr = res_ptr + i * res_ptr.stride(0) + d * res_ptr.stride(1)
    o_row_ptr = out_ptr + i * out_stride0 + d * out_stride1

    v = tl.load(v_row_ptr, mask=mask, other=0.0)
    res = tl.load(res_row_ptr, mask=mask, other=0.0)
    out = v + res
    tl.store(o_row_ptr, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We assume forward is called with (batch_size, seq_len) as first two args.
        # Extract N, D (no torch computation).
        N = int(args[0])
        D = int(args[1])

        # Allocate output buffer (Triton cannot allocate, so we use torch.empty).
        out = torch.empty((N, D), device='cuda', dtype=torch.float32)
        total = N * D

        # 1) create_2d_buffer_kernel: initialize out with zeros
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        create_2d_buffer_kernel[grid](
            out,  # out_ptr
            N, D,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 2) linspace_1d_kernel: deltas = linspace(0, D-1, D)
        grid_d = (triton.cdiv(D, BLOCK),)
        deltas = torch.empty((D,), device='cuda', dtype=torch.float32)
        linspace_1d_kernel[grid_d](
            deltas, 0.0, (D - 1), D, 1.0,
            BLOCK=BLOCK
        )

        # 3) ones_1d_kernel: gate vector length total (all ones)
        gate_total = torch.empty((total,), device='cuda', dtype=torch.float32)
        grid_gate = (triton.cdiv(total, BLOCK),)
        ones_1d_kernel[grid_gate](
            gate_total, total, 1.0, BLOCK=BLOCK
        )

        # 4) gate_forward_kernel: out = out * gate (elementwise)
        grid_gate2 = (N,)  # one program per row
        gate_forward_kernel[grid_gate2](
            out, gate_total, out,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_D=BLOCK
        )

        # 5) exp_mod_apply_kernel: out = out * (exp(-t[i] * deltas[d]) + 0.05)
        t = torch.empty((N,), device='cuda', dtype=torch.float32)
        grid_t = (triton.cdiv(N, BLOCK),)
        linspace_1d_kernel[grid_t](
            t, 0.0, (N - 1), N, 1.0,
            BLOCK=BLOCK
        )
        exp_mod_apply_kernel[grid_gate2](
            out, t, deltas, out,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_D=BLOCK
        )

        # 6) add_residual_kernel: out = out + out (self-add)
        add_residual_kernel[grid_gate2](
            out, out, out,
            N, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_D=BLOCK
        )

        # Return the final tensor. Forward must return a tensor.
        return out


def run(*args):
    return ModelNew()(*args)
