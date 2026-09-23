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
    # Each program handles BLOCK consecutive elements of a linearized (N, D) buffer.
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = idx < total
    # Map linear index to (i, d): i = idx // D, d = idx % D
    i = idx // D
    d = idx - i * D  # equivalent to idx % D
    out_row_ptr = out_ptr + i * out_stride0 + d * out_stride1
    tl.store(out_row_ptr, 0.0, mask=mask)


@triton.jit
def linspace_1d_kernel(
    out_ptr, start, end, length,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < length
    step = (end - start) / (length - 1)
    vals = start + idx * step
    tl.store(out_ptr + idx, vals, mask=mask)


@triton.jit
def ones_1d_kernel(
    out_ptr, length,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < length
    tl.store(out_ptr + idx, 1.0, mask=mask)


@triton.jit
def gate_forward_kernel(
    v_in_ptr, gate_ptr, out_ptr,
    N, D,
    v_in_stride0, v_in_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # Elementwise: out[i, d] = v_in[i, d] * gate[i, d]
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = idx < total
    i = idx // D
    d = idx - i * D
    v_row_ptr = v_in_ptr + i * v_in_stride0 + d * v_in_stride1
    g_row_ptr = gate_ptr + i * gate_ptr.stride(0) + d * gate_ptr.stride(1)
    out_row_ptr = out_ptr + i * out_stride0 + d * out_stride1
    v = tl.load(v_row_ptr, mask=mask, other=0.0)
    g = tl.load(g_row_ptr, mask=mask, other=1.0)
    tl.store(out_row_ptr, v * g, mask=mask)


@triton.jit
def exp_mod_apply_kernel(
    v_ptr, t_ptr, deltas_ptr, out_ptr,
    N, D,
    v_stride0, v_stride1,
    out_stride0, out_stride1,
    shift,
    BLOCK: tl.constexpr
):
    # Elementwise over flattened (N, D): out[i*D + d] = v[i*D + d] * (exp(-t[i] * deltas[d]) + shift)
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = idx < total
    i = idx // D
    d = idx - i * D
    v_row_ptr = v_ptr + i * v_stride0 + d * v_stride1
    out_row_ptr = out_ptr + i * out_stride0 + d * out_stride1
    t = tl.load(t_ptr + i, mask=mask, other=0.0)
    deltas = tl.load(deltas_ptr + d, mask=mask, other=0.0)
    v = tl.load(v_row_ptr, mask=mask, other=0.0)
    mod = tl.exp(-t * deltas) + shift
    tl.store(out_row_ptr, v * mod, mask=mask)


@triton.jit
def add_residual_kernel(
    v_in_ptr, out_ptr,
    N, D,
    v_in_stride0, v_in_stride1,
    out_stride0, out_stride1,
    BLOCK: tl.constexpr
):
    # Elementwise: out[i, d] = out[i, d] + v_in[i, d]
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    mask = idx < total
    i = idx // D
    d = idx - i * D
    out_row_ptr = out_ptr + i * out_stride0 + d * out_stride1
    v_row_ptr = v_in_ptr + i * v_in_stride0 + d * v_in_stride1
    out_val = tl.load(out_row_ptr, mask=mask, other=0.0)
    v_val = tl.load(v_row_ptr, mask=mask, other=0.0)
    tl.store(out_row_ptr, out_val + v_val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args[0] is axes dict: {'batch_size': B, 'seq_len': L}
        axes = args[0]
        B = int(axes['batch_size'])
        L = int(axes['seq_len'])
        D = 256  # as per original code
        device = torch.device('cuda')  # Triton runs on CUDA
        total = B * D

        # 1) Create output buffer via Triton (no torch ops in forward)
        out = torch.empty((B, D), device=device, dtype=torch.float32)
        out_1d = out.view(-1)
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        create_2d_buffer_kernel[grid](
            out_1d,
            B, D,
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 2) Prepare t: linspace(0, L-1, L)
        t = torch.empty((L,), device=device, dtype=torch.float32)
        grid_t = (triton.cdiv(L, BLOCK),)
        linspace_1d_kernel[grid_t](t, 0.0, float(L - 1), L, BLOCK=BLOCK)

        # 3) Prepare deltas: 0..D-1
        deltas = torch.empty((D,), device=device, dtype=torch.float32)
        grid_d = (triton.cdiv(D, BLOCK),)
        linspace_1d_kernel[grid_d](deltas, 0.0, float(D - 1), D, BLOCK=BLOCK)

        # 4) gate_forward: out = out * 1 (gate is ones vector of length total)
        gate = torch.empty((total,), device=device, dtype=torch.float32)
        grid_gate = (triton.cdiv(total, BLOCK),)
        ones_1d_kernel[grid_gate](gate, total, BLOCK=BLOCK)
        gate_1d = gate  # flattened ones
        gate_forward_kernel[grid](
            out_1d, gate_1d, out_1d,
            B, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        # 5) exp_mod_apply: out = out * (exp(-t[i] * deltas[d]) + 0.05)
        shift = 0.05
        exp_mod_apply_kernel[grid](
            out_1d, t, deltas, out_1d,
            B, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            shift,
            BLOCK=BLOCK
        )

        # 6) add_residual: out = out + out (self-add)
        add_residual_kernel[grid](
            out_1d, out_1d,
            B, D,
            out.stride(0), out.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK
        )

        return out


def run(*args):
    return ModelNew()(*args)
