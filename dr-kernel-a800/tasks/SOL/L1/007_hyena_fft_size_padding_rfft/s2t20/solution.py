import torch
import triton
import triton.language as tl


@triton.jit
def pad_kernel(
    x_ptr,          # *float32, input x of shape (B, C, L), contiguous
    out_ptr,        # *float32, output buffer of shape (B*C, two_L), contiguous
    L: tl.constexpr,            # seqlen
    two_L: tl.constexpr,        # 2 * L
    BC: tl.constexpr,           # B * C
    b: tl.constexpr,            # batch index
    c: tl.constexpr,            # channel index
):
    # Compute the linear index for the (b, c) slice in out buffer
    slice_index = b * C + c
    # Base pointer for this (b, c) slice in out
    base = slice_index * two_L

    # Load x[b, c, :] and store into the first L positions of out[b, c, :]
    # x is contiguous (B, C, L), so we can linearly index: idx = ((b*C + c) * L) + t
    # But we don't have b, c, L as runtime; we pass BC and L. The caller ensures x is (B, C, L) contiguous.
    # Here, we rely on x being laid out as (B, C, L) contiguous, so idx = ((b*C + c) * L) + t is correct if we pass x_ptr accordingly.
    # To avoid complexity, we use the fact that x_ptr points to (B*C, L) contiguous for each (b,c) as a row. The caller ensures x is contiguous.
    # We'll compute the base index for x slice: ((b*C + c) * L)
    x_row_base = (b * C + c) * L
    # Copy x[b, c, :] into out[b, c, :L]
    for t in range(0, L):
        x_val = tl.load(x_ptr + x_row_base + t)
        tl.store(out_ptr + base + t, x_val)


@triton.jit
def real_dft_scalar_kernel(
    out_real_ptr,   # *float32, output real part buffer of shape (BC, L+1), contiguous
    out_imag_ptr,   # *float32, output imag part buffer of shape (BC, L+1), contiguous
    in_ptr,         # *float32, input padded buffer of shape (BC, two_L), contiguous
    L: tl.constexpr,            # seqlen
    two_L: tl.constexpr,        # 2 * L
    BC: tl.constexpr,           # B * C
    b: tl.constexpr,            # batch index
    c: tl.constexpr,            # channel index
    k: tl.constexpr,            # frequency index
):
    # This kernel computes the real part of the k-th bin for (b, c).
    # Imaginary part for real inputs is zero.
    slice_index = b * C + c
    base_in = slice_index * two_L
    base_out = slice_index * (L + 1)

    # Accumulator for the real part
    acc = 0.0
    # Compute sum over t=0..two_L-1 of in[t] * cos(2*pi*k*t / two_L)
    for t in range(0, two_L):
        x_t = tl.load(in_ptr + base_in + t)
        angle = 2.0 * 3.141592653589793 * float(k) * float(t) / float(two_L)
        cos_val = tl.cos(angle)
        acc += x_t * cos_val

    # Normalize by 2*L
    acc = acc / float(two_L)

    # Store real and imag parts
    tl.store(out_real_ptr + base_out + k, acc)
    tl.store(out_imag_ptr + base_out + k, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L), float32
        assert x.dtype == torch.float32, "Input must be float32"
        x = x.contiguous()
        B, C, L = x.shape
        two_L = 2 * L
        BC = B * C

        # Prepare padded input: (BC, two_L)
        # We'll build zeros vector and copy first L elements from x using Triton.
        # Allocate out buffer for padded inputs
        in_buf = torch.zeros((BC, two_L), dtype=torch.float32, device=x.device)

        # Launch pad_kernel to copy x[b, c, :] into in_buf[b*C + c, :L]
        grid = (BC,)
        pad_kernel[grid](
            x, in_buf,
            L=L, two_L=two_L, BC=BC,
            b=0, c=0,  # dummy; Triton treats them as constexpr; grid handles (b,c)
        )
        # Note: The pad_kernel grid should iterate over (b,c). Triton requires a single program per (b,c),
        # but we can't pass b/c into a single grid. Instead, we launch with grid=(BC,) and compute b,c via pid.
        # However Triton doesn't allow passing b,c here. To fix, we launch one program per (b,c) slice by looping in Python.
        # So we'll replace the above call with a loop:

        # Since Triton requires per-(b,c) programs, we launch one program per slice. Triton doesn't expose 'pid' directly.
        # Workaround: launch grid=(BC,) and pass b,c as constexpr from Python. But Triton's kernel signature requires b,c as tl.constexpr.
        # The simplest approach: run BC separate launches. Triton supports grid as a tuple; each program can read b,c from runtime.
        # To avoid complexity, we implement a loop in Python around Triton launch. Triton requires @triton.jit to be launched; we can't loop inside Triton here.
        # Therefore, we manually launch pad for each (b,c): use grid=(BC,) and let Triton handle per-slice via b,c passed as constexpr by Python.

        # Better approach: use a Python loop to call Triton per (b,c). Triton kernel signature should accept b,c as tl.constexpr.
        # We redefine pad_kernel to accept b,c as tl.constexpr and launch per (b,c).
        # Let's redefine pad_kernel accordingly:

        # Redefine pad_kernel with b,c in signature:
        @triton.jit
        def pad_kernel_bc(
            x_ptr, out_ptr,
            L: tl.constexpr, two_L: tl.constexpr, BC: tl.constexpr,
            b: tl.constexpr, c: tl.constexpr,
        ):
            slice_index = b * C + c
            base = slice_index * two_L
            x_row_base = (b * C + c) * L
            for t in range(0, L):
                x_val = tl.load(x_ptr + x_row_base + t)
                tl.store(out_ptr + base + t, x_val)

        # Launch per (b,c):
        in_buf = torch.zeros((BC, two_L), dtype=torch.float32, device=x.device)
        for b in range(B):
            for c in range(C):
                pad_kernel_bc[(1,)](
                    x, in_buf,
                    L=L, two_L=two_L, BC=BC,
                    b=b, c=c,
                )

        # Similarly, redefine real_dft_scalar_kernel with b,c,k in signature:
        @triton.jit
        def real_dft_scalar_kernel_bc(
            out_real_ptr, out_imag_ptr, in_ptr,
            L: tl.constexpr, two_L: tl.constexpr, BC: tl.constexpr,
            b: tl.constexpr, c: tl.constexpr, k: tl.constexpr,
        ):
            slice_index = b * C + c
            base_in = slice_index * two_L
            base_out = slice_index * (L + 1)

            acc = 0.0
            for t in range(0, two_L):
                x_t = tl.load(in_ptr + base_in + t)
                angle = 2.0 * 3.141592653589793 * float(k) * float(t) / float(two_L)
                cos_val = tl.cos(angle)
                acc += x_t * cos_val

            acc = acc / float(two_L)

            tl.store(out_real_ptr + base_out + k, acc)
            tl.store(out_imag_ptr + base_out + k, 0.0)

        # Allocate outputs: (BC, L+1) real and imag
        out_real = torch.empty((BC, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((BC, L + 1), dtype=torch.float32, device=x.device)

        # Launch per (b,c) for k in [0..L-1]:
        for b in range(B):
            for c in range(C):
                for k in range(L):
                    real_dft_scalar_kernel_bc[(1,)](
                        out_real, out_imag, in_buf,
                        L=L, two_L=two_L, BC=BC,
                        b=b, c=c, k=k,
                    )

        # Reshape to (B, C, L+1)
        out_real = out_real.view(B, C, L + 1)
        out_imag = out_imag.view(B, C, L + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
