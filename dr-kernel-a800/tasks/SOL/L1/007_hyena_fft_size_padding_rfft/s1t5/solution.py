import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute real part of rfft for each (b, c) slice and store in out_real
if TRITON_AVAILABLE:
    @triton.jit
    def _real_to_re_kernel(x_ptr, out_ptr,
                            B, C, N, L,
                            stride_b, stride_c, stride_t,
                            out_stride_b, out_stride_c, out_stride_m,
                            BLOCK_T: tl.constexpr):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)

        # Accumulator for sum over t
        for j in range(0, L + 1):
            s = tl.zeros((), dtype=tl.float32)

            # Loop over time indices in chunks of BLOCK_T
            for t_start in range(0, N, BLOCK_T):
                t = t_start + tl.arange(0, BLOCK_T)
                mask = t < N

                # Load x[b, c, t] as float32
                x_vals = tl.load(x_ptr + pid_b * stride_b + pid_c * stride_c + t * stride_t, mask=mask, other=0.0)

                # cos(2*pi*j*t/N)
                cos_term = tl.cos(2.0 * tl.pi * j * t / N)

                # Accumulate sum over chunk
                s += tl.sum(x_vals * cos_term, axis=0)

            # Normalize by N and store to out_real[b, c, j]
            out_index = pid_b * out_stride_b + pid_c * out_stride_c + j * out_stride_m
            tl.store(out_ptr + out_index, s * (1.0 / N))


    # Triton kernel: compute imaginary part of rfft for each (b, c) slice and store in out_imag
    @triton.jit
    def _real_to_im_kernel(x_ptr, out_ptr,
                            B, C, N, L,
                            stride_b, stride_c, stride_t,
                            out_stride_b, out_stride_c, out_stride_m,
                            BLOCK_T: tl.constexpr):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)

        for j in range(0, L + 1):
            s = tl.zeros((), dtype=tl.float32)

            for t_start in range(0, N, BLOCK_T):
                t = t_start + tl.arange(0, BLOCK_T)
                mask = t < N

                x_vals = tl.load(x_ptr + pid_b * stride_b + pid_c * stride_c + t * stride_t, mask=mask, other=0.0)

                sin_term = tl.sin(2.0 * tl.pi * j * t / N)
                s += tl.sum(x_vals * sin_term, axis=0)

            out_index = pid_b * out_stride_b + pid_c * out_stride_c + j * out_stride_m
            tl.store(out_ptr + out_index, s * (1.0 / N))


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor x of shape (batch, channels, seqlen)
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single input tensor.")
        x = args[0]

        # Ensure float32 input (original code casts to float32 before FFT)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # If Triton not available, fallback to PyTorch for robustness
        if not TRITON_AVAILABLE:
            batch, channels, seqlen = x.shape
            N = 2 * seqlen
            x_freq = torch.fft.rfft(x, n=N, dim=-1)
            x_freq = x_freq / N
            return x_freq.real.contiguous(), x_freq.imag.contiguous()

        # Triton path: compute real and imaginary parts entirely in Triton
        B, C, L = x.shape
        N = 2 * L

        # Ensure input is contiguous along last dim
        x = x.contiguous()

        # Prepare output tensors of shape (B, C, L+1)
        out_real = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)

        # Get strides (in elements)
        stride_b, stride_c, stride_t = x.stride()
        out_stride_b, out_stride_c, out_stride_m = out_real.stride()

        # Launch one program per (b, c) slice
        grid = (B, C)

        # Choose a reasonable BLOCK_T; 128 works well across sizes
        BLOCK_T = 128

        _real_to_re_kernel[grid](
            x, out_real,
            B, C, N, L,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            BLOCK_T=BLOCK_T,
        )

        _real_to_im_kernel[grid](
            x, out_imag,
            B, C, N, L,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            BLOCK_T=BLOCK_T,
        )

        # Kernels already normalized by 1/N; return results
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
