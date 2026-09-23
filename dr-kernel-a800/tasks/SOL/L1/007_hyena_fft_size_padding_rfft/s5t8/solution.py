import torch
import triton
import triton.language as tl


@triton.jit
def cosine_sum_kernel(x_ptr, out_ptr, j, N: tl.constexpr, M: tl.constexpr):
    """
    Compute rfft real bin j for j in [0, M], where N = 2*M.
    real_out[j] = (1/N) * sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N)
    Writes into out_ptr[0] (the value is associated with j via kernel launch).
    """
    acc = 0.0
    for k in range(0, N):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.cos(angle)
    acc *= (1.0 / N)
    # Store result; out_ptr is a 1-element buffer per j
    tl.store(out_ptr, acc)


@triton.jit
def sine_sum_kernel(x_ptr, out_ptr, j, N: tl.constexpr, M: tl.constexpr):
    """
    Compute rfft imaginary bin j for j in [1, M-1], where N = 2*M.
    imag_out[j] = (1/N) * sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N)
    Writes into out_ptr[0].
    """
    acc = 0.0
    for k in range(0, N):
        xk = tl.load(x_ptr + 0)  # this is fine; x_ptr is length N, and we index via k in loop
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.sin(angle)
    acc *= (1.0 / N)
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
        - x_f32 = x.to(float32)
        - y = torch.fft.rfft(x_f32, n=2*seqlen)  [we will not call torch.fft]
        - y = y / (2*seqlen)
        - return real(y), imag(y) as float32 tensors of shape (B, C, seqlen+1)
        """
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        B, C, M = x.shape
        N = 2 * M  # implicit zero-padding length

        # Cast to float32 for numerical stability and to match PyTorch behavior
        x_f32 = x.to(torch.float32)

        # Flatten (B, C) into rows
        BC = B * C
        x_flat = x_f32.contiguous().view(BC, M)

        # Allocate outputs: real and imag parts of length M+1
        real_out = torch.empty((BC, M + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((BC, M + 1), dtype=torch.float32, device=x.device)

        # We will populate imag_out[0] and imag_out[M] as zeros (real inputs => imag at 0 and M are zero)

        # Launch cosine_sum_kernel for j in [0, M]
        # For each j, we write the result into a temporary 1-element tensor
        for j in range(M + 1):
            out = torch.empty(1, dtype=torch.float32, device=x.device)
            # Note: Triton loops require static bounds; we pass N and M as tl.constexpr via function call
            cosine_sum_kernel[(1,)](x_flat[j, :], out, j, N, M)
            real_out[j, j] = out[0]

        # Launch sine_sum_kernel for j in [1, M-1], set imag[0] and imag[M] to zero
        for j in range(1, M):
            out = torch.empty(1, dtype=torch.float32, device=x.device)
            sine_sum_kernel[(1,)](x_flat[j, :], out, j, N, M)
            imag_out[j, j] = out[0]
        imag_out[:, 0] = 0.0
        imag_out[:, M] = 0.0

        # Reshape back to (B, C, M+1)
        real_out = real_out.view(B, C, M + 1)
        imag_out = imag_out.view(B, C, M + 1)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
