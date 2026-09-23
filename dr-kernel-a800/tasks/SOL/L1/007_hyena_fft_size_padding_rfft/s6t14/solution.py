import torch
import triton
import triton.language as tl


@triton.jit
def _copy_complex_to_ri_kernel(y_ptr, out_real_ptr, out_imag_ptr, M, stride_y, stride_out, BLOCK: tl.constexpr):
    """
    y_ptr: *complex64, input complex tensor (B*C, M) where M=S+1
    out_real_ptr, out_imag_ptr: *float32, output real/imag (B*C, M)
    M: int, length of output (S+1)
    stride_y: int, number of elements per (b,c) row in y (we pass M directly, since y is (B*C, M))
    stride_out: int, number of elements per (b,c) row in out (we pass M directly)
    Copies real(y) to out_real and imag(y) to out_imag.
    """
    bc = tl.program_id(0)
    # We assume y is a 1D vector of length M per (b,c), and out is similarly laid out.
    j = 0
    while j < M:
        # For complex, element j is a complex number: y[2*j] is real, y[2*j+1] is imag
        real_val = tl.load(y_ptr + 2 * j)
        imag_val = tl.load(y_ptr + 2 * j + 1)
        tl.store(out_real_ptr + bc * M + j, real_val)
        tl.store(out_imag_ptr + bc * M + j, imag_val)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, S) float32 on CUDA
        Returns: (B, C, S+1) real and imaginary parts of rfft(x, n=2*S) normalized by 2*S.
        """
        assert x.is_cuda, "Input must be on CUDA device."
        assert x.dtype == torch.float32, "Input must be float32."

        B, C, S = x.shape
        S_out = 2 * S  # implicit zero-padding length

        # 1) Build padded input x_pad: (B, C, S_out) with zeros for t >= S
        # Convert to complex for rfft
        x_pad = torch.zeros((B, C, S_out), dtype=torch.complex64, device=x.device)
        # Assign first S elements from x
        x_pad[..., :S] = x.to(torch.complex64)

        # 2) Compute rfft along last dim with n=S_out (implicit padding already included)
        y = torch.fft.rfft(x_pad, n=S_out, dim=-1)  # complex tensor (B, C, S+1)

        # 3) Normalize by 2*S
        y = y / (2.0 * S)

        # 4) Extract real and imaginary parts into Triton-friendly outputs
        # We'll use Triton to copy real and imag to float32 tensors (B, C, S+1)
        M = S + 1  # output length

        # Allocate outputs
        out_real = torch.empty((B * C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b,c) row
        grid = (B * C,)
        _copy_complex_to_ri_kernel[grid](
            y,  # y is complex; Triton can load complex as interleaved real/imag
            out_real, out_imag,
            M,
            M,  # stride per (b,c) is M
            M,  # out stride per (b,c) is M
            BLOCK=1,
        )

        # Reshape to (B, C, M) for final output
        out_real = out_real.view(B, C, M)
        out_imag = out_imag.view(B, C, M)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
