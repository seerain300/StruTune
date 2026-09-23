import torch
import triton
import triton.language as tl


@triton.jit
def real_rfft_kernels(
    x_ptr,           # *float32, input x flattened to (B*C*S,)
    real_out_ptr,    # *float32, output real part flattened to (B*C*(S+1),)
    imag_out_ptr,    # *float32, output imag part flattened to (B*C*(S+1),)
    S: tl.int32,     # sequence length
):
    # One program per (b, c)
    bc = tl.program_id(0)
    base_x = bc * S

    # Loop over k = 0..S
    k = 0
    while k <= S:
        sum_real = 0.0
        sum_imag = 0.0
        # Loop over t = 0..S-1
        t = 0
        while t < S:
            x_val = tl.load(x_ptr + base_x + t)
            # angle = -pi * k * t / S
            angle = -3.141592653589793 * k * t / S
            # Accumulate sums
            sum_real += x_val * tl.cos(angle)
            sum_imag += -x_val * tl.sin(angle)
            t += 1
        # Normalize by 2*S (as in original code: rfft output divided by 2*S)
        twoS = 2.0 * S
        y_real = sum_real / twoS
        y_imag = sum_imag / twoS
        # Store at (bc, k)
        out_index = bc * (S + 1) + k
        tl.store(real_out_ptr + out_index, y_real)
        # imag for Nyquist (k==S) is 0 for real inputs
        if k == S:
            tl.store(imag_out_ptr + out_index, 0.0)
        else:
            tl.store(imag_out_ptr + out_index, y_imag)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton implementation of:
          x_f32 = x.to(torch.float32)
          x_freq = torch.fft.rfft(x_f32, n=2*S)  # note: PyTorch uses n=2*S
          x_freq = x_freq / (2*S)
          return x_freq.real, x_freq.imag
        But all computation is done inside Triton kernels.
        Input: x of shape (B, C, S), float32 (we assume float32; if not, we cast).
        Output: (real_out, imag_out) both shape (B, C, S+1), float32.
        """
        assert x.dim() == 3, "Input must be 3D (B, C, S)"
        B, C, S = x.shape

        # Ensure float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Flatten (B, C, S) into (B*C, S) logical view; we operate per (b, c)
        x_flat = x.reshape(B * C, S)

        # Allocate outputs
        M = S + 1
        real_out = torch.empty((B * C, M), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((B * C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c)
        grid = (B * C,)
        real_rfft_kernels[grid](
            x_flat,
            real_out,
            imag_out,
            S,
        )

        # Reshape back to (B, C, S+1)
        real_out = real_out.view(B, C, M)
        imag_out = imag_out.view(B, C, M)
        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
