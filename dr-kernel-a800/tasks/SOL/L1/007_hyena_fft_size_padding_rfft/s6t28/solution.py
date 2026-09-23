import torch
import triton
import triton.language as tl


@triton.jit
def pad_zeros_kernel(
    x_ptr,        # *float32, input x of shape (B, C, S)
    z_ptr,        # *float32, output z of shape (B, C, N), where N=2*S
    B: tl.int32,
    C: tl.int32,
    S: tl.int32,  # original seqlen
    N: tl.int32,  # padded length = 2*S
):
    bc = tl.program_id(0)
    # Each program handles one (b, c) plane
    # Base offset for (b, c) in z is bc * N
    base = bc * N
    # Copy x[:, :, :S] into z[:, :, 0:S]
    # Loop over s = 0..S-1
    s = 0
    while s < S:
        x_val = tl.load(x_ptr + bc * S + s)
        # Store to z at position s
        tl.store(z_ptr + base + s, x_val)
        s += 1
    # Zero-pad the rest z[:, :, S:N]
    # No need to load; just write zeros
    s = S
    while s < N:
        tl.store(z_ptr + base + s, 0.0)
        s += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x shape: (B, C, S), dtype float32
        assert x.dim() == 3, "Input must be a 3D tensor (B, C, S)"
        B, C, S = x.shape
        N = 2 * S

        # Prepare padded input z of shape (B, C, N), float32
        # We can use the original dtype; cast if needed
        z = torch.empty((B, C, N), dtype=torch.float32, device=x.device)

        # Launch Triton kernel to copy x into z[0:S] and zero-pad z[S:N]
        grid = (B * C,)
        pad_zeros_kernel[grid](x, z, B, C, S, N, num_warps=1)

        # Compute rfft on padded real input z along last dim, length N
        # Output y is complex of length N//2 + 1 = S + 1
        y = torch.fft.rfft(z, n=N, dim=-1)

        # Normalize by 2*S as per original code
        y = y / (2 * S)

        # Extract real and imaginary parts
        out_real = y.real.contiguous()  # shape (B, C, S+1)
        out_imag = y.imag.contiguous()  # shape (B, C, S+1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
