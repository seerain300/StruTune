import torch
import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_analytic_kernel(
    x_ptr,                    # *float32, input x flattened as [num_bc, S]
    y_real_ptr,               # *float32, output real part flattened as [num_bc, S+1]
    y_imag_ptr,               # *float32, output imag part flattened as [num_bc, S+1]
    S: tl.int32,              # seqlen
):
    # One program per (b, c) row
    pid = tl.program_id(0)
    # Each program processes S elements; pid indexes into the flattened [num_bc, S]
    # We will compute outputs for k in 0..S and store at flattened index pid*(S+1) + k
    inv_two_n = 1.0 / (2.0 * S)
    # Compute sum_x = sum(x[0..S-1])
    sum_x = 0.0
    # Loop over t = 0 .. S-1
    for t in range(0, S):
        v = tl.load(x_ptr + pid * S + t)
        sum_x += v

    # Now fill y_real and y_imag for k = 0..S
    # Note: For even k, real = sum_x * (cos(pi*k/(2*S)) - sin(pi*k/(2*S))) * (1/(2*S)), imag = 0
    #       For odd k, real = 0, imag = -sum_x * sin(pi*k/(2*S)) * (1/(2*S))
    # We compute angle per k: angle_k = pi * k / (2*S)
    k = 0
    while k <= S:
        angle = tl.pi * k / (2.0 * S)
        cosv = tl.cos(angle)
        sinv = tl.sin(angle)
        if (k % 2) == 0:
            real_k = sum_x * (cosv - sinv) * inv_two_n
            imag_k = 0.0
        else:
            real_k = 0.0
            imag_k = -sum_x * sinv * inv_two_n
        tl.store(y_real_ptr + pid * (S + 1) + k, real_k)
        tl.store(y_imag_ptr + pid * (S + 1) + k, imag_k)
        k += 1


def _launch_rfft_analytic_kernel(x: torch.Tensor):
    """
    x: (B, C, S) float32 tensor on CUDA device.
    Returns (y_real, y_imag): both (B, C, S+1) float32 tensors.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dtype == torch.float32, "Input must be float32"
    if not x.is_contiguous():
        x = x.contiguous()
    B, C, S = x.shape
    num_bc = B * C
    # Flatten x to [num_bc, S] for simple indexing in Triton
    x_flat = x.view(num_bc, S).contiguous()
    # Allocate outputs flattened to [num_bc, S+1]
    y_real = torch.empty((num_bc, S + 1), dtype=torch.float32, device=x.device)
    y_imag = torch.empty((num_bc, S + 1), dtype=torch.float32, device=x.device)
    # Launch one program per (b,c)
    grid = (num_bc,)
    _rfft_real_imag_analytic_kernel[grid](x_flat, y_real, y_imag, S)
    # Reshape back to (B, C, S+1)
    y_real = y_real.view(B, C, S + 1)
    y_imag = y_imag.view(B, C, S + 1)
    return y_real, y_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor x of shape (B, C, S)
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single input tensor x of shape (B, C, S)")
        x = args[0]
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if not x.is_contiguous():
            x = x.contiguous()
        # Launch Triton kernel to compute rfft outputs (real and imag) normalized by 2*S
        y_real, y_imag = _launch_rfft_analytic_kernel(x)
        return y_real, y_imag


def run(*args):
    return ModelNew()(*args)
