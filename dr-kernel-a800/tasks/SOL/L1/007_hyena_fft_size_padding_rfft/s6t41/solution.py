import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: computes y_real and y_imag for each (b, c) based on x[b, c, :].
# x is flattened per (b,c) to length S, outputs y are flattened per (b,c) to length S+1.
@triton.jit
def _rfft_real_imag_kernel(
    x_ptr,                # *float32, input x of length S per (b,c)
    y_real_ptr,           # *float32, output real part of length S+1 per (b,c)
    y_imag_ptr,           # *float32, output imag part of length S+1 per (b,c)
    S: tl.int32,          # seqlen
    inv_scale: tl.float32 # 1.0 / (2*S)
):
    pid = tl.program_id(0)  # corresponds to a specific (b,c)

    # Compute SUM = sum(x[pid*S : pid*S + S])
    SUM = 0.0
    idx = 0
    while idx < S:
        v = tl.load(x_ptr + pid * S + idx)
        SUM += v
        idx += 1

    # Now fill outputs for k in 0..S
    k = 0
    while k <= S:
        if k == 0:
            tl.store(y_real_ptr + pid * (S + 1) + k, SUM * inv_scale)
            tl.store(y_imag_ptr + pid * (S + 1) + k, 0.0)
        else:
            is_even = (k % 2) == 0
            angle = tl.float32(math.pi * k / (2.0 * S))
            c = tl.cos(angle)
            s = tl.sin(angle)
            scaled_sum = SUM * inv_scale
            if is_even:
                real_k = scaled_sum * (c - s)
                tl.store(y_real_ptr + pid * (S + 1) + k, real_k)
                tl.store(y_imag_ptr + pid * (S + 1) + k, 0.0)
            else:
                imag_k = -scaled_sum * s
                tl.store(y_real_ptr + pid * (S + 1) + k, 0.0)
                tl.store(y_imag_ptr + pid * (S + 1) + k, imag_k)
        k += 1


def _launch_rfft_kernel(x: torch.Tensor):
    """
    x: (B, C, S) float32 tensor on CUDA device.
    Returns (y_real, y_imag): both (B, C, S+1) float32 tensors.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dtype == torch.float32, "Input must be float32"
    if not x.is_contiguous():
        x = x.contiguous()
    B, C, S = x.shape
    num_bc = B * C
    # Flatten x for kernel: one program per (b,c), S elements each
    x_flat = x.view(num_bc, S).contiguous()
    # Allocate outputs as flat: length num_bc*(S+1)
    y_real = torch.empty(num_bc * (S + 1), dtype=torch.float32, device=x.device)
    y_imag = torch.empty(num_bc * (S + 1), dtype=torch.float32, device=x.device)
    # Launch kernel: one program per (b,c)
    grid = (num_bc,)
    inv_scale = 1.0 / (2.0 * S)
    _rfft_real_imag_kernel[grid](x_flat, y_real, y_imag, S, inv_scale)
    # Reshape to (B, C, S+1)
    y_real = y_real.view(B, C, S + 1)
    y_imag = y_imag.view(B, C, S + 1)
    return y_real, y_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor x: (B, C, S)
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single input tensor x of shape (B, C, S)")
        x = args[0]
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if not x.is_contiguous():
            x = x.contiguous()
        # Triton kernel computes rfft outputs (real and imag) normalized by 2*S
        y_real, y_imag = _launch_rfft_kernel(x)
        return y_real, y_imag


def run(*args):
    return ModelNew()(*args)
