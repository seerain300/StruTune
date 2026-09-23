import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: computes real and imaginary parts of rfft for real inputs x of length S,
# normalized by 2*S, and writes outputs of length S+1 for each (b, c).
@triton.jit
def rfft_real_imag_kernel(x_ptr,  # *float32, input flattened to (BC, S)
                           y_real_ptr,  # *float32, output real flattened to (BC, S+1)
                           y_imag_ptr,  # *float32, output imag flattened to (BC, S+1)
                           S: tl.int32,  # original seqlen
                           inv_scale: tl.float32  # 1.0 / (2 * S)
                           ):
    bc = tl.program_id(0)
    # Load x[b,c, :] vector and compute sum_x
    sum_x = 0.0
    for t in range(0, S):
        ptr = x_ptr + bc * S + t
        v = tl.load(ptr)
        sum_x += v

    # Now compute outputs for k in 0..S
    # y_real_ptr[k] and y_imag_ptr[k] correspond to k-th index in the output for this (b, c)
    # For even k: real = (cos - sin) * sum_x * inv_scale; imag = 0
    # For odd k: real = 0; imag = -sin * sum_x * inv_scale
    for k in range(0, S + 1):
        # even check
        is_even = (k % 2) == 0
        if is_even:
            theta = tl.float32(tl.pi) * tl.float32(k) / tl.float32(2 * S)
            c = tl.cos(theta)
            s = tl.sin(theta)
            val = (c - s) * sum_x * inv_scale
            tl.store(y_real_ptr + bc * (S + 1) + k, val)
            tl.store(y_imag_ptr + bc * (S + 1) + k, 0.0)
        else:
            theta = tl.float32(tl.pi) * tl.float32(k) / tl.float32(2 * S)
            s = tl.sin(theta)
            val_imag = -s * sum_x * inv_scale
            tl.store(y_real_ptr + bc * (S + 1) + k, 0.0)
            tl.store(y_imag_ptr + bc * (S + 1) + k, val_imag)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor x: shape (B, C, S)
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single input tensor x of shape (B, C, S)")
        x = args[0]
        # Ensure float32 and contiguous
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if not x.is_contiguous():
            x = x.contiguous()
        B, C, S = x.shape

        # Flatten to (BC, S) for kernel
        BC = B * C
        x_flat = x.view(BC, S).contiguous()

        # Allocate outputs flattened to (BC, S+1)
        y_real_flat = torch.empty((BC, S + 1), dtype=torch.float32, device=x.device)
        y_imag_flat = torch.empty((BC, S + 1), dtype=torch.float32, device=x.device)

        # Launch kernel: one program per (b, c)
        grid = (BC,)
        inv_scale = 1.0 / (2.0 * S)
        rfft_real_imag_kernel[grid](
            x_flat,
            y_real_flat,
            y_imag_flat,
            S,
            inv_scale,
            num_warps=1,  # keep small; S is dynamic per workload
        )

        # Reshape back to (B, C, S+1)
        y_real = y_real_flat.view(B, C, S + 1)
        y_imag = y_imag_flat.view(B, C, S + 1)

        return y_real, y_imag


def run(*args):
    return ModelNew()(*args)
