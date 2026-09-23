import torch
import triton
import triton.language as tl


# Kernel 1: Zero-pad input of length L to a buffer of length n (n = 2*L)
@triton.jit
def _pad_to_n_kernel(x_ptr, padded_ptr, L, n, BLOCK: tl.constexpr):
    """
    x_ptr: *float32, length L (original seqlen)
    padded_ptr: *float32, length n (2*seqlen), output buffer to be zero-padded
    Each program writes one element of padded_ptr.
    """
    pid = tl.program_id(axis=0)
    if pid < L:
        tl.store(padded_ptr + pid, tl.load(x_ptr + pid))
    else:
        tl.store(padded_ptr + pid, 0.0)


# Kernel 2: Direct rFFT for k in [0..L] using explicit padding to n
# y[k] = sum_{j=0}^{n-1} x[j] * exp(-i * 2*pi*k*j / n)
# We compute real and imag parts separately: real += xj*cos, imag += xj*sin
@triton.jit
def _rfft_simple_kernel(padded_ptr, out_real_ptr, out_imag_ptr, L, n, BLOCK_K: tl.constexpr):
    """
    padded_ptr: *float32, length n, contains input x padded with zeros
    out_real_ptr: *float32, length (L+1)
    out_imag_ptr: *float32, length (L+1)
    """
    pid = tl.program_id(axis=0)
    k0 = pid * BLOCK_K
    ks = k0 + tl.arange(0, BLOCK_K)
    mask_k = ks <= L  # output length is L+1

    acc_real = tl.zeros([BLOCK_K], dtype=tl.float32)
    acc_imag = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Sum over j from 0 to n-1
    for j in range(0, n):
        xj = tl.load(padded_ptr + j)  # xj is float32
        # angle = 2*pi*k*j / n
        angle = 2.0 * 3.141592653589793 * ks * j * (1.0 / n)
        cosv = tl.cos(angle)
        sinv = tl.sin(angle)
        acc_real += xj * cosv
        acc_imag += xj * sinv

    # Store results for ks up to L (Nyquist index)
    tl.store(out_real_ptr + ks, acc_real, mask=mask_k)
    tl.store(out_imag_ptr + ks, acc_imag, mask=mask_k)


# Kernel 3: Elementwise normalization by division with scalar
@triton.jit
def _normalize_divide_kernel(in_ptr, out_ptr, numel, scale, BLOCK: tl.constexpr):
    """
    out[i] = in[i] / scale
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = x / scale
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect single input tensor of shape (batch, channels, seqlen)
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single input tensor")
        x = args[0]
        if x.dim() != 3:
            raise ValueError("Input must be a 3D tensor of shape (batch, channels, seqlen)")
        # Cast to float32 (original code does this)
        x_f32 = x.to(torch.float32)
        batch, channels, seqlen = x_f32.shape
        device = x_f32.device

        L = seqlen
        n = 2 * L  # padding size as in original

        # 1) Zero-pad input to length n
        padded = torch.empty(n, dtype=torch.float32, device=device)
        # Launch padding kernel: one program per index
        grid_pad = (n,)
        _pad_to_n_kernel[grid_pad](x_f32, padded, L, n, BLOCK=1)

        # 2) Compute rFFT for k in [0..L] using direct method on padded buffer
        out_real = torch.empty(L + 1, dtype=torch.float32, device=device)
        out_imag = torch.empty(L + 1, dtype=torch.float32, device=device)

        # Choose BLOCK_K; since output length is small in typical cases, 128 works fine.
        BLOCK_K = 128
        grid_k = (triton.cdiv(L + 1, BLOCK_K),)
        _rfft_simple_kernel[grid_k](padded, out_real, out_imag, L, n, BLOCK_K=BLOCK_K)

        # 3) Normalize by 2*L using Triton elementwise division
        scale = 2.0 * L
        grid_norm = (triton.cdiv(out_real.numel(), 1024),)
        _normalize_divide_kernel[grid_norm](out_real, out_real, out_real.numel(), scale, BLOCK=1024)
        grid_norm = (triton.cdiv(out_imag.numel(), 1024),)
        _normalize_divide_kernel[grid_norm](out_imag, out_imag, out_imag.numel(), scale, BLOCK=1024)

        # 4) Shape to (batch, channels, seqlen+1) and return
        x_freq_real = out_real.unsqueeze(0).expand(batch, channels, L + 1).contiguous()
        x_freq_imag = out_imag.unsqueeze(0).expand(batch, channels, L + 1).contiguous()
        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
