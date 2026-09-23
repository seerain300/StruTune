import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_kernel(x_ptr, out_ptr, L: tl.int32):
    # Copy x[pid, :] into out[pid, 0:L]
    pid = tl.program_id(0)
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + pid * L + i)
        tl.store(out_ptr + pid * L + i, val)


@triton.jit
def _dot_product_real_kernel(padded_ptr, realk_ptr, out_ptr, N: tl.int32):
    # Each program handles one k (scalar), accumulates sum over j
    k = tl.program_id(0)
    acc = 0.0
    for j in tl.static_range(0, N):
        xj = tl.load(padded_ptr + j)
        rk = tl.load(realk_ptr + j)
        acc += xj * rk
    # Store the result
    tl.store(out_ptr + k, acc)


@triton.jit
def _dot_product_imag_kernel(padded_ptr, imagk_ptr, out_ptr, N: tl.int32):
    # Each program handles one k (scalar), accumulates sum over j
    k = tl.program_id(0)
    acc = 0.0
    for j in tl.static_range(0, N):
        xj = tl.load(padded_ptr + j)
        ik = tl.load(imagk_ptr + j)
        acc += xj * ik
    tl.store(out_ptr + k, acc)


@triton.jit
def _elementwise_div_kernel(in_ptr, out_ptr, val: tl.float32, N: tl.int32):
    # out = in / val
    for i in tl.static_range(0, N):
        vi = tl.load(in_ptr + i)
        tl.store(out_ptr + i, vi * (1.0 / val))


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L)
        B, C, L = x.shape
        device = x.device
        # Precompute real and imag bins on host (PyTorch tensors)
        N = 2 * L
        j = torch.arange(N, device=device, dtype=torch.float32)
        inv_N = 1.0 / float(N)
        k_all = torch.arange(L + 1, device=device, dtype=torch.float32)  # indices 0..L
        real_bins = torch.cos(2.0 * 3.141592653589793 * (k_all[:, None]) * j[None, :] * inv_N).T  # (N,)
        imag_bins = torch.sin(2.0 * 3.141592653589793 * (k_all[:, None]) * j[None, :] * inv_N).T  # (N,)

        # Prepare outputs
        real_out = torch.empty((B, C, L + 1), device=device, dtype=torch.float32)
        imag_out = torch.empty((B, C, L + 1), device=device, dtype=torch.float32)

        # Cast input to float32 for numerical stability (explicit, per requirement)
        x_f32 = x.to(torch.float32)

        # For each (b, c), copy row into padded buffer of length N, zero the tail
        # Allocate padded_x for this (b, c) slice and fill zeros
        for b in range(B):
            for c in range(C):
                # padded_x has length N
                padded_x = torch.zeros(N, device=device, dtype=torch.float32)
                # Copy x[b, c, :] into padded_x[0:L]
                # x[b, c, :] is contiguous of length L
                # We need a pointer to start at offset b*C*L
                # However, since x is (B, C, L), the memory layout is contiguous in L, then C, then B.
                # We can flatten and compute pointer: (b*C + c) * L
                x_row_ptr = x_f32[b, c, :].contiguous()
                out_row_ptr = padded_x
                # Launch copy kernel
                # Note: Triton grid is 1D with size 1 (one program)
                _copy_row_kernel[(1,)](x_row_ptr, out_row_ptr, L)
                # Now compute dot products for all k in [0..L]
                # We will write results into real_out[b, c, :] and imag_out[b, c, :]
                for k in range(L + 1):
                    # Launch real and imag dot product kernels
                    # out_real_k[k] and out_imag_k[k]
                    # We can use a 1D grid with size 1 per k
                    # real_bins and imag_bins are 1D of length N
                    real_k = torch.empty(1, device=device, dtype=torch.float32)
                    imag_k = torch.empty(1, device=device, dtype=torch.float32)
                    _dot_product_real_kernel[(1,)](padded_x, real_bins, real_k, N)
                    _dot_product_imag_kernel[(1,)](padded_x, imag_bins, imag_k, N)
                    # Store to outputs
                    # Flatten to 1D length L+1
                    real_out[b, c, k] = real_k[0]
                    imag_out[b, c, k] = imag_k[0]
                # Normalize by N=2*L
                real_out_norm = torch.empty_like(real_out[b, c, :])
                imag_out_norm = torch.empty_like(imag_out[b, c, :])
                _elementwise_div_kernel[(L + 1,)](real_out[b, c, :], real_out_norm, float(N), L + 1)
                _elementwise_div_kernel[(L + 1,)](imag_out[b, c, :], imag_out_norm, float(N), L + 1)
                # Assign back
                real_out[b, c, :] = real_out_norm
                imag_out[b, c, :] = imag_out_norm

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
