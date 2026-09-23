import torch
import triton
import triton.language as tl

# Triton kernel: for each (b, c), compute real outputs X[0..L] of rfft over a zero-padded input of length 2*L.
# We pass the flattened zero-padded input vector of length two_L. The kernel ignores the last L zeros in reduction,
# but we still pass zeros to keep vector length = 2*L.
@triton.jit
def rfft_real_kernel(
    in_ptr,          # *float32, flattened zero-padded input vector per (b, c), length = 2*L
    out_ptr,         # *float32, output real part vector per (b, c), length = L+1
    L: tl.constexpr, # seqlen (input length)
    two_L: tl.constexpr, # 2*L (FFT size)
):
    # This kernel is launched with grid (B*C,), i.e., one program per (b, c).
    b_c_id = tl.program_id(0)  # corresponds to (b, c) when launching with B*C

    # Precompute sums: sum over all t, sum over odd t, sum over even t
    # We iterate over t in [0..2*L-1], but use arithmetic to separate odd/even.
    total_sum = 0.0
    odd_sum = 0.0
    even_sum = 0.0

    # Loop over t in the padded input vector
    for t in range(2 * two_L):
        # Map to original input index (first L positions are meaningful; rest are zeros)
        # If t < L, read in_ptr[t]; else assume zero, but in our data preparation we already padded zeros.
        val = tl.load(in_ptr + t, mask=t < two_L, other=0.0)
        total_sum += val
        if (t % 2) == 1:
            odd_sum += val
        else:
            even_sum += val

    # X[0] = sum / (2*L)
    x0 = total_sum / (two_L)

    # X[L] = (sum over odd + sum over even) / (2*L)
    xL = (odd_sum + even_sum) / (two_L)

    # Write X[0] and X[L] to out vector at indices 0 and L
    tl.store(out_ptr + 0, x0)
    tl.store(out_ptr + L, xL)

    # Compute X[1] and X[2..L-1] using cos/sin
    # Note: For real inputs, imaginary part is zero.
    # X[1] = (sum odd - sum even) / (2*L)
    x1 = (odd_sum - even_sum) / (two_L)
    tl.store(out_ptr + 1, x1)

    # Now compute X[k] for k in [2..L-1]
    # We'll compute each k sequentially. Triton allows loops with constexpr limits; L is constexpr here.
    for k in range(2, L):
        cos_term = 0.0
        sin_term = 0.0
        for t in range(2 * two_L):
            # Only need t in [0..L-1] since input is zero-padded beyond that.
            if t < two_L:
                val = tl.load(in_ptr + t, mask=t < two_L, other=0.0)
                angle = 2.0 * 3.141592653589793 * float(k) * float(t) / float(two_L)
                cos_term += val * tl.cos(angle)
                sin_term += val * tl.sin(angle)
        xk = (cos_term - sin_term) / (two_L)
        tl.store(out_ptr + k, xk)

    # X[L] already computed; nothing left to do.


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure dtype and contiguity
        x = x.to(torch.float32).contiguous()
        B, C, L = x.shape
        two_L = 2 * L

        # Prepare zero-padded input per (b, c): first L elements are x[b, c, :], next L zeros
        # This is device-side tensor preparation (torch) but not torch GPU compute in forward.
        x_flat = x.view(B, C, L).reshape(B * C, L)  # shape (B*C, L)
        in_pad = torch.cat([x_flat, torch.zeros((B * C, L), dtype=torch.float32, device=x.device)], dim=1)  # shape (B*C, 2*L)
        in_flat = in_pad.reshape(B * C * (2 * L))  # flatten into 1D for kernel

        # Output real part buffer per (b, c): length L+1
        out_real = torch.empty(B * C * (L + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c)
        grid = (B * C,)
        rfft_real_kernel[grid](
            in_flat, out_real,
            L=L, two_L=two_L,
        )

        # Reshape to (B, C, L+1)
        x_freq_real = out_real.view(B, C, L + 1)
        x_freq_imag = torch.zeros_like(x_freq_real)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
