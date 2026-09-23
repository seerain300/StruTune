import torch
import triton
import triton.language as tl

# Triton kernel: computes the real DFT of a 1D real vector x of length L,
# pads with zeros to 2*L, and writes normalized real and imaginary parts.
# Launch with grid (B, C, 2*L). Each program handles one k index in the padded domain.
@triton.jit
def real_dft_zero_padded_kernel(
    x_ptr,                 # *const float, input vector of length L (flattened)
    out_real_ptr,          # *float, output real part of length 2*L (flattened)
    out_imag_ptr,          # *float, output imag part of length 2*L (flattened)
    L: tl.constexpr,       # int, original seqlen
    total_len: tl.constexpr,  # int, 2*L (padded length)
    # B and C are not strictly needed in-kernel; we reshape outputs on host.
):
    # Program ids: grid = (B, C, total_len) => one program per (b, c, k)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_k = tl.program_id(2)  # k index in [0, 2*L - 1]

    # Each (b, c) owns a contiguous vector of length L starting at offset (pid_b*C + pid_c)*L.
    # We pass x as a flat vector so that offset is just (pid_b*C + pid_c)*L.
    base_offset = (pid_b * C + pid_c) * L  # C and base_offset not needed since x is flat; left for clarity.

    # Accumulator for sum in complex
    acc = 0.0 + 0.0j

    # Constants
    two_L = total_len
    inv_two_L = 1.0 / two_L
    pi = 3.141592653589793

    # Loop over t from 0 to 2*L - 1. Note: x_ptr + (pid_b*C + pid_c)*L + t is valid only for t < L.
    # For t >= L, x[t] should be treated as 0 (zero-padding). We load x_val and it will be 0 for t >= L
    # because x is length L; however, to explicitly zero-pad, we can branch. But since we flattened x,
    # the load will not access beyond L. To be correct, we keep the loop up to two_L - 1.
    # For t >= L, x_ptr + base_offset + t points out of the original x; since we flattened x to length B*C*L,
    # those elements aren't guaranteed to exist. Therefore, we must ensure x is length total_len or guard loads.
    # The simplest: we pass x_flat and the kernel assumes x_flat has length B*C*L and we do not load beyond that.
    # However, here we want to sum up to 2*L - 1, so we must pass x padded. For simplicity, we assume x is long
    # enough; but since we only have L elements, we cannot. Fix: allocate a padded_x of length two_L on host
    # and pass it to the kernel. To avoid changing host, we modify the kernel to require padded_x, but since
    # the original run pads implicitly in torch.fft.rfft, we can instead modify host to pass a padded tensor.

    # NOTE: The above comment indicates a design gap. In practice, we cannot have x_flat length B*C*L and
    # iterate up to two_L because x only has L elements. Therefore, we need to allocate x_padded of length two_L
    # and pass it to the kernel. To keep the interface simple, we change the kernel to accept padded_x directly.
    # But since the original forward uses x of shape (B, C, L), we cannot create padded_x here. Hence, we adjust
    # the approach: we will pad x on host and then flatten padded_x.

    # REWORK: Instead of relying on flattened pointer arithmetic, we will make the host pass a contiguous padded
    # vector of length two_L. This simplifies the kernel and guarantees correct zero-padding. We will do that
    # by padding x on host and passing the padded vector to the kernel.

    # The current implementation is simpler: we pass x_flat of length B*C*L and loop up to two_L. For t >= L,
    # x_ptr + base_offset + t is out-of-range. This is illegal. Therefore, the kernel must receive a padded_x
    # of length two_L. To satisfy that, we will change the kernel to require padded_x_ptr. Since Triton doesn't
    # allow easy redefinition, we implement a correct version below that receives padded_x.

    # The following is a corrected version of the kernel that expects a padded_x_ptr of length two_L.

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton version of the original run function.
        Computes real FFT with zero-padding to 2*seqlen using a custom Triton kernel,
        and returns real and imaginary parts (normalized by 2*L to match the original code).
        """
        assert x.ndim == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape

        # Ensure contiguous and float32
        x = x.contiguous().to(torch.float32)

        # Total length after padding
        two_L = 2 * L

        # Create a zero-padded input of length 2*L
        # Note: original torch.fft.rfft(x, n=2*L) implicitly pads with zeros. Here we explicitly pad to ensure
        # the Triton kernel sees the padded data. The padded data for t in [L, 2*L-1] should be zeros.
        # We need x_padded with shape (B, C, two_L). Then we flatten.
        x_padded = torch.zeros((B, C, two_L), device=x.device, dtype=x.dtype)
        # Copy original x into the first L positions along the last dimension
        x_padded[..., :L] = x
        x_flat_padded = x_padded.reshape(-1).contiguous()  # length = B*C*two_L? No: B*C*L for x_padded is (B,C,2L) => B*C*2*L elements? Not right.
        # Correction: x_padded is (B, C, 2*L), so flattening gives B*C*2*L elements, which we do not want.
        # Instead, we will flatten only the last dimension for each (b,c), i.e., treat each (b,c) as a separate
        # vector of length two_L. The easiest is to treat the whole tensor as a single contiguous vector of length B*C*2*L,
        # but then each program (b,c,k) cannot access only its (b,c) slice. Therefore, we will instead pass x_padded
        # as a single contiguous vector of length B*C*2*L and compute base_offset = (pid_b*C + pid_c)*two_L for each (b,c).

        # Simpler approach: make x_padded_flat of length (B*C*2*L) where each (b,c) slice occupies two_L consecutive
        # elements. We can build this by reshaping x_padded to (-1,) and using base_offset for each (b,c).

        # Prepare x_flat_padded correctly: flatten entire (B, C, 2*L) to one vector of length B*C*2*L
        x_flat_padded = x_padded.reshape(-1).contiguous()  # length = B*C*2*L

        # Allocate outputs (real and imag), shape (B, C, 2*L), float32
        out_real = torch.empty((B, C, two_L), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, two_L), device=x.device, dtype=torch.float32)

        # Launch Triton kernel: grid over (B, C, 2*L), one program per (b, c, k)
        grid = (B, C, two_L)

        # We need to compute base offset for each (b, c) slice in the flattened vector.
        # Each (b,c) slice has two_L elements. So for a flattened vector of length M = B*C*2*L,
        # the start index for (b,c) is idx0 = (b*C + c) * two_L. Then x_flat_padded[idx0 + t] = padded_x[b,c,t].
        # The kernel will use base_offset = idx0 per (b,c). We pass this via tl.constexpr-like indexing by using
        # that the program ids (b,c) can compute idx0 = pid_b*C*two_L + pid_c*two_L. But Triton requires tensors,
        # not host-computed idx0. To handle this, we redesign: we launch a 2D grid over (B*C, 2*L) and pass base_offset
        # via the combined program_id(0).

        # Revised kernel launch: use 2D grid (B*C, 2*L)

        # We will create a simpler kernel that expects a single flattened vector of length M = B*C*2*L and uses
        # base_offset = pid % (C*two_L) * two_L + k, but that doesn't work cleanly. Instead, we switch to the
        # combined (B*C, 2*L) grid.

        # Implement the revised Triton kernel that receives x_ptr of length M and computes for each (b,c,k):
        # base_offset = (pid_b * C + pid_c) * two_L; then load x_ptr[base_offset + k].

        # Note: We need a new kernel for this approach. We define it now.

@triton.jit
def real_dft_zero_padded_kernel_2d(
    x_ptr,                 # *const float, input padded vector of length M = B*C*2*L
    out_real_ptr,          # *float, output real part flattened of length M
    out_imag_ptr,          # *float, output imag part flattened of length M
    L: tl.constexpr,       # int, original seqlen
    two_L: tl.constexpr,   # int, 2*L
    C: tl.constexpr,       # int, channels
):
    # Grid: (B*C, 2*L) => pid0 = b*c index, pid1 = k
    pid0 = tl.program_id(0)  # combined b*c
    pid1 = tl.program_id(1)  # k index in [0, 2*L-1]

    # Compute base offset for (b,c) slice
    bc = pid0
    # Each (b,c) slice has two_L elements. The flattened vector packs (b,c) slices consecutively.
    base_offset = bc * two_L

    # Accumulator for sum in complex
    acc = 0.0 + 0.0j

    # Constants
    pi = 3.141592653589793
    inv_two_L = 1.0 / two_L

    # Loop over t from 0 to 2*L - 1
    for t in range(0, two_L):
        # Load x[t] from the padded vector
        x_val = tl.load(x_ptr + base_offset + t)  # x_val is real
        angle = -2.0 * pi * pid1 * t * inv_two_L
        exp_real = tl.cos(angle)
        exp_imag = tl.sin(angle)
        # acc += x_val * (exp_real + i*exp_imag)
        acc += x_val * (exp_real + exp_imag * 1j)

    # Normalize by 2*L (matching original code)
    acc_norm = acc * inv_two_L

    # Store real and imaginary parts
    tl.store(out_real_ptr + base_offset + pid1, acc_norm.real)
    tl.store(out_imag_ptr + base_offset + pid1, acc_norm.imag)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton version of the original run function.
        Computes real FFT with zero-padding to 2*seqlen using a custom Triton kernel,
        and returns real and imaginary parts (normalized by 2*L to match the original code).
        """
        assert x.ndim == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape

        # Ensure contiguous and float32
        x = x.contiguous().to(torch.float32)

        # Total length after padding
        two_L = 2 * L

        # Create a zero-padded input of shape (B, C, 2*L)
        x_padded = torch.zeros((B, C, two_L), device=x.device, dtype=x.dtype)
        # Copy original x into the first L positions along the last dimension
        x_padded[..., :L] = x

        # Flatten padded tensor to a single vector of length M = B*C*2*L
        x_flat_padded = x_padded.reshape(-1).contiguous()  # length = B*C*2*L

        # Allocate outputs (real and imag), shape (B, C, 2*L), float32, flattened
        out_real = torch.empty((B, C, two_L), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, two_L), device=x.device, dtype=torch.float32)
        out_real_flat = out_real.reshape(-1)  # length = B*C*2*L
        out_imag_flat = out_imag.reshape(-1)  # length = B*C*2*L

        # Launch Triton kernel: grid over (B*C, 2*L), one program per (b,c,k)
        grid = (B * C, two_L)

        real_dft_zero_padded_kernel_2d[grid](
            x_flat_padded, out_real_flat, out_imag_flat,
            L=L, two_L=two_L, C=C,
            num_warps=1,  # small work per program; keep it simple
            num_stages=1,
        )

        # Return real and imaginary parts as requested
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
