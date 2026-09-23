import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def build_y_kernel(
    x_ptr,           # *f32, input tensor (B, C, L), contiguous
    y_ptr,           # *f32, output tensor (B, C, 2*L), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,     # original seqlen
    stride_x_b: tl.int32,
    stride_x_c: tl.int32,
    stride_x_l: tl.int32,
    stride_y_b: tl.int32,
    stride_y_c: tl.int32,
    stride_y_l: tl.int32,
    twoL: tl.int32,
    BLOCK_N: tl.constexpr,  # tile size for j
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_x = b * stride_x_b + c * stride_x_c
    base_y = b * stride_y_b + c * stride_y_c

    # First half: copy x[0..L-1] to y[0..L-1]
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_x + j * stride_x_l, mask=mask, other=0.0)
        tl.store(y_ptr + base_y + j * stride_y_l, vals, mask=mask)
        start += BLOCK_N

    # Second half: write reverse of x into y[L..2L-1]
    # For j in [L, 2L-1], let idx = 2L - 1 - j + L -> idx = 3L - 1 - j
    j = L
    while j < twoL:
        idx = (3 * L - 1) - j
        # Load from x at reversed position
        # idx is in [0, L-1] when j in [L, 2L-1]
        val = tl.load(x_ptr + base_x + idx * stride_x_l, mask=True, other=0.0)
        tl.store(y_ptr + base_y + j * stride_y_l, val)
        j += 1


@triton.jit
def real_dft_kernel(
    y_ptr,            # *f32, input extended tensor (B, C, 2*L), contiguous
    real_out_ptr,     # *f32, output real part (B, C, L+1), contiguous
    imag_out_ptr,     # *f32, output imag part (B, C, L+1), contiguous
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,      # original seqlen
    out_stride_b: tl.int32,
    out_stride_c: tl.int32,
    out_stride_l: tl.int32,
    twoL: tl.int32,
    BLOCK_K: tl.constexpr,   # tile size for k (frequency index)
    BLOCK_N: tl.constexpr,   # tile size for j (time index)
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_y = b * y_ptr.stride(0) + c * y_ptr.stride(1)
    base_out = b * real_out_ptr.stride(0) + c * real_out_ptr.stride(1)

    # Initialize accumulators for X[k] for k = 0..L
    # We'll compute real and imag parts separately. Note: rfft(X_real) has imaginary part 0,
    # but here we directly compute X via direct DFT for real input:
    # X[k] = sum_{j=0}^{2*L-1} y[j] * (cos(2*pi*k*j/twoL) - i*sin(2*pi*k*j/twoL))
    # For k > L, we don't need to compute since output is (L+1), but we will compute up to k=L.
    twoL_f = twoL.to(tl.float32)

    # Tile over k
    k_start = 0
    while k_start <= L:
        k_vec = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_vec <= L

        real_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        imag_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

        # Accumulate over j = 0..2*L-1
        j = 0
        while j < twoL:
            j_vec = j + tl.arange(0, BLOCK_N)
            mask_j = j_vec < twoL

            # Load y[b, c, j]
            vals = tl.load(y_ptr + base_y + j_vec * y_ptr.stride(2), mask=mask_j, other=0.0)  # shape [BLOCK_N]

            # Compute theta = 2*pi*k*j / twoL
            theta = (2.0 * tl.pi) * (k_vec[:, None].to(tl.float32)) * (j_vec[None, :].to(tl.float32) / twoL_f)

            cos_t = tl.cos(theta)
            sin_t = tl.sin(theta)

            # Accumulate: sum_j vals[j] * (cos - i*sin)
            prod_real = vals * cos_t
            prod_imag = vals * sin_t

            real_acc += tl.sum(prod_real, axis=1)
            imag_acc += tl.sum(prod_imag, axis=1)

            j += BLOCK_N

        # Normalize by 2*L (original division by 2*seqlen)
        norm = 1.0 / (2.0 * L)
        real_acc = real_acc * norm
        imag_acc = imag_acc * norm

        # Store results for k = k_start .. k_start + BLOCK_K - 1
        # For each k lane, if valid, store to output
        for kk in range(BLOCK_K):
            if mask_k[kk]:
                out_idx = k_start + kk
                tl.store(real_out_ptr + base_out + out_idx * out_stride_l, real_acc[kk])
                tl.store(imag_out_ptr + base_out + out_idx * out_stride_l, imag_acc[kk])

        k_start += BLOCK_K


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Expect x of shape (batch, channels, seqlen)
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        batch, channels, L = x.shape
        twoL = 2 * L

        # Cast to float32 for numerical stability
        x_f32 = x.to(torch.float32)

        # Allocate y of shape (B, C, 2*L), contiguous
        y = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch kernel to build y: first half = x, second half = reverse(x)
        grid_build = (batch * channels,)
        build_y_kernel[grid_build](
            x_f32, y,
            batch, channels, L,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            twoL,
            BLOCK_N=256,
            num_warps=4,
        )

        # Allocate outputs: real and imag parts of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch real DFT Triton kernel to compute X[k] for k = 0..L
        grid_dft = (batch * channels,)
        real_dft_kernel[grid_dft](
            y, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            twoL,
            BLOCK_K=64,    # tile for k (frequency index)
            BLOCK_N=256,   # tile for j (time index)
            num_warps=4,
        )

        # Note: For real input, rfft has only real output. imag_out should be zeros.
        # We didn't compute Y[L] explicitly in the kernel. To match torch.rfft:
        # X[seqlen] = Y[L] = sum_j y[j] * exp(-i*pi*j) = sum_j y[j] * (-1)^j
        # Compute it separately and set imag to 0.
        # We compute it via torch ops (still fast) and write into the last index.
        # Construct y_slice per (b,c) and sum y[j]*(-1)^j.
        # We can do this in a simple loop or use torch.view and indexing.
        # Since y is contiguous and (B,C,2L), we index per slice.
        # Let's compute per (b,c):
        # y_per = y.view(batch, channels, twoL)  # already correct shape
        # For each (b,c), compute sum y[j]*((-1)**j) and write to real_out[:, :, L], imag_out[:, :, L] = 0.
        # But we need to avoid torch here to stay Triton-only. Instead, we can do it in Triton with a tiny kernel.

        # Triton kernel to compute X[L] = sum_j y[j] * (-1)^j per (b,c):
        # Implement a simple 1D kernel that iterates over twoL and accumulates.
        @triton.jit
        def compute_x_at_L_kernel(
            y_ptr,            # *f32
            real_out_ptr,     # *f32
            batch: tl.int32,
            channels: tl.int32,
            L: tl.int32,
            stride_y_b: tl.int32,
            stride_y_c: tl.int32,
            stride_y_l: tl.int32,
            out_stride_b: tl.int32,
            out_stride_c: tl.int32,
            out_stride_l: tl.int32,
            twoL: tl.int32,
        ):
            pid = tl.program_id(axis=0)
            b = pid // channels
            c = pid % channels
            base_y = b * stride_y_b + c * stride_y_c
            base_out = b * out_stride_b + c * out_stride_c

            total = tl.zeros([1], dtype=tl.float32)
            j = 0
            while j < twoL:
                val = tl.load(y_ptr + base_y + j * stride_y_l)
                # (-1)^j: parity check
                parity = (j % 2).to(tl.int32)
                # if j is even, add; else subtract
                total += val if parity == 0 else (-val)
                j += 1

            # Write real part at index L, imag part = 0
            tl.store(real_out_ptr + base_out + L * out_stride_l, total[0])
            tl.store(imag_out_ptr + base_out + L * out_stride_l, 0.0)

        # Launch the tiny kernel to set the Nyquist term correctly
        grid_nyq = (batch * channels,)
        compute_x_at_L_kernel[grid_nyq](
            y,
            real_out,
            batch, channels, L,
            y.stride(0), y.stride(1), y.stride(2),
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            twoL,
            num_warps=1,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
