import torch
import triton
import triton.language as tl

# Triton kernel: compute rfft output for one (b, c) row into out_real/out_imag of shape (L+1)
@triton.jit
def rfft_row_kernel(x_ptr, out_real_ptr, out_imag_ptr,
                     B, C, L, stride_b, stride_c, stride_l,
                     out_stride_b, out_stride_c, out_stride_l,
                     N: tl.constexpr,  # N = 2*L
                     BLOCK_J: tl.constexpr):
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    # Base pointers for the current (b, c) row
    x_row_ptr = x_ptr + b * stride_b + c * stride_c
    out_row_real_ptr = out_real_ptr + b * out_stride_b + c * out_stride_c
    out_row_imag_ptr = out_imag_ptr + b * out_stride_b + c * out_stride_c

    invN = 1.0 / N  # normalization factor

    # Precompute 2*pi / N for cosine/sine
    two_pi_over_N = 2.0 * 3.141592653589793 / N

    # For each output bin k in [0..L], compute sum_{j=0..N-1} x[j] * (cos(2*pi*k*j/N) + i*sin(2*pi*k*j/N))
    for k in tl.static_range(0, L + 1):
        real_acc = 0.0
        imag_acc = 0.0

        # Loop over j in tiles of BLOCK_J
        for jj in range(0, N, BLOCK_J):
            j_vec = jj + tl.arange(0, BLOCK_J)
            mask = j_vec < N

            # Load x[b, c, j] for j < L; for j >= L, it's zero in input, but we avoid invalid loads via mask
            # We only use j < L, so load with mask & (j_vec < L)
            valid = mask & (j_vec < L)
            # For masked loads, we use other=0.0 to avoid undefined values
            x_vals = tl.load(x_row_ptr + j_vec * stride_l, mask=valid, other=0.0)

            # Compute cos and sin for this k across the BLOCK_J j's
            angle = two_pi_over_N * k * j_vec
            cos_t = tl.cos(angle)
            sin_t = tl.sin(angle)

            # Accumulate: real += sum(x_vals * cos_t), imag += sum(x_vals * sin_t)
            # Mask out contributions for j >= L: x_vals for j>=L is already 0, so safe.
            real_acc += tl.sum(x_vals * cos_t, axis=0)
            imag_acc += tl.sum(x_vals * sin_t, axis=0)

        # Normalize by N
        real_acc *= invN
        imag_acc *= invN

        # Store to out[b, c, k]
        tl.store(out_row_real_ptr + k * out_stride_l, real_acc)
        tl.store(out_row_imag_ptr + k * out_stride_l, imag_acc)

# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, L) float tensor on CUDA device
        returns:
            out_real: (B, C, L+1) float32
            out_imag: (B, C, L+1) float32
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), "Input must be a floating dtype."
        # Cast to float32 for numerical stability (matches original behavior which casts to float32)
        x_f32 = x.to(torch.float32)
        B, C, L = x_f32.shape
        N = 2 * L  # length for rfft

        # Allocate outputs (B, C, L+1) float32
        out_real = torch.empty((B, C, L + 1), device=x_f32.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L + 1), device=x_f32.device, dtype=torch.float32)

        # Strides
        stride_b, stride_c, stride_l = x_f32.stride()
        out_stride_b, out_stride_c, out_stride_l = out_real.stride()

        # Launch one program per (b, c) row
        grid = (B * C,)
        # Choose a block size for j-loop; 128 is a reasonable default. Triton will compile per L and N constexpr.
        BLOCK_J = 128

        rfft_row_kernel[grid](
            x_f32, out_real, out_imag,
            B, C, L, stride_b, stride_c, stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            N=N, BLOCK_J=BLOCK_J,
            num_warps=4,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
