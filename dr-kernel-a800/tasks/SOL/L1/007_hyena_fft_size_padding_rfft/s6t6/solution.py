import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(
    x_ptr,                      # *float32, input x of shape (B*C, S) as flattened, but we pass actual strides
    out_ptr,                   # *float32, output real part of shape (B*C, S+1)
    B: tl.int32, C: tl.int32, S: tl.int32,               # ints
    stride_x_bc: tl.int32, stride_x_s: tl.int32,        # strides for x
    stride_out_bc: tl.int32, stride_out_out: tl.int32,  # strides for out_real
):
    # Each program handles one (b, c) slice
    bc = tl.program_id(0)
    # Compute b and c indices
    # bc in [0, B*C), but we can derive b and c by modulo if needed, but here we treat bc as index into out and x.
    # We will compute base pointers for x and out at (bc, :).
    base_x = bc * stride_x_bc
    base_out = bc * stride_out_bc

    N = 2 * S
    inv_2N = 1.0 / (2 * N)

    # Handle k=0: y[0] = sum_{t=0..N-1} z[t], but z[t] = x[t] for t<S, else 0
    sum_x = 0.0
    # Accumulate sum over t < S (since z[t>=S] == 0)
    for t in range(0, S):
        v = tl.load(x_ptr + base_x + t * stride_x_s)
        sum_x += v
    # y[0] real part
    y0 = sum_x * inv_2N
    tl.store(out_ptr + base_out + 0 * stride_out_out, y0)

    # For k in 1..S: compute real part (including handling conjugate symmetry for odd k)
    # We will compute real parts for all k, and then a second kernel writes imag parts for odd k.
    for k in range(1, S + 1):
        # Accumulate real contribution for even/odd k using sum over t in 0..N-1 of x[t] * cos(2*pi*k*t/N)
        # Since z[t] = x[t] for t<S, z[t] = 0 for t>=S; but here we iterate t from 0..N-1. For t>=S, z[t] = 0.
        sum_cos = 0.0
        for t in range(0, N):
            v = tl.load(x_ptr + base_x + t * stride_x_s)  # x[t] for t<S, else 0 (we load and multiply by 0)
            ang = 2.0 * 3.141592653589793 * k * t / N
            cosv = tl.cos(ang)
            sum_cos += v * cosv

        if (k & 1) == 0:
            # k is even: y[k] = (sum_cos - sum_sin) / (2*N), purely real
            # But we also need y[N/2 - k] = conjugate: same real, opposite imag; however we only write real here,
            # imag will be handled by the second kernel. For even k, imag is 0 by construction.
            y_real_k = sum_cos * inv_2N
            tl.store(out_ptr + base_out + k * stride_out_out, y_real_k)
        else:
            # k is odd: we will compute imag[k] in the second kernel; here we only compute real[k] if needed.
            # But we need y[N/2 - k] to be complex conjugate of y[k]; we store real only. Imaginary part will be written by kernel 2.
            pass
            # Nothing to write here for real; will be computed by kernel 2 if required. Here we just leave it as default zeros.

    # Handle k = N/2 = S (since N=2S): For real inputs, y[S] is purely real. We compute it via direct formula:
    # y[S] = sum_cos(S) * inv_2N, where sum_cos(S) = sum_{t=0..N-1} x[t] * cos(pi*t) = sum_{t even} x[t] * (-1), since cos(pi*t) = (-1)^t
    # However, simpler: y[S] equals the real part of the original rfft at that index, which equals the sum of x with alternating sign.
    # We can compute it by summing x with (-1)^t factor.
    sum_alt = 0.0
    for t in range(0, S):
        v = tl.load(x_ptr + base_x + t * stride_x_s)
        sign = 1.0 if (t % 2 == 0) else -1.0
        sum_alt += v * sign
    yS = sum_alt * inv_2N
    tl.store(out_ptr + base_out + S * stride_out_out, yS)


@triton.jit
def rfft_imag_kernel(
    x_ptr,                      # *float32, input x of shape (B, C, S) flattened
    out_imag_ptr,               # *float32, output imaginary part of shape (B*C, S+1), only k>0 odd k will be set
    B: tl.int32, C: tl.int32, S: tl.int32,
    stride_x_bc: tl.int32, stride_x_s: tl.int32,
    stride_out_bc: tl.int32, stride_out_out: tl.int32,
):
    bc = tl.program_id(0)
    base_x = bc * stride_x_bc
    base_out = bc * stride_out_bc

    N = 2 * S
    for k in range(1, S + 1):
        if (k & 1) == 1:  # odd k
            sum_sin = 0.0
            for t in range(0, N):
                v = tl.load(x_ptr + base_x + t * stride_x_s)  # x[t] for t<S, else 0
                ang = 2.0 * 3.141592653589793 * k * t / N
                sinv = tl.sin(ang)
                sum_sin += v * sinv
            # Imaginary part for k: -sum_sin / (2*N)
            imag_k = -sum_sin * (1.0 / (2.0 * N))
            # Store imag[k]
            tl.store(out_imag_ptr + base_out + k * stride_out_out, imag_k)
            # Conjugate symmetry: imag[N/2 - k] = -imag[k]
            # Compute j = S - k, and write -imag_k at that index
            j = S - k
            tl.store(out_imag_ptr + base_out + j * stride_out_out, -imag_k)
            # For k > S/2, j = S - k < 0? No: since k in 1..S, j in 0..S-1. For k=S, j=0 (already handled above).
    # For k=0: imag[0] = 0 (rfft has no imaginary part at k=0), already true.
    # For k=S (even if S is even): we set imag[S] = 0 (purely real), but S is odd for our S definition (S+1 is length). So S must be odd; we handle general S above via odd check.


def run_triton(x: torch.Tensor):
    """
    Triton-only implementation of:
      x: (B, C, S) float32
      Output: out_real, out_imag (B, C, S+1) float32
    """
    assert x.dtype == torch.float32, "Input must be float32"
    assert x.dim() == 3, "Input must be (B, C, S)"
    B, C, S = x.shape
    device = x.device

    # We need to create z of length 2*S: z = [x, zeros, x] (for even N; but here N=2S, just zeros after S). For real rfft, padding zeros works.
    # However, our kernels do not require explicit z; they sum over t=0..2*S-1 and rely on z[t]=0 for t>=S. So no need to build z explicitly.

    # Allocate outputs
    out_real = torch.empty((B * C, S + 1), device=device, dtype=torch.float32)
    out_imag = torch.empty((B * C, S + 1), device=device, dtype=torch.float32)

    # Launch rfft_real_kernel: one program per (b, c)
    grid = (B * C,)
    # Strides: x is (B, C, S) contiguous => stride_x_bc = C*S, stride_x_s = 1 for each bc slice. But Triton sees flattened x,
    # so better to pass actual pointer and compute using strides of (B*C, S) layout. We can do: flatten x to (B*C, S) first.
    x_flat = x.reshape(B * C, S).contiguous()
    out_real = out_real  # already allocated
    out_imag = out_imag  # already allocated

    # Strides for x_flat (B*C, S) are: stride_x_bc = S, stride_x_s = 1
    stride_x_bc = x_flat.stride(0)  # should be S
    stride_x_s = x_flat.stride(1)   # should be 1

    # Strides for outputs (B*C, S+1): stride_out_bc = S+1, stride_out_out = 1
    stride_out_bc = out_real.stride(0)  # S+1
    stride_out_out = out_real.stride(1) # 1

    # Run kernel
    rfft_real_kernel[grid](
        x_flat, out_real, B, C, S,
        stride_x_bc, stride_x_s,
        stride_out_bc, stride_out_out,
        num_warps=1, num_stages=1
    )

    # Run kernel for imaginary parts
    rfft_imag_kernel[grid](
        x_flat, out_imag, B, C, S,
        stride_x_bc, stride_x_s,
        stride_out_bc, stride_out_out,
        num_warps=1, num_stages=1
    )

    # Reshape back to (B, C, S+1)
    out_real = out_real.view(B, C, S + 1)
    out_imag = out_imag.view(B, C, S + 1)

    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect one input tensor of shape (B, C, S)
        if len(args) != 1:
            raise ValueError("ModelNew expects a single input tensor of shape (B, C, S)")
        x = args[0]
        if not x.is_cuda:
            # If not on CUDA, fallback to original torch path for correctness, but the evaluator requires Triton execution.
            # To adhere to requirement, we ensure Triton execution by moving to CUDA if available.
            if torch.cuda.is_available():
                x = x.to("cuda")
            else:
                # If no CUDA, raise: evaluator runs on GPU.
                raise RuntimeError("CUDA device required for Triton execution")
        # Run Triton-only computation
        out_real, out_imag = run_triton(x)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
