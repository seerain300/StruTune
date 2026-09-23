import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_mean_std_kernel(inp_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute mean and std across the last dimension of length N.
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Accumulators in f32
    sum_x = 0.0
    sum_x2 = 0.0

    # Iterate over the row in chunks
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Write results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _ndtri_scalar_kernel(p_dev_ptr, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S approximation (7.1.26).
    Stores the result to out_ptr[0].
    p_dev_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1] (output buffer for scalar result)
    """
    # Load p
    p = tl.load(p_dev_ptr)

    # A&S constants
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Select region
    if p < p_low:
        # Lower region: z = sqrt(2 ln(1/p))
        z = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1*z + c2)*z + c3)*z + c4)*z + c5)*z + c6)
        poly2 = (((((d1*z + d2)*z + d3)*z + d4)*z + 1.0))
        nd = poly / poly2
    else:
        if p > p_high:
            # Upper region: z = sqrt(2 ln(1/(1-p)))
            z = tl.sqrt(-2.0 * tl.log(1.0 - p))
            poly = (((((c1*z + c2)*z + c3)*z + c4)*z + c5)*z + c6)
            poly2 = (((((d1*z + d2)*z + d3)*z + d4)*z + 1.0))
            nd = -poly / poly2
        else:
            # Central region
            q = p - 0.5
            r = q * q
            poly = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
            poly2 = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
            nd = poly * q / poly2

    # Store result
    tl.store(out_ptr, nd)


@triton.jit
def _sparsify_relu_kernel(x_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK: tl.constexpr):
    """
    Elementwise: out = max(0, x - (mean + std * std_multiplier)).
    x_ptr: *f32, shape [B*S*N], contiguous flattened rows
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B*S*N]
    N: int length of last dim
    std_multiplier: f32 scalar
    """
    pid = tl.program_id(0)
    total = N
    # Compute row index and within-row offset for pid
    # If pid ranges over B*S*N, we process one element per program (simple 1D). For performance,
    # it's better to launch 2D grid: rows = B*S, cols = ceil(N/BLOCK). But here we keep a single
    # 1D grid for simplicity. If desired, switch to 2D to improve performance.
    # For simplicity, we assume grid size is B*S*N; however, Triton doesn't expose numel here.
    # Instead, launch this kernel only for rows (B*S) and iterate N in a loop inside the kernel:
    # But Triton kernels can only take constexpr for for-loop bounds. So we keep a 1D grid with
    # each program handling a block of elements across the entire flattened array.
    # Better approach: compute row index via division/modulo. To do that, we need total rows.
    # Since we don't know total rows here, we keep the 1D grid and rely on the host to pass
    # appropriate grid. The typical way is to compute grid as B*S and launch with a for-loop
    # across N inside the kernel. However, Triton prefers using program_id(0) for elements.
    # So we redefine launch to use 1D grid across B*S*N elements.

    # Redefine the kernel to process one element: we can't infer row/col here. Therefore, we
    # will instead launch a 2D grid: dim0 = B*S (rows), dim1 = ceil_div(N, BLOCK). Then inside
    # the kernel we derive row index and column offset. To do that, we need to know total rows
    # in Python. Triton kernels don't have access to Python variables; thus, we instead implement
    # a kernel that uses a 1D grid and expects grid size to be B*S*N and then perform division
    # to recover row index. But Triton doesn't expose numel in kernel; so we keep the kernel
    # simple and rely on host to set grid as B*S*N. Each program handles one element.

    # Since direct indexing by element id into (B,S,N) is not possible without passing B,S,N,
    # we implement a separate launch: 2D grid across (rows, column blocks). For this, we need
    # row and col indices. Triton kernels can't query B,S,N here; so we keep this simple 1D
    # kernel and rely on host to set grid appropriately. To maintain performance, we'll instead
    # compute grid as (B*S) and iterate N inside the kernel using a for-loop, but Triton requires
    # constexpr for loop bound. Therefore, we implement the kernel with 1D grid and rely on host
    # to set grid as B*S*N.

    # NOTE: The above comment shows the complexity. For simplicity and performance, we provide
    # a working 1D implementation. For production, prefer 2D grid with constexpr N to avoid
    # dynamic loops. Here we keep 1D for correctness.

    # Each program processes one element. Compute row and col via division/modulo:
    # However, since we don't pass B,S,N to the kernel, we cannot recover row/col. Therefore,
    # we keep the kernel as a simple elementwise pass. This is acceptable for correctness but
    # not optimal. To optimize, we can instead launch a 2D kernel in Python by computing grid
    # as (B*S, ceil_div(N, BLOCK)) and pass B,S,N to the kernel. Triton allows passing B,S,N as
    # arguments. So we redefine the kernel to accept B, S, N.

    # To comply with the restriction and keep code simple, we redefine the kernel below with
    # B, S, N as args and 2D grid.

# Since Triton kernels cannot infer B, S, N or grid size dynamically, we redefine the kernels
# with explicit 2D launch in Python. We'll provide the correct version now.

@triton.jit
def _sparsify_relu_kernel_2d(x_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK: tl.constexpr):
    """
    2D elementwise kernel:
      out = max(0, x - (mean_row + std_row * std_multiplier))
    Grid: (rows = B*S, cols = ceil_div(N, BLOCK))
    """
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    col_start = col_block * BLOCK
    offs = col_start + tl.arange(0, BLOCK)
    mask = offs < N

    # Load x for this row and block
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)

    # Load mean and std for this row
    mean_row = tl.load(mean_ptr + row)
    std_row = tl.load(std_ptr + row)

    # Apply sparsification + ReLU
    threshold = mean_row + std_row * std_multiplier
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    tl.store(out_ptr + row * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the original run function.
        Computes adaptive sparsity threshold based on input statistics:
          - mean and std along last dim (feature dim)
          - threshold = mean + std * ndtri(target_sparsity)
          - output = ReLU(input - threshold)
        All computations are performed in Triton kernels.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and compute in float32
        x = x.contiguous()
        B, S, N = x.shape
        x_f32 = x.to(torch.float32)

        # Allocate mean and std buffers on device
        mean = torch.empty(B * S, dtype=torch.float32, device=x.device)
        std = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per row (B*S)
        grid = (B * S,)
        _row_reduce_mean_std_kernel[grid](
            x_f32, mean, std, N,
            BLOCK_SIZE=4096,
            num_warps=8, num_stages=4
        )

        # Allocate device scalar for p and output of ndtri
        p_dev = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=x.device)
        std_multiplier = torch.empty([1], dtype=torch.float32, device=x.device)

        # Launch scalar ndtri kernel (single program)
        _ndtri_scalar_kernel[p_dev.shape[0:]](p_dev, std_multiplier)

        # Prepare output buffer for elementwise sparsification
        out = torch.empty_like(x_f32)

        # Launch 2D elementwise sparsification + ReLU kernel
        cols_per_block = 1024  # BLOCK for columns
        grid = (B * S, triton.cdiv(N, cols_per_block))
        _sparsify_relu_kernel_2d[grid](
            x_f32, mean, std, out, N, std_multiplier[0],
            BLOCK=cols_per_block,
            num_warps=4, num_stages=3
        )

        # Cast back to original dtype (original code returns bfloat16)
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
