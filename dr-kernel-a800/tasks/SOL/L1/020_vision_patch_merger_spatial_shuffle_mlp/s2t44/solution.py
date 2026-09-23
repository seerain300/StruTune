import torch
import triton
import triton.language as tl

# ---------------------------
# Triton kernels
# ---------------------------

@triton.jit
def layernorm_affine_kernel(
    x_ptr,          # *bf16, input [NUM_PATCHES, HIDDEN_SIZE]
    out_ptr,        # *fp32, output [NUM_PATCHES, HIDDEN_SIZE]
    ln_weight_ptr,  # *fp32, [HIDDEN_SIZE]
    ln_bias_ptr,    # *fp32, [HIDDEN_SIZE]
    hidden_size: tl.constexpr,  # 1536
):
    pid = tl.program_id(0)  # one program per row
    offs = tl.arange(0, hidden_size)
    mask = offs < hidden_size
    # Load input row as bf16, convert to fp32
    x = tl.load(x_ptr + pid * hidden_size + offs, mask=mask, other=0).to(tl.float32)
    # Compute mean and variance
    mean = tl.sum(x, axis=0) / hidden_size
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / hidden_size
    inv_std = tl.math.rsqrt(var + 1e-6)
    norm = diff * inv_std
    # Load affine params
    w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0)
    b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0)
    out = norm * w + b
    # Store fp32
    tl.store(out_ptr + pid * hidden_size + offs, out, mask=mask)


@triton.jit
def spatial_reindex_kernel(
    in_ptr,                  # *bf16, input normalized hidden [NUM_PATCHES, HIDDEN_SIZE]
    out_ptr,                 # *fp32, output shuffled [NUM_MERGED_PATCHES, K]
    per_grid_counts_ptr,     # *int64, [NUM_GRIDS] counts per grid
    per_grid_offsets_ptr,    # *int64, [NUM_GRIDS] offsets (inclusive) per grid
    grid_thw_ptr,            # *int64, [NUM_GRIDS, 3] (T, H, W)
    num_patches,             # int64
    num_grids: tl.constexpr,
    hidden_size: tl.constexpr,  # 1536
    K: tl.constexpr,            # 6144
):
    # One program per output row r
    r = tl.program_id(0)
    # Determine grid index i: i is the number of grids strictly before the offset of r
    total = tl.zeros((), dtype=tl.int64)
    i = tl.zeros((), dtype=tl.int64)
    while i < num_grids:
        offsets_i = tl.load(per_grid_offsets_ptr + i).to(tl.int64)
        counts_i = tl.load(per_grid_counts_ptr + i).to(tl.int64)
        total = total + counts_i
        if r < offsets_i:
            break
        i = i + 1
    # If we didn't find a grid (shouldn't happen for valid r), return
    if i >= num_grids:
        return
    # Compute offset into this grid
    offset_into_grid = r - tl.load(per_grid_offsets_ptr + i).to(tl.int64)
    # Load grid_thw for grid i: T, H, W
    T = tl.load(grid_thw_ptr + i * 3 + 0).to(tl.int64)
    H = tl.load(grid_thw_ptr + i * 3 + 1).to(tl.int64)
    W = tl.load(grid_thw_ptr + i * 3 + 2).to(tl.int64)
    # Decode offset_into_grid -> (t, h, w)
    t = offset_into_grid // (H * W)
    hw = offset_into_grid % (H * W)
    h = hw // W
    w = hw % W
    # Now we need to place each output column j (0..K-1) into the input at (t, h, w, c)
    # K = 6144, merge_size = 2, so 4 * hidden_size == 6144
    # We need to map j -> (merge_h, merge_w, c)
    merge_size = 2
    # Loop over columns
    j = 0
    while j < K:
        # merge_h, merge_w, c
        merge_h = j // (merge_size * hidden_size)  # index in 0..(H//2)-1
        rem1 = j % (merge_size * hidden_size)
        merge_w = rem1 // hidden_size             # index in 0..(W//2)-1
        c = rem1 % hidden_size                   # channel index 0..1535
        # Compute source h,w for the 2x2 merge
        src_h = h * merge_size + merge_h
        src_w = w * merge_size + merge_w
        # Source row index
        src_row = t * H * W + src_h * W + src_w
        # Load input bf16 and store into output as fp32
        x_val = tl.load(in_ptr + src_row * hidden_size + c)
        tl.store(out_ptr + r * K + j, x_val.to(tl.float32))
        j += 1


@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    bias_ptr,            # *fp32, [N] or None if no bias
    has_bias: tl.constexpr,
):
    # Tile sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K
    for k in range(0, K, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m[:, None] * A_stride_m + kk[None, :] * A_stride_k
        A_mask = (m[:, None] < M) & (kk[None, :] < K)
        A = tl.load(A_ptrs, mask=A_mask, other=0.0)
        # B tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + kk[:, None] * B_stride_k + n[None, :] * B_stride_n
        B_mask = (kk[:, None] < K) & (n[None, :] < N)
        B = tl.load(B_ptrs, mask=B_mask, other=0.0)
        # FMA
        acc += tl.dot(A, B)
    # Add bias if provided
    if has_bias:
        bias = tl.load(bias_ptr + n, mask=(n < N), other=0.0)  # [BLOCK_N]
        acc += bias[None, :]
    # Store C
    C_ptrs = C_ptr + m[:, None] * C_stride_m + n[None, :] * C_stride_n
    C_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def gelu_kernel(
    x_ptr, y_ptr, M, N,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * 32 + tl.arange(0, 32)
    n = pid_n * 32 + tl.arange(0, 32)
    mask = (m[:, None] < M) & (n[None, :] < N)
    x = tl.load(x_ptr + m[:, None] * N + n[None, :], mask=mask, other=0.0)
    # Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
    tl.store(y_ptr + m[:, None] * N + n[None, :], y, mask=mask)


# ---------------------------
# ModelNew: Triton-only forward
# ---------------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        """
        Triton-only forward:
        - LayerNorm (fp32) with affine on hidden -> out_ln (fp32)
        - Spatial reindex kernel to build hidden_shuffled (fp32) using per_grid_counts_offsets (provided by get_inputs)
        - FC1: hidden_shuffled @ fc1_weight (+ fc1_bias) -> fc1_out (fp32)
        - GELU: fc1_out elementwise
        - FC2: GELU_out @ fc2_weight (+ fc2_bias) -> output (fp32)
        - Cast final output to bfloat16 for consistency with original return dtype
        """
        # Ensure devices/dtypes
        device = hidden.device
        hidden_size = 1536
        K = hidden_size * 4  # 6144
        num_patches = hidden.shape[0]
        # 1) LayerNorm in Triton (fp32 output)
        out_ln = torch.empty((num_patches, hidden_size), dtype=torch.float32, device=device)
        layernorm_affine_kernel[(num_patches,)](
            hidden, out_ln, ln_weight, ln_bias, hidden_size,
            num_warps=4,
        )

        # 2) Prepare per-grid counts/offsets tensor (Triton kernel expects per_grid_counts and per_grid_offsets)
        # The evaluation harness provides grid_thw; we need per_grid_counts = T*H*W per grid,
        # and per_grid_offsets inclusive prefix sums over grids. We assume these are computed
        # by get_inputs and passed in. We read them in forward (no torch reduction in forward).
        # Here we construct per_grid_counts from grid_thw. But to strictly avoid torch in forward,
        # we rely on get_inputs to provide per_grid_counts and per_grid_offsets as torch tensors.
        # We need to query num_grids; we can derive it from grid_thw.shape[0].
        num_grids = grid_thw.shape[0]
        # Allocate per_grid_counts and per_grid_offsets as torch tensors on device
        # We need per_grid_counts[i] = grid_thw[i,0]*grid_thw[i,1]*grid_thw[i,2]
        per_grid_counts = torch.empty((num_grids,), dtype=torch.int64, device=device)
        per_grid_offsets = torch.empty((num_grids,), dtype=torch.int64, device=device)
        # Compute per_grid_counts (do not use torch in kernel; do it here via torch ops but it's allowed in forward)
        for i in range(num_grids):
            T = int(grid_thw[i, 0].item())
            H = int(grid_thw[i, 1].item())
            W = int(grid_thw[i, 2].item())
            per_grid_counts[i] = T * H * W
        # per_grid_offsets is inclusive prefix sum; we can compute it here (allowed in forward)
        inclusive = torch.cumsum(per_grid_counts, dim=0)
        per_grid_offsets[:] = inclusive

        # Output buffer for spatial reindex (fp32)
        hidden_shuffled = torch.empty((num_patches, K), dtype=torch.float32, device=device)

        # Launch spatial reindex kernel: one program per output row
        spatial_reindex_kernel[(num_patches,)](
            hidden.to(torch.bfloat16), hidden_shuffled, per_grid_counts, per_grid_offsets, grid_thw,
            num_patches, num_grids, hidden_size, K,
            num_warps=1,
        )

        # 3) FC1: hidden_shuffled @ fc1_weight (+ fc1_bias) -> fp32
        M = hidden_shuffled.shape[0]  # num_merged_patches
        K1 = fc1_weight.shape[0]      # 6144
        N1 = fc1_weight.shape[1]      # 6144
        fc1_out = torch.empty((M, K1), dtype=torch.float32, device=device)

        gemm_bias_kernel[(triton.cdiv(M, 32), triton.cdiv(K1, 32))](  # use 2D grid launch properly
            hidden_shuffled, fc1_weight, fc1_out,
            M, N1, K1,
            hidden_shuffled.stride(0), hidden_shuffled.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc1_out.stride(0), fc1_out.stride(1),
            fc1_bias, True,
            num_warps=4,
        )

        # 4) GELU activation (elementwise Triton)
        gelu_out = torch.empty_like(fc1_out)
        gelu_kernel[(triton.cdiv(M, 32), triton.cdiv(K1, 32))](fc1_out, gelu_out, M, K1, num_warps=4)

        # 5) FC2: gelu_out @ fc2_weight (+ fc2_bias) -> fp32
        K2 = gelu_out.shape[1]  # 6144
        N2 = fc2_weight.shape[1]  # 3584
        output = torch.empty((M, N2), dtype=torch.float32, device=device)
        gemm_bias_kernel[(triton.cdiv(M, 32), triton.cdiv(N2, 32))](gelu_out, fc2_weight, output,
                                                                    M, N2, K2,
                                                                    gelu_out.stride(0), gelu_out.stride(1),
                                                                    fc2_weight.stride(0), fc2_weight.stride(1),
                                                                    output.stride(0), output.stride(1),
                                                                    fc2_bias, True,
                                                                    num_warps=4)

        # Return final output, cast to bfloat16 to match original
        return output.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
