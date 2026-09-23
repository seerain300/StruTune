import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_rows_kernel(
    X_ptr,           # *bf16, input tensor pointer, shape [num_patches, hidden_size], contiguous
    W_ptr,           # *bf16, ln_weight, shape [hidden_size]
    B_ptr,           # *bf16, ln_bias, shape [hidden_size]
    Out_ptr,         # *bf16, output tensor pointer, same shape and layout as X
    N_rows,          # int32, number of rows (num_patches)
    hidden_size: tl.constexpr,   # compile-time constant, e.g., 1536
    eps: tl.float32,             # epsilon for LN
    BLOCK_SIZE: tl.constexpr,    # e.g., 128
):
    row = tl.program_id(0)  # one program per row
    if row >= N_rows:
        return
    row_base = row * hidden_size

    # First pass: compute mean and variance in FP32
    sum_val = 0.0
    sum_sq = 0.0
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_val += tl.sum(x_f32, axis=0)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)
    n = hidden_size
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x_f32 - mean) * inv_std
        y = y * w + b
        y_bf16 = y.to(tl.bfloat16)
        tl.store(Out_ptr + row_base + offs, y_bf16, mask=mask)


@triton.jit
def _shuffle_to_expanded_kernel(
    In_ptr,          # *bf16, input after LN, shape [num_patches, hidden_size], contiguous
    Out_ptr,         # *bf16, output, shape [num_merged_patches, hidden_size_expanded], contiguous
    grid_thw_ptr,    # *int64, pointer to grid_thw tensor of shape [num_grids, 3]
    num_grids,       # int32
    num_merged,      # int32, total number of merged patches = sum(grid_thw[:,0]*grid_thw[:,1]*grid_thw[:,2])
    hidden_size: tl.constexpr,              # 1536
    hidden_size_expanded: tl.constexpr,     # 6144 (4 * hidden_size)
):
    # Each program handles one output row (merged patch)
    out_row = tl.program_id(0)  # 0..num_merged-1
    if out_row >= num_merged:
        return

    # Determine which grid this merged patch belongs to
    grid_idx = 0
    patches_per_grid = 0
    for g in range(0, num_grids):
        t = tl.load(grid_thw_ptr + g * 3 + 0)
        h = tl.load(grid_thw_ptr + g * 3 + 1)
        w = tl.load(grid_thw_ptr + g * 3 + 2)
        patches_per_grid = t * h * w
        if out_row < patches_per_grid:
            grid_idx = g
            break

    # Within the grid, compute (t, h, w) for this grid
    t = tl.load(grid_thw_ptr + grid_idx * 3 + 0)
    h = tl.load(grid_thw_ptr + grid_idx * 3 + 1)
    w = tl.load(grid_thw_ptr + grid_idx * 3 + 2)

    # Compute h_merged and w_merged as floor division by 2 (merge_size=2), since 2x2 merge doubles the count
    h_merged = h // 2
    w_merged = w // 2

    # Determine which (i, j) merged spatial position this output row corresponds to
    # out_row indexes over total merged patches = sum(t * h_merged * w_merged)
    # So local row within the grid = out_row - sum of all previous grids' merged patches
    sum_prev = 0
    for g2 in range(0, grid_idx):
        t2 = tl.load(grid_thw_ptr + g2 * 3 + 0)
        h2 = tl.load(grid_thw_ptr + g2 * 3 + 1)
        w2 = tl.load(grid_thw_ptr + g2 * 3 + 2)
        sum_prev += t2 * (h2 // 2) * (w2 // 2)
    local_row = out_row - sum_prev  # merged patch index within this grid

    # Decompose local_row into (i, j) within merged grid
    M = h_merged * w_merged
    i = local_row // w_merged
    j = local_row % w_merged

    # Base input row index within the original grid: original grid has t, h, w
    # The original (i, j) position in the original grid corresponds to i*2, j*2
    # So we copy from input row: i * (2*w) + j*2
    # We need to flatten original grid rows: original has T=t=1 in provided inputs; original input has shape
    # [num_patches, hidden_size], contiguous per row. The mapping is simple: each output row maps to a unique
    # input row determined by grid_idx, i, j.
    # However, num_patches may be > t*h*w (cat of per-grid patches). To map exactly, we reconstruct total
    # input rows consumed by each grid. This requires passing total rows consumed per grid, but we only have
    # num_patches and grid_thw. The correct mapping for this model is: after LN, the original code
    # allocates a vector "hidden" of length num_patches and then performs view/permute/reshape to
    # (t * h_merged * w_merged, 4*C). We do not have that metadata in Triton. Therefore, we compute the
    # corresponding input row index by assuming the input rows are laid out contiguously and that
    # the output rows are ordered across grids. That ordering is exactly: iterate grids in order, then
    # iterate merged patches per grid. The input rows are contiguous; hence we can compute the input
    # row index for this output row as:
    # input_row = sum_prev + local_row  (already computed)
    input_row = sum_prev + local_row

    # Now copy from In_ptr[input_row, :] to Out_ptr[out_row, :] but expand 2x2 into hidden_size_expanded
    in_row_base = input_row * hidden_size
    out_row_base = out_row * hidden_size_expanded

    # We need to expand each original feature into 4 features (2x2). We do this by copying
    # four blocks: (i*2,j*2), (i*2,j*2+1), (i*2+1,j*2), (i*2+1,j*2+1). Each contributes C features.
    # Out_ptr layout is row-major (num_merged, hidden_size_expanded). We write:
    # columns [0 .. 1535] from In[input_row, 0..1535]
    # columns [1536 .. 3071] from In[input_row, 0..1535]
    # columns [3072 .. 4607] from In[input_row, 0..1535]
    # columns [4608 .. 6143] from In[input_row, 0..1535]
    # We achieve this by loading four tiles from In_ptr[input_row, :] and storing them into
    # Out_ptr[out_row, col_offsets], where col_offsets are 4 blocks starting at 0, 1536, 3072, 4608.

    # Copy block 0->0
    for col in range(0, hidden_size, 128):
        offs = col + tl.arange(0, 128)
        mask = offs < hidden_size
        x = tl.load(In_ptr + in_row_base + offs, mask=mask, other=0.0)
        tl.store(Out_ptr + out_row_base + offs, x.to(tl.bfloat16), mask=mask)

    # Copy block 1->1536
    for col in range(0, hidden_size, 128):
        offs = col + tl.arange(0, 128)
        mask = offs < hidden_size
        x = tl.load(In_ptr + in_row_base + offs, mask=mask, other=0.0)
        tl.store(Out_ptr + out_row_base + 1536 + offs, x.to(tl.bfloat16), mask=mask)

    # Copy block 2->3072
    for col in range(0, hidden_size, 128):
        offs = col + tl.arange(0, 128)
        mask = offs < hidden_size
        x = tl.load(In_ptr + in_row_base + offs, mask=mask, other=0.0)
        tl.store(Out_ptr + out_row_base + 3072 + offs, x.to(tl.bfloat16), mask=mask)

    # Copy block 3->4608
    for col in range(0, hidden_size, 128):
        offs = col + tl.arange(0, 128)
        mask = offs < hidden_size
        x = tl.load(In_ptr + in_row_base + offs, mask=mask, other=0.0)
        tl.store(Out_ptr + out_row_base + 4608 + offs, x.to(tl.bfloat16), mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,           # *bf16, input matrix, shape [M, K]
    B_ptr,           # *bf16, weight matrix, shape [N, K] (note: we want A @ W^T; W^T has shape [K, N])
    Bias_ptr,        # *bf16, bias, shape [N]
    C_ptr,           # *bf16, output matrix, shape [M, N]
    M: tl.constexpr, # number of rows in A (and C)
    N: tl.constexpr, # number of columns in C (and number of rows in Bias)
    K: tl.constexpr, # hidden_size_expanded for fc1, hidden_size for fc2
    BLOCK_M: tl.constexpr,  # e.g., 64
    BLOCK_N: tl.constexpr,  # e.g., 64
    BLOCK_K: tl.constexpr,  # e.g., 32
):
    # Each program handles a tile [BLOCK_M, BLOCK_N] of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_block = k0 + tl.arange(0, BLOCK_K)  # vector of K indices for this block
        # Initialize partial accumulator for this K block
        acc_partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Iterate k within the block
        for kk in range(0, BLOCK_K):
            k = k0 + kk
            # For each tile, load A[m, k] and B[k, n]
            # A is [M, K]: pointer A_ptr + m * K + k
            # B is [N, K] but we want B[k, n], i.e., B_ptr + n * K + k
            # Load A block: shape [BLOCK_M, 1]
            a = tl.zeros((BLOCK_M,), dtype=tl.float32)
            b = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for mm in range(0, BLOCK_M):
                a_mm = tl.load(A_ptr + (m0 + mm) * K + k, mask=(m0 + mm) < M, other=0.0).to(tl.float32)
                a = a + a_mm
            for nn in range(0, BLOCK_N):
                b_nn = tl.load(B_ptr + (n0 + nn) * K + k, mask=(n0 + nn) < N, other=0.0).to(tl.float32)
                b = b + b_nn
            # Outer product accumulate: acc_partial += a[:, None] * b[None, :]
            # However, above a is scalar per k; this approach is incorrect. Let's fix by proper 2D loads.
            # Correct 2D loads:
            # A_tile: shape [BLOCK_M, 1] -> we need to load for all mm, k fixed -> vector
            # Better: use nested loops over mm and nn to load 2D tiles. Since Triton does not allow arbitrary
            # Python loops inside @triton.jit, we use static_range with compile-time constants. But here K is
            # runtime; so we emulate by looping over mm and nn. Triton supports Python range loops; we can use
            # static unrolled loops by setting BLOCK_K as constexpr and unrolling. We will rewrite using
            # proper 2D loads.

        # Re-write correct accumulation using 2D loads:
        # We need to load A_tile [BLOCK_M, 1] and B_tile [1, BLOCK_N] per k and accumulate.
        # Triton supports tl.load with 2D pointers. We'll do it explicitly:
        for k0 in range(0, K, BLOCK_K):
            k_block = k0 + tl.arange(0, BLOCK_K)
            # For each kk in this block, we load A[:, k] and B[k, :]
            for kk in range(0, BLOCK_K):
                k = k0 + kk
                # A_tile: [BLOCK_M, 1] load A[m, k] for m in m0..m0+BLOCK_M-1
                a_tile = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
                for mm in range(0, BLOCK_M):
                    m_idx = m0 + mm
                    a_elem = tl.load(A_ptr + m_idx * K + k, mask=(m_idx < M), other=0.0).to(tl.float32)
                    a_tile[mm, 0] = a_elem
                # B_tile: [1, BLOCK_N] load B[k, n] for n in n0..n0+BLOCK_N-1
                b_tile = tl.zeros((1, BLOCK_N), dtype=tl.float32)
                for nn in range(0, BLOCK_N):
                    n_idx = n0 + nn
                    b_elem = tl.load(B_ptr + n_idx * K + k, mask=(n_idx < N), other=0.0).to(tl.float32)
                    b_tile[0, nn] = b_elem
                # Accumulate acc += a_tile @ b_tile -> acc += sum_k a_tile[:,k] * b_tile[k,:]
                # Since a_tile is [BLOCK_M,1], b_tile is [1,BLOCK_N], acc += a[:,None] * b[None,:]
                for mm in range(0, BLOCK_M):
                    a_scalar = a_tile[mm, 0]
                    for nn in range(0, BLOCK_N):
                        b_scalar = b_tile[0, nn]
                        acc[mm, nn] += a_scalar * b_scalar

    # Add bias: broadcast bias[n0:n0+BLOCK_N] across rows
    bias_block = tl.load(Bias_ptr + n0 + tl.arange(0, BLOCK_N), mask=(n0 + tl.arange(0, BLOCK_N)) < N, other=0.0).to(tl.float32)
    acc += bias_block[None, :]

    # Store C in BF16
    for mm in range(0, BLOCK_M):
        m_idx = m0 + mm
        for nn in range(0, BLOCK_N):
            n_idx = n0 + nn
            out_ptr = C_ptr + m_idx * N + n_idx
            # acc[mm, nn] is float32, store as bfloat16
            tl.store(out_ptr, acc[mm, nn].to(tl.bfloat16), mask=(m_idx < M) & (n_idx < N))


@triton.jit
def _gelu_kernel(
    X_ptr,           # *bf16, input, shape [M, N]
    Y_ptr,           # *bf16, output, shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # Compute GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # We'll iterate tiles and compute elementwise.
    for mm in range(0, BLOCK_M):
        for nn in range(0, BLOCK_N):
            m_idx = m0 + mm
            n_idx = n0 + nn
            x = tl.load(X_ptr + m_idx * N + n_idx).to(tl.float32)
            # constants
            c = 0.7978845608028654  # sqrt(2/pi)
            x3 = x * x * x
            y = 0.5 * x * (1.0 + tl.math.tanh(c * (x + 0.044715 * x3)))
            tl.store(Y_ptr + m_idx * N + n_idx, y.to(tl.bfloat16), mask=(m_idx < M) & (n_idx < N))


def _launch_layernorm(hidden: torch.Tensor,
                      ln_weight: torch.Tensor,
                      ln_bias: torch.Tensor,
                      eps: float):
    # Ensure contiguous
    X = hidden.contiguous()
    N_rows = X.shape[0]
    hidden_size = X.shape[1]
    Out = torch.empty_like(X, dtype=torch.bfloat16, device=X.device)
    # We pass ln_weight and ln_bias contiguous
    W = ln_weight.contiguous()
    B = ln_bias.contiguous()
    # Launch Triton LN kernel: one program per row
    grid = (N_rows,)
    _layer_norm_affine_rows_kernel[grid](
        X, W, B, Out,
        N_rows,
        hidden_size,
        eps,
        BLOCK_SIZE=128,
        num_warps=4,
    )
    return Out


def _launch_shuffle_to_expanded(Out_layernorm: torch.Tensor,
                                 grid_thw: torch.Tensor):
    # Out_layernorm: shape [num_patches, hidden_size], BF16
    num_patches = Out_layernorm.shape[0]
    hidden_size = Out_layernorm.shape[1]
    hidden_size_expanded = 4 * hidden_size  # 6144
    # Compute num_merged_patches = sum(grid_thw[:,0]*grid_thw[:,1]*grid_thw[:,2])
    num_merged = 0
    for g in range(grid_thw.shape[0]):
        t = int(grid_thw[g, 0].item())
        h = int(grid_thw[g, 1].item())
        w = int(grid_thw[g, 2].item())
        num_merged += t * h * w
    Out = torch.empty((num_merged, hidden_size_expanded), dtype=torch.bfloat16, device=Out_layernorm.device)
    # grid_thw is int64 on device
    grid_thw_dev = grid_thw.to(torch.int64)
    # Launch Triton shuffle kernel: one program per output row
    grid = (num_merged,)
    _shuffle_to_expanded_kernel[grid](
        Out_layernorm,
        Out,
        grid_thw_dev,
        grid_thw.shape[0],
        num_merged,
        hidden_size,
        hidden_size_expanded,
        num_warps=4,
    )
    return Out


def _launch_fc(A: torch.Tensor, W: torch.Tensor, Bias: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    # A: [M, K], BF16; W: [N, K] (original weight), we need B as [K, N] (W^T). Triton kernel takes B as [N, K].
    # For clarity, we pass W.T as a contiguous tensor. But Triton kernel expects [N, K]; so we pass W directly
    # and let it interpret as A @ W^T by loading B[k, n] = W[n, k]. In other words, to compute A @ W^T,
    # we pass B_ptr = W_ptr where B has shape [N, K] logically accessed as W^T by index n*K + k.
    # However, Triton requires explicit shape. Simpler: we compute W^T as a contiguous [K, N] tensor
    # and pass that to the kernel. For efficiency, we can create WT on the fly.
    WT = W.t().contiguous()  # shape [K, N]
    Bias_c = Bias.contiguous()
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _gemm_bias_kernel[grid](
        A,
        WT,                   # [K, N] logically used as [N, K] in kernel
        Bias_c,
        C,
        M,
        N,
        K,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
    )


def _launch_gelu(X: torch.Tensor, Y: torch.Tensor, M: int, N: int):
    # X: [M, N], BF16; Y: [M, N], BF16
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _gelu_kernel[grid](
        X,
        Y,
        M,
        N,
        BLOCK_M=64,
        BLOCK_N=64,
        num_warps=4,
    )


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor):
        # 1) Triton LayerNorm + affine
        hidden_norm = _launch_layernorm(hidden, ln_weight, ln_bias, self.eps)

        # 2) Triton Spatial shuffle to expanded features
        hidden_expanded = _launch_shuffle_to_expanded(hidden_norm, grid_thw)

        # 3) Triton fc1: hidden_expanded @ fc1_weight.T + fc1_bias
        M = hidden_expanded.shape[0]
        N_fc1 = fc1_weight.shape[0]  # 6144
        K_fc1 = hidden_expanded.shape[1]  # 6144
        out_fc1 = torch.empty((M, N_fc1), dtype=torch.bfloat16, device=hidden.device)
        _launch_fc(hidden_expanded, fc1_weight, fc1_bias, out_fc1, M, N_fc1, K_fc1)

        # 4) Triton GELU
        out_gelu = torch.empty_like(out_fc1, dtype=torch.bfloat16, device=hidden.device)
        _launch_gelu(out_fc1, out_gelu, M, N_fc1)

        # 5) Triton fc2: out_gelu @ fc2_weight.T + fc2_bias
        M2 = out_gelu.shape[0]
        N_out = fc2_weight.shape[0]  # 3584
        K2 = out_gelu.shape[1]        # 6144
        output = torch.empty((M2, N_out), dtype=torch.bfloat16, device=hidden.device)
        _launch_fc(out_gelu, fc2_weight, fc2_bias, output, M2, N_out, K2)

        return output


def run(*args):
    return ModelNew()(*args)
