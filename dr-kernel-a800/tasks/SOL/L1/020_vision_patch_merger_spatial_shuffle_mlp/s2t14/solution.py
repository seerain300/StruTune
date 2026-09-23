import math
import triton
import triton.language as tl

# Kernel 1: LayerNorm per-row with affine ln_weight, ln_bias.
# X: input [rows, hidden], bf16; LN_W, LN_B: [hidden], bf16; Y: output [rows, hidden], bf16.
@triton.jit
def layernorm_affine_kernel(X_ptr, Y_ptr, LN_W_ptr, LN_B_ptr,
                             rows, hidden, eps,
                             X_stride_row, X_stride_col,
                             Y_stride_row, Y_stride_col,
                             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    sum_ = 0.0
    sumsq_ = 0.0
    # compute sum and sum of squares
    for k in range(0, hidden, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = (row < rows) & (offs < hidden)
        x = tl.load(X_ptr + row * X_stride_row + offs * X_stride_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sumsq_ += tl.sum(x * x, axis=0)
    mean = sum_ / hidden
    var = sumsq_ / hidden - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    # normalize and apply affine
    for k in range(0, hidden, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = (row < rows) & (offs < hidden)
        x = tl.load(X_ptr + row * X_stride_row + offs * X_stride_col, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(LN_W_ptr + offs, mask=offs < hidden, other=1.0).to(tl.float32)
        b = tl.load(LN_B_ptr + offs, mask=offs < hidden, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(Y_ptr + row * Y_stride_row + offs * Y_stride_col, y.to(tl.bfloat16), mask=mask)

# Kernel 2: Spatial shuffle: write shuffled patches from normalized hidden into Y [M, hidden_expanded].
# Inputs:
# - Xn: normalized hidden [num_patches, hidden], bf16
# - grid_thw: [num_grids, 3], int64 (T, H, W per grid)
# - total_per_grid: [num_grids], int32
# - offsets: [num_grids], int32 cumulative offsets into the flattened grid
# - M: num_merged_patches, int32
# - hidden_expanded: 6144, int32
# - merge_size: 2 (compile-time constant, passed as tl.constexpr)
# - Y: output [M, hidden_expanded], bf16
@triton.jit
def spatial_shuffle_kernel(Xn_ptr, grid_thw_ptr, total_per_grid_ptr, offsets_ptr, Y_ptr,
                            M, hidden, hidden_expanded,
                            Xn_stride_row, Xn_stride_col,
                            Y_stride_row, Y_stride_col,
                            merge_size: tl.constexpr):
    r = tl.program_id(0)  # row index in output, 0..M-1
    j = tl.program_id(1)  # column index in output, 0..hidden_expanded-1

    # Decode j into (merge_h, merge_w, c)
    # For 2x2 merge: each merged spatial position corresponds to 4*C elements (C=1536), so hidden_expanded = 4*1536 = 6144
    c = j % 1536
    inner = j // 1536  # belongs to [0, 4)
    merge_h = inner // 2
    merge_w = inner % 2

    # Determine which grid contains row r and compute (t, h, w)
    # r is in [0, sum(total_per_grid)), with offsets[i] as boundaries
    # Binary search on offsets to find the grid id gid for row r.
    lo = 0
    hi = tl.load(total_per_grid_ptr + 0)  # start with first grid size (dummy, not used in loop)
    gid = 0
    while lo < hi:
        mid = (lo + hi) // 2
        offs_mid = tl.load(offsets_ptr + mid)
        if r < offs_mid:
            hi = mid
        else:
            lo = mid + 1
    gid = lo  # gid is the grid index containing row r

    # Compute (t, h, w) for gid
    # We need T, H, W for gid. Read from grid_thw[gid, :]
    T = tl.load(grid_thw_ptr + gid * 3 + 0)
    H = tl.load(grid_thw_ptr + gid * 3 + 1)
    W = tl.load(grid_thw_ptr + gid * 3 + 2)

    # Number of merged positions in this grid: THW_merged = T * (H // merge_size) * (W // merge_size)
    THW_merged = T * (H // merge_size) * (W // merge_size)

    # Determine the base index in normalized hidden for this grid
    # total so far is sum of all previous grids' sizes; we compute it with offsets
    # total_prev = offsets[gid - 1] if gid > 0 else 0
    total_prev = 0
    if gid > 0:
        total_prev = tl.load(offsets_ptr + (gid - 1))
    # Row r within this grid
    r_in_grid = r - total_prev  # r_in_grid in [0, total_per_grid[gid]-1]

    t = r_in_grid // ( (H // merge_size) * (W // merge_size) )
    rem = r_in_grid % ( (H // merge_size) * (W // merge_size) )
    h = rem // (W // merge_size)
    w = rem % (W // merge_size)

    # Base row in normalized hidden: (gid * THW_merged + t * (H//merge_size) * (W//merge_size) + h * (W//merge_size) + w) * hidden
    THMER = (H // merge_size) * (W // merge_size)
    base_row = gid * THW_merged + t * THMER + h * (W // merge_size) + w
    src_row = base_row * hidden

    # Column inside patch: c
    src_col = c

    # Load from normalized hidden and store to Y[r, j]
    x = tl.load(Xn_ptr + src_row + src_col * Xn_stride_col, mask=True, other=0.0).to(tl.float32)
    # Store as bf16
    tl.store(Y_ptr + r * Y_stride_row + j * Y_stride_col, x.to(tl.bfloat16), mask=True)

# Kernel 3: Matmul with bias: A[M, K] @ B[K, N] + Bias[N] -> C[M, N]
# A_ptr: [M, K], bf16; B_ptr: [K, N], bf16; Bias_ptr: [N], bf16; C_ptr: [M, N], bf16
@triton.jit
def matmul_bias_kernel(A_ptr, B_ptr, Bias_ptr, C_ptr,
                        M, N, K,
                        A_stride_m, A_stride_k,
                        B_stride_k, B_stride_n,
                        C_stride_m, C_stride_n,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * A_stride_m + offs_k[None, :] * A_stride_k
        b_ptrs = B_ptr + offs_k[:, None] * B_stride_k + offs_n[None, :] * B_stride_n
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)

# Kernel 4: Elementwise GELU (tanh approximation) on X[M] -> Y[M], bf16 in/out
@triton.jit
def gelu_kernel(X_ptr, Y_ptr, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + c * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(Y_ptr + offs, y.to(tl.bfloat16), mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.hidden_size = 1536
        self.hidden_expanded = 6144
        self.out_hidden_size = 3584
        self.merge_size = 2
        self.eps = 1e-6

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor):
        """
        hidden: [num_patches, hidden_size], bfloat16, device
        grid_thw: [num_grids, 3], int64 (T,H,W per grid)
        ln_weight, ln_bias: [hidden_size], bfloat16
        fc1_weight: [hidden_expanded, hidden_expanded], bfloat16
        fc1_bias: [hidden_expanded], bfloat16
        fc2_weight: [out_hidden_size, hidden_expanded], bfloat16
        fc2_bias: [out_hidden_size], bfloat16
        """
        assert hidden.is_cuda and grid_thw.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and \
               fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, \
            "All tensors must be on CUDA for Triton."

        device = hidden.device
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()
        grid_thw = grid_thw.contiguous()

        rows = hidden.shape[0]
        hidden_size = hidden.shape[1]

        # 1) LayerNorm + affine on hidden -> Hn [rows, hidden_size], bf16
        Hn = torch.empty((rows, hidden_size), dtype=torch.bfloat16, device=device)

        # Strides
        Xn_stride_row = hidden_size
        Xn_stride_col = 1
        Yn_stride_row = hidden_size
        Yn_stride_col = 1

        # Launch LayerNorm kernel: grid over rows
        grid_ln = (rows,)
        layernorm_affine_kernel[grid_ln](
            hidden, Hn, ln_weight, ln_bias,
            rows, hidden_size, self.eps,
            Xn_stride_row, Xn_stride_col,
            Yn_stride_row, Yn_stride_col,
            BLOCK=1024
        )

        # 2) Compute per-grid counts (pure Python, no torch for computation)
        num_grids = grid_thw.shape[0]
        T = [int(grid_thw[i, 0].item()) for i in range(num_grids)]
        H = [int(grid_thw[i, 1].item()) for i in range(num_grids)]
        W = [int(grid_thw[i, 2].item()) for i in range(num_grids)]
        total_per_grid = [(T[i] * (H[i] // self.merge_size) * (W[i] // self.merge_size)) for i in range(num_grids)]
        total_per_grid = torch.tensor(total_per_grid, dtype=torch.int32, device=device)
        # offsets into flattened grid: cumulative sum of total_per_grid
        offsets = torch.zeros(num_grids, dtype=torch.int32, device=device)
        running = 0
        for i in range(num_grids):
            offsets[i] = running
            running += int(total_per_grid[i].item())
        M = int(offsets[-1].item()) if num_grids > 0 else 0

        # 3) Spatial shuffle: Hn [rows, hidden], produce Y [M, hidden_expanded], bf16
        Y = torch.empty((M, self.hidden_expanded), dtype=torch.bfloat16, device=device)

        # Strides for Y
        Y_stride_row = self.hidden_expanded
        Y_stride_col = 1

        grid_shuffle = (M, self.hidden_expanded)
        spatial_shuffle_kernel[grid_shuffle](
            Hn, grid_thw, total_per_grid, offsets, Y,
            M, hidden_size, self.hidden_expanded,
            Xn_stride_row, Xn_stride_col,
            Y_stride_row, Y_stride_col,
            merge_size=self.merge_size
        )

        # 4) fc1: Y[M, 6144] @ fc1_weight.T[6144, 6144] + fc1_bias[6144] -> fc1_out[M, 6144], bf16
        A = Y.contiguous()  # M x K
        B = fc1_weight.transpose(0, 1).contiguous()  # K x K
        Bias = fc1_bias.contiguous()
        fc1_out = torch.empty((M, self.hidden_expanded), dtype=torch.bfloat16, device=device)

        A_stride_m = self.hidden_expanded
        A_stride_k = 1
        B_stride_k = self.hidden_expanded
        B_stride_n = 1
        C_stride_m = self.hidden_expanded
        C_stride_n = 1

        grid_matmul = (triton.cdiv(M, 64), triton.cdiv(self.hidden_expanded, 64))
        matmul_bias_kernel[grid_matmul](
            A, B, Bias, fc1_out,
            M, self.hidden_expanded, self.hidden_expanded,
            A_stride_m, A_stride_k,
            B_stride_k, B_stride_n,
            C_stride_m, C_stride_n,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4
        )

        # 5) GELU activation: fc1_out -> fc1_out_gelu
        fc1_out_gelu = torch.empty_like(fc1_out)
        grid_gelu = (triton.cdiv(M * self.hidden_expanded, 1024),)
        gelu_kernel[grid_gelu](fc1_out, fc1_out_gelu, M * self.hidden_expanded, BLOCK=1024)

        # 6) fc2: fc1_out_gelu[M, 6144] @ fc2_weight.T[6144, 3584] + fc2_bias[3584] -> output[M, 3584], bf16
        B2 = fc2_weight.transpose(0, 1).contiguous()  # K x N
        Bias2 = fc2_bias.contiguous()
        output = torch.empty((M, self.out_hidden_size), dtype=torch.bfloat16, device=device)

        A2_stride_m = self.hidden_expanded
        A2_stride_k = 1
        B2_stride_k = self.hidden_expanded
        B2_stride_n = self.out_hidden_size
        C2_stride_m = self.out_hidden_size
        C2_stride_n = 1

        grid_matmul2 = (triton.cdiv(M, 64), triton.cdiv(self.out_hidden_size, 64))
        matmul_bias_kernel[grid_matmul2](
            fc1_out_gelu, B2, Bias2, output,
            M, self.out_hidden_size, self.hidden_expanded,
            A2_stride_m, A2_stride_k,
            B2_stride_k, B2_stride_n,
            C2_stride_m, C2_stride_n,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4
        )

        return output


def run(*args):
    return ModelNew()(*args)
