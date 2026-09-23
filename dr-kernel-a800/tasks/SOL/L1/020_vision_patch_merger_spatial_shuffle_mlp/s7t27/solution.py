import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_affine_kernel(
    hidden_ptr,      # *bf16, [num_patches, hidden_size], contiguous
    ln_weight_ptr,   # *bf16, [hidden_size]
    ln_bias_ptr,     # *bf16, [hidden_size]
    out_ptr,         # *bf16, [num_patches, hidden_size]
    num_patches: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per row (patch)
    # Accumulate sum and sum of squares over C in fp32
    total_sum = 0.0
    total_sumsq = 0.0
    for c0 in range(0, hidden_size, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        mask = cols < hidden_size
        x = tl.load(hidden_ptr + pid * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)

    mean = total_sum / hidden_size
    var = total_sumsq / hidden_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    for c0 in range(0, hidden_size, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        mask = cols < hidden_size
        x = tl.load(hidden_ptr + pid * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * w + b
        y = y.to(tl.bfloat16)
        tl.store(out_ptr + pid * hidden_size + cols, y, mask=mask)


@triton.jit
def spatial_shuffle_to_fc1_kernel(
    ln_ptr,         # *bf16, [num_patches, hidden_size], contiguous
    X_ptr,          # *bf16, [num_merged_patches, hidden_size_expanded], output of reorder
    grid_thw_ptr,   # *int64, [num_grids, 3] = [T, H, W]
    num_grids,      # int
    num_patches,    # int (sum of all grids)
    hidden_size,    # int
    hidden_expanded, # int
    merge_size,     # int (must be 2)
):
    pid = tl.program_id(0)  # each program handles one grid
    # Read T, H, W for this grid
    T = tl.load(grid_thw_ptr + pid * 3 + 0).to(tl.int32)
    H = tl.load(grid_thw_ptr + pid * 3 + 1).to(tl.int32)
    W = tl.load(grid_thw_ptr + pid * 3 + 2).to(tl.int32)

    # Compute offset of this grid in global num_patches (sum of previous grids)
    total_prev = 0
    for g in range(0, pid):
        t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
        h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
        w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)
        total_prev += t * h * w

    # Merged dims
    Hm = H // merge_size
    Wm = W // merge_size
    num_patches_grid = T * H * W
    num_patches_merged_grid = T * Hm * Wm

    # Precompute total offset for this grid in global output: total_prev + row
    # We will write to X_ptr at row = total_prev + merged index.
    # Iterate over all original patches and features
    # Note: Since Hm and Wm depend on H, W, we use merge_size (2) and assume H, W are multiples of 2.
    for t in range(0, T):
        for i in range(0, H):
            for j in range(0, W):
                row_src = t * H * W + i * W + j
                # Map to merged index
                im = i // merge_size
                jm = j // merge_size
                merged_row = t * Hm * Wm + im * Wm + jm
                global_row = total_prev + merged_row
                # Iterate over features (hidden_size)
                for c in range(0, hidden_size, 64):  # loop over features with BLOCK
                    cols = c + tl.arange(0, 64)
                    feat_mask = cols < hidden_size
                    val = tl.load(ln_ptr + row_src * hidden_size + cols, mask=feat_mask, other=0.0).to(tl.bfloat16)
                    # Store into X_ptr at [global_row, c:c+64]
                    # We need to build addresses for all columns; for each col we store one element.
                    # Triton requires vectorized column indexing; we store per col with a loop:
                    # This pattern is supported via tl.store with per-element masks.
                    # To store vector, we write element by element for cols:
                    for k in range(0, 64):
                        col = c + k
                        if col < hidden_size:
                            tl.store(X_ptr + global_row * hidden_expanded + col, val[k], mask=True)


@triton.jit
def matmul_bias_kernel(
    A_ptr,           # *bf16, [M, K], row-major
    B_ptr,           # *bf16, [K, N], row-major (note: we pass fc1_weight or fc2_weight)
    Bias_ptr,        # *bf16, [N]
    C_ptr,           # *bf16, [M, N], output
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am, stride_ak,  # strides for A: row, col
    stride_bk, stride_bn,  # strides for B: row, col (row stride is K, col stride is N)
    stride_cm, stride_cn,  # strides for C: row, col
    eps: tl.constexpr,     # unused, kept for signature consistency
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first block of A and B
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulate in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

        # Advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store back as bf16 to C
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def gelu_tanh_kernel(
    x_ptr,          # *bf16, [M, N], input
    y_ptr,          # *bf16, [M, N], output
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)

    # We assume M and N are large; for small sizes, this still covers. We load tile, apply GELU, store.
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    # tanh-based GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    u = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(u))

    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)


# --- ModelNew forward uses only Triton kernels ---

class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        hidden: torch.Tensor,             # [num_patches, hidden_size], bf16
        grid_thw: torch.Tensor,           # [num_grids, 3], int64 (T, H, W)
        ln_weight: torch.Tensor,          # [hidden_size], bf16
        ln_bias: torch.Tensor,            # [hidden_size], bf16
        fc1_weight: torch.Tensor,         # [hidden_size_expanded, hidden_size_expanded], bf16
        fc1_bias: torch.Tensor,           # [hidden_size_expanded], bf16
        fc2_weight: torch.Tensor,         # [out_hidden_size, hidden_size_expanded], bf16
        fc2_bias: torch.Tensor,           # [out_hidden_size], bf16
        hidden_size: int,                 # 1536
        hidden_size_expanded: int,        # 6144
        out_hidden_size: int,             # 3584
        merge_size: int = 2,              # spatial merge size
    ):
        # Ensure tensors are on CUDA (Triton requires CUDA). The provided get_inputs creates device tensors.
        assert hidden.is_cuda and grid_thw.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and \
               fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, "All tensors must be on CUDA."

        # 1) LayerNorm (pre-shuffle) on hidden
        num_patches = hidden.shape[0]
        ln_out = torch.empty_like(hidden)  # [num_patches, hidden_size]
        BLOCK_C = 64
        layernorm_affine_kernel[(num_patches,)](
            hidden, ln_weight, ln_bias, ln_out,
            num_patches=num_patches,
            hidden_size=hidden_size,
            eps=self.eps,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        # 2) Spatial 2x2 reorder to produce input for first linear
        X_fc1 = torch.empty((num_patches * hidden_size // merge_size // merge_size, hidden_size_expanded), dtype=torch.bfloat16, device=hidden.device)
        # Launch per-grid programs
        grid = (grid_thw.shape[0],)
        spatial_shuffle_to_fc1_kernel[grid](
            ln_out, X_fc1, grid_thw, grid_thw.shape[0], num_patches, hidden_size, hidden_size_expanded, merge_size,
            num_warps=4,
        )

        # 3) First Linear: X_fc1 @ fc1_weight^T + fc1_bias
        M = X_fc1.shape[0]
        K1 = hidden_size_expanded
        N1 = hidden_size_expanded
        C_fc1 = torch.empty((M, N1), dtype=torch.bfloat16, device=hidden.device)

        # We need B to be [K1, N1] where N1 is in-dim of fc1_weight, but we pass fc1_weight as [N1, K1] and use transpose in kernel via strides:
        # In Triton kernel, we pass B as fc1_weight and interpret it as [K, N] by using stride_bk pointing to fc1_weight[k, n].
        # To avoid host-side .transpose(), we pass fc1_weight as [N, K] logically by setting strides accordingly:
        # For our kernel, B is [K, N] with B[k, n] = fc1_weight[n, k]. We can pass B_ptr as fc1_weight, and set stride_bk = hidden_size_expanded, stride_bn = 1.
        # This way, tl.load(B_ptr + k*stride_bk + n*stride_bn) returns fc1_weight[n, k].
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32

        matmul_bias_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))](
            X_fc1, fc1_weight, fc1_bias, C_fc1,
            M, N1, K1,
            stride_am=hidden_size_expanded, stride_ak=1,     # A is [M, K] row-major: stride_am = N, stride_ak = 1
            stride_bk=hidden_size_expanded, stride_bn=1,     # B is [K, N], we access as fc1_weight[n, k] so stride_bk = N, stride_bn = 1
            stride_cm=hidden_size_expanded, stride_cn=1,     # C is [M, N] row-major
            eps=self.eps,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # 4) GELU activation on C_fc1
        gelu_tanh_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))](
            C_fc1, C_fc1,  # y overwrites x
            M, N1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        # 5) Second Linear: C_fc1 @ fc2_weight^T + fc2_bias
        M2 = M
        K2 = N1  # 6144
        N2 = fc2_weight.shape[0]  # out_hidden_size (3584)
        C_final = torch.empty((M2, N2), dtype=torch.bfloat16, device=hidden.device)

        matmul_bias_kernel[(triton.cdiv(M2, BLOCK_M), triton.cdiv(N2, BLOCK_N))](
            C_fc1, fc2_weight, fc2_bias, C_final,
            M2, N2, K2,
            stride_am=K2, stride_ak=1,                      # A is [M2, K2] row-major: stride_am = K2, stride_ak = 1
            stride_bk=N2, stride_bn=1,                      # B is [K2, N2], access as fc2_weight[n, k]: stride_bk = N2, stride_bn = 1
            stride_cm=N2, stride_cn=1,                      # C is [M2, N2] row-major
            eps=self.eps,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        return C_final


def run(*args):
    return ModelNew()(*args)
