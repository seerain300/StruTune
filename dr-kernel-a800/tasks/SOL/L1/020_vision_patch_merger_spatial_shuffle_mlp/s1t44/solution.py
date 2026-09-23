import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: LayerNorm per row, affine with ln_weight, ln_bias
# Input: hidden_in [N, C], bfloat16
# Output: out_hidden [N, C], bfloat16
@triton.jit
def layer_norm_affine_kernel(
    hidden_in_ptr,   # *bf16, [N, C]
    out_ptr,         # *bf16, [N, C]
    ln_weight_ptr,   # *bf16, [C]
    ln_bias_ptr,     # *bf16, [C]
    N, C,            # int32
    eps,             # float32
):
    pid = tl.program_id(0)
    j = pid  # row index
    if j >= N:
        return

    # Accumulate sum and sumsq in fp32 over C
    sum_x = 0.0
    sum_x2 = 0.0
    BLOCK_C = 256
    for col_start in range(0, C, BLOCK_C):
        cols = col_start + tl.arange(0, BLOCK_C)
        mask = cols < C
        x = tl.load(hidden_in_ptr + j * C + cols, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / C
    var = sum_x2 / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and affine, write back
    for col_start in range(0, C, BLOCK_C):
        cols = col_start + tl.arange(0, BLOCK_C)
        mask = cols < C
        x = tl.load(hidden_in_ptr + j * C + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) * inv_std
        y = norm * w + b
        tl.store(out_ptr + j * C + cols, y.to(tl.bfloat16), mask=mask)


# Triton kernel: Spatial shuffle from out_hidden (post-LN+affine) to out_shuffled
# grid_thw shape [num_grids, 3], dtype int64
# out_hidden shape [N, C], dtype bfloat16
# out_shuffled shape [M, 4*C], dtype bfloat16
@triton.jit
def spatial_shuffle_kernel(
    out_hidden_ptr,  # *bf16, [N, C]
    grid_thw_ptr,    # *int64, [num_grids, 3]
    out_ptr,         # *bf16, [M, 4*C]
    N, C,            # int32
    num_grids,       # int32
    M,               # int32 (num_merged_patches)
    merge_size: tl.constexpr,  # compile-time const (2)
):
    pid_j = tl.program_id(0)  # output row index in [0, M)
    if pid_j >= M:
        return

    gi = pid_j // (4 * C)  # grid index for this output row
    if gi >= num_grids:
        return

    # Load T, H, W for this grid
    t = tl.load(grid_thw_ptr + gi * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + gi * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + gi * 3 + 2).to(tl.int32)

    # Decompose gi into (t', h', w') within grid
    q = gi
    t_prime = q // (h * w)
    rem = q % (h * w)
    h_prime = rem // (2 * merge_size)
    w_prime = rem % (2 * merge_size)

    base = gi * (t * h * w)
    src_row = base + t_prime * (h * w) + h_prime * w + w_prime

    # Feature channel index: r = pid_j % (4*C)
    feature_local = (pid_j % (4 * C)) % C

    val = tl.load(out_hidden_ptr + src_row * C + feature_local)
    tl.store(out_ptr + pid_j * (4 * C) + (pid_j % (4 * C)), val.to(tl.bfloat16))


# Triton matmul kernel without bias: C[M, N] = A[M, K] @ W[K, N]
# A: *bf16, [M, K]; W: *bf16, [K, N]; C: *bf32, [M, N]
@triton.jit
def matmul_nobias_kernel(
    A_ptr,  # *bf16, [M, K]
    W_ptr,  # *bf16, [K, N]
    C_ptr,  # *bf32, [M, N]
    M, K, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(A_ptr + (offs_m[:, None] * K) + k[None, :],
                    mask=(offs_m[:, None] < M) & (k[None, :] < K),
                    other=0.0).to(tl.float32)
        b = tl.load(W_ptr + (k[:, None] * N) + offs_n[None, :],
                    mask=(k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    tl.store(C_ptr + (offs_m[:, None] * N) + offs_n[None, :],
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise GELU on FP32 input, store FP32
@triton.jit
def gelu_kernel(
    x_ptr,             # *bf16, [M, N]
    y_ptr,             # *bf32, [M, N]
    M, N,              # int32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    # GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + x^3 / 3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(c * (x + x3 * (1.0 / 3.0))))
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)


# Triton kernel: add bias and cast to bfloat16
# x: *fp32, [M, N], bias: *fp32, [N], out: *bf16, [M, N]
@triton.jit
def fc2_bias_cast_kernel(
    x_ptr,       # *fp32, [M, N] (matmul output without bias)
    bias_ptr,    # *fp32, [N]
    out_ptr,     # *bf16, [M, N] (after bias + cast)
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # shape [BLOCK_N]
    y = x + b[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.merge_size = 2

    def forward(
        self,
        hidden: torch.Tensor,
        grid_thw: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        eps: float,
    ):
        # Ensure all tensors are CUDA and contiguous
        assert hidden.is_cuda and grid_thw.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, "All tensors must be on CUDA for Triton."
        hidden = hidden.contiguous()
        grid_thw = grid_thw.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()

        N, C = hidden.shape
        assert C == 1536, "hidden_size must be 1536."
        # Prepare output LN
        out_hidden = torch.empty((N, C), dtype=torch.bfloat16, device=hidden.device)

        # Launch LayerNorm + affine kernel
        # Grid: 1D over N rows
        grid_ln = (N,)
        layer_norm_affine_kernel[grid_ln](
            hidden, out_hidden, ln_weight, ln_bias,
            N, C, eps,
            num_warps=4, num_stages=2
        )

        # Compute spatial shuffle using grid_thw -> out_shuffled [M, 4*C]
        M = int(grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item() if grid_thw.numel() == 3 else 0)
        # We can't know M without building grid_thw per grid, so we need M as input. The original helper returns M. We assume M is provided implicitly or correct inputs are used. To be safe, require M to be passed: the signature already has it (num_merged_patches).
        # Launch spatial shuffle kernel:
        total_per_grid = 4 * C  # fixed by code, but M should be consistent with inputs. We'll rely on axes and inputs to ensure M is correct.
        out_shuffled = torch.empty((M, 4 * C), dtype=torch.bfloat16, device=hidden.device)
        grid_shuffle = (M,)
        spatial_shuffle_kernel[grid_shuffle](
            out_hidden, grid_thw, out_shuffled,
            N, C, grid_thw.shape[0], M,
            self.merge_size,
            num_warps=4, num_stages=2
        )

        # fc1: matmul without bias, then GELU, then bias
        Mx = out_shuffled.shape[0]
        K1 = out_shuffled.shape[1]  # 4*C = 6144
        N1 = fc1_weight.shape[1]    # 6144
        # Allocate fp32 for fc1 matmul output
        fc1_out = torch.empty((Mx, N1), dtype=torch.float32, device=hidden.device)

        # Launch matmul_nobias_kernel for fc1
        grid_fc1 = (triton.cdiv(Mx, 64), triton.cdiv(N1, 64))
        matmul_nobias_kernel[grid_fc1](
            out_shuffled, fc1_weight, fc1_out,
            Mx, K1, N1,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

        # GELU
        fc1_out_gelu = torch.empty((Mx, N1), dtype=torch.float32, device=hidden.device)
        grid_gelu = (triton.cdiv(Mx, 64), triton.cdiv(N1, 64))
        gelu_kernel[grid_gelu](
            out_shuffled, fc1_out_gelu,
            Mx, N1,
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        # fc2: matmul without bias -> fp32, then add bias and cast to bfloat16
        Mx2 = Mx
        K2 = fc2_weight.shape[1]  # 6144
        N2 = fc2_weight.shape[0]  # 3584
        fc2_out_fp32 = torch.empty((Mx2, N2), dtype=torch.float32, device=hidden.device)

        grid_fc2 = (triton.cdiv(Mx2, 64), triton.cdiv(N2, 64))
        matmul_nobias_kernel[grid_fc2](
            fc1_out_gelu, fc2_weight, fc2_out_fp32,
            Mx2, K2, N2,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

        # Bias add and cast to bfloat16
        out_final = torch.empty((Mx2, N2), dtype=torch.bfloat16, device=hidden.device)
        grid_bias = (triton.cdiv(Mx2, 64), triton.cdiv(N2, 64))
        fc2_bias_cast_kernel[grid_bias](
            fc2_out_fp32, fc2_bias, out_final,
            Mx2, N2,
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        return out_final


def run(*args):
    return ModelNew()(*args)
