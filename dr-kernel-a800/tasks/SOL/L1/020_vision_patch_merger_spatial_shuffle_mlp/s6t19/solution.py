import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_affine_kernel(
    x_ptr,        # *const bfloat16
    weight_ptr,   # *const bfloat16 (length K)
    bias_ptr,     # *const bfloat16 (length K)
    y_ptr,        # *bfloat16
    M: tl.constexpr,  # number of rows (num_patches)
    K: tl.constexpr,  # hidden size (1536)
    eps,                     # float32 epsilon
    BLOCK_K: tl.constexpr,   # tile size for K dimension
):
    # one program per row
    row = tl.program_id(0)
    if row >= M:
        return

    # compute mean in fp32
    mean = tl.zeros((), dtype=tl.float32)
    col = 0
    while col < K:
        offs = col + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        mean += tl.sum(x, axis=0)
        col += BLOCK_K
    mean = mean / K

    # compute variance in fp32
    var = tl.zeros((), dtype=tl.float32)
    col = 0
    while col < K:
        offs = col + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        var += tl.sum((x - mean) * (x - mean), axis=0)
        col += BLOCK_K
    var = var / K

    # normalize and apply affine, store as bf16
    col = 0
    while col < K:
        offs = col + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = ((x - mean) / tl.sqrt(var + eps)) * w + b
        y = y.to(tl.bfloat16)
        tl.store(y_ptr + row * K + offs, y, mask=mask)
        col += BLOCK_K


@triton.jit
def gemm_bias_kernel(
    A_ptr,       # *const bfloat16, shape (M, K)
    B_ptr,       # *const bfloat16, shape (N, K) -- note: we pass weight with shape (N, K) but index as (K, N) by swapping
    bias_ptr,    # *const bfloat16, shape (N)
    C_ptr,       # *bfloat16, shape (M, N)
    M: tl.constexpr,   # rows of A
    N: tl.constexpr,   # cols of C / size of bias
    K: tl.constexpr,   # inner dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        # A_tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + rm[:, None] * K + rk[None, :]
        mask_a = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)
        # B_tile: [BLOCK_K, BLOCK_N] where B[k, n] = weight[n, k]
        b_ptrs = B_ptr + rn[None, :] * K + rk[:, None]
        mask_b = (rn[None, :] < N) & (rk[:, None] < K)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)
        # dot: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, b)
    # add bias
    bias = tl.load(bias_ptr + rn, mask=(rn < N), other=0.0).to(tl.float32)  # [BLOCK_N]
    acc += bias[None, :]
    # store as bf16
    c_ptrs = C_ptr + rm[:, None] * N + rn[None, :]
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)


@triton.jit
def gelu_tanh_kernel(
    X_ptr,       # *const bfloat16, shape (M, N)
    Y_ptr,       # *bfloat16, shape (M, N)
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # one program per row
    row = tl.program_id(0)
    if row >= M:
        return
    col = 0
    while col < N:
        offs = col + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        # tanh approximation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c0 = 0.044715
        c1 = 0.7978845608028654  # sqrt(2/pi)
        x3 = x * x * x
        t = c1 * (x + c0 * x3)
        y = 0.5 * x * (1.0 + tl.tanh(t))
        tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_N


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden: torch.Tensor,
        grid_thw: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,      # shape (N, K) where N=6144, K=4*1536
        fc1_bias: torch.Tensor,        # shape (N,)
        fc2_weight: torch.Tensor,      # shape (out_N, N) where out_N=3584, N=4*1536
        fc2_bias: torch.Tensor,        # shape (out_N,)
        eps: float,
    ):
        """
        Triton-only implementation of the original run:
        1) LayerNorm over last dim on 'hidden', affine
        2) Pack: view normalized to (num_patches//4, 4*1536)
        3) fc1: GEMM + bias
        4) GELU
        5) fc2: GEMM + bias
        """
        assert hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, "All tensors must be on CUDA for Triton."

        # 1) LayerNorm + affine
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536
        # We will perform LN in FP32 and store as BF16
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)

        # Launch Triton kernel: one program per row
        grid_ln = (num_patches,)
        layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=num_patches, K=hidden_size, eps=float(eps),
            BLOCK_K=1024, num_warps=4, num_stages=2
        )

        # 2) Pack: view normalized tensor to (num_patches//4, 4*hidden_size)
        M_out = num_patches // 4  # guaranteed by get_inputs
        K_packed = hidden_size * 4  # 4*1536 = 6144
        packed = ln_out.view(M_out, K_packed)  # no data movement, just metadata change

        # 3) fc1: (M_out, K_packed) @ (K_packed, K_packed)^T + bias
        K1 = K_packed  # 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((M_out, N1), dtype=torch.bfloat16, device=hidden.device)

        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 64, 64, 32
        grid_fc1 = (triton.cdiv(M_out, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        # Note: fc1_weight is (N1, K1). In kernel, we index as B[k, n] = weight[n, k].
        gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M=M_out, N=N1, K=K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation
        K_after = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=hidden.device)
        BLOCK_N_gelu = 256
        grid_gelu = (M_out, triton.cdiv(K_after, BLOCK_N_gelu))
        gelu_tanh_kernel[grid_gelu](
            fc1_after_gelu, fc1_out,
            M=M_out, N=K_after,
            BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=1
        )

        # 5) fc2: (M_out, 6144) @ (3584, 6144)^T + bias
        out_N = fc2_weight.shape[0]  # 3584
        fc2_out = torch.empty((M_out, out_N), dtype=torch.bfloat16, device=hidden.device)

        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 64, 64, 32
        grid_fc2 = (triton.cdiv(M_out, BLOCK_M2), triton.cdiv(out_N, BLOCK_N2))
        # fc2_weight is (out_N, K1). Index as B[k, n] = weight[n, k].
        gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, fc2_out,
            M=M_out, N=out_N, K=K1,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return fc2_out


def run(*args):
    return ModelNew()(*args)
