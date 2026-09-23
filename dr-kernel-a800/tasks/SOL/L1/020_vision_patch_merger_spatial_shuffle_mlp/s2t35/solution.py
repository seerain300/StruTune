import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_affine_kernel(
    x_ptr,          # *bf16, input [NUM_PATCHES, HIDDEN_SIZE]
    out_ptr,        # *bf16, output [NUM_PATCHES, HIDDEN_SIZE]
    ln_weight_ptr,  # *bf16, [HIDDEN_SIZE]
    ln_bias_ptr,    # *bf16, [HIDDEN_SIZE]
    hidden_size: tl.constexpr,   # 1536
    NUM_PATCHES: tl.constexpr,   # num_patches
    eps,            # float32 scalar
    BLOCK_SIZE: tl.constexpr,    # 1536
):
    pid = tl.program_id(0)  # one program per row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    # Load x as bf16, cast to fp32 for computation
    x_bf = tl.load(x_ptr + pid * hidden_size + offs, mask=mask, other=0.0)
    x = x_bf.to(tl.float32)

    # Compute mean and variance over the hidden_size dimension
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x / hidden_size
    var = sum_x2 / hidden_size - mean * mean
    inv_std = tl.math.rsqrt(var + eps)

    # Normalize
    norm = (x - mean) * inv_std

    # Load ln_weight and ln_bias as bf16, cast to fp32
    w_bf = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0)
    b_bf = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0)
    w = w_bf.to(tl.float32)
    b = b_bf.to(tl.float32)

    out = norm * w + b
    # Cast back to bf16 for storage
    out_bf = out.to(tl.bfloat16)
    tl.store(out_ptr + pid * hidden_size + offs, out_bf, mask=mask)


@triton.jit
def gemm_bias_kernel(
    A_ptr,          # *fp32, [M, K]
    B_ptr,          # *fp32, [K, N]
    bias_ptr,       # *fp32, [N]
    C_ptr,          # *fp32, [M, N]
    M, N, K,        # sizes (runtime)
    stride_am, stride_ak,  # strides for A: row, col
    stride_bk, stride_bn,  # strides for B: row, col
    stride_cm, stride_cn,  # strides for C: row, col
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    C_ptrs = C_ptr + C_offsets

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)

        A_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        a = tl.load(A_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    # add bias
    bias = tl.load(bias_ptr + rn, mask=(rn < N), other=0.0)
    acc += bias[None, :]

    tl.store(C_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def gelu_kernel(
    X_ptr,          # *fp32, input [M, N]
    Y_ptr,          # *fp32, output [M, N]
    M, N,           # sizes (runtime)
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    X_offsets = rm[:, None] * stride_xm + rn[None, :] * stride_xn
    Y_offsets = rm[:, None] * stride_ym + rn[None, :] * stride_yn

    x = tl.load(X_ptr + X_offsets, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    t = tl.math.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    tl.store(Y_ptr + Y_offsets, y, mask=(rm[:, None] < M) & (rn[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float):
        """
        hidden:      [num_patches, 1536], bfloat16
        grid_thw:    [num_grids, 3], int64 (ignored in Triton path)
        ln_weight:   [1536], bfloat16
        ln_bias:     [1536], bfloat16
        fc1_weight:  [6144, 6144], bfloat16
        fc1_bias:    [6144], bfloat16
        fc2_weight:  [3584, 6144], bfloat16
        fc2_bias:    [3584], bfloat16
        eps:         float
        Returns:     [num_patches, 3584] output (bfloat16), following the original run's final output shape.
        """
        # Triton-only forward: no torch operations
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        assert hidden_size == 1536, "hidden_size must be 1536"
        assert fc1_weight.shape == (6144, 6144), "fc1_weight must be [6144, 6144]"
        assert fc2_weight.shape == (3584, 6144), "fc2_weight must be [3584, 6144]"

        # LayerNorm in Triton: compute in fp32, output in bfloat16
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16)
        grid_ln = (num_patches,)
        layernorm_affine_kernel[grid_ln](
            hidden, hidden_norm, ln_weight, ln_bias,
            hidden_size=1536,
            NUM_PATCHES=num_patches,
            eps=eps,
            BLOCK_SIZE=1536,
            num_warps=4,
        )

        # FC1: hidden_norm [num_patches, 6144] @ fc1_weight [6144, 6144] (+ fc1_bias)
        M1 = num_patches
        K1 = 6144
        N1 = 6144

        # Cast A to fp32 for GEMM
        A = hidden_norm.to(torch.float32)
        # Ensure weights are contiguous fp32
        B = fc1_weight.to(torch.float32).contiguous()
        out1 = torch.empty((M1, N1), dtype=torch.float32, device=hidden.device)

        # Launch GEMM kernel
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64
        grid_fc1 = (triton.cdiv(M1, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        gemm_bias_kernel[grid_fc1](
            A, B, fc1_bias.to(torch.float32), out1,
            M1, N1, K1,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            out1.stride(0), out1.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # GELU activation in Triton
        M2 = M1
        N2 = N1
        gelu_out = torch.empty((M2, N2), dtype=torch.float32, device=hidden.device)
        gelu_kernel[(triton.cdiv(M2, BLOCK_M), triton.cdiv(N2, BLOCK_N))](
            out1, gelu_out,
            M2, N2,
            out1.stride(0), out1.stride(1),
            gelu_out.stride(0), gelu_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # FC2: gelu_out [num_patches, 6144] @ fc2_weight [3584, 6144] (+ fc2_bias)
        K2 = N1  # 6144
        N3 = fc2_weight.shape[0]  # 3584

        C = torch.empty((M2, N3), dtype=torch.float32, device=hidden.device)

        # Launch GEMM kernel
        BLOCK_M2 = 64
        BLOCK_N2 = 128
        BLOCK_K2 = 64
        grid_fc2 = (triton.cdiv(M2, BLOCK_M2), triton.cdiv(N3, BLOCK_N2))
        gemm_bias_kernel[grid_fc2](
            gelu_out, fc2_weight.to(torch.float32).contiguous(), fc2_bias.to(torch.float32),
            C,
            M2, N3, K2,
            gelu_out.stride(0), gelu_out.stride(1),
            fc2_weight.to(torch.float32).contiguous().stride(0), fc2_weight.to(torch.float32).contiguous().stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2,
        )

        # Final output in bfloat16
        final = C.to(torch.bfloat16)

        return final


def run(*args):
    return ModelNew()(*args)
