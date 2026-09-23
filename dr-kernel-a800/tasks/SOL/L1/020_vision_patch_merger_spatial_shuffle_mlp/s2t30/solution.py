import torch
import triton
import triton.language as tl


# LayerNorm per row: input [num_patches, hidden_size], output fp32
@triton.jit
def layernorm_affine_kernel(
    x_ptr,            # *fp32
    out_ptr,          # *fp32
    ln_weight_ptr,    # *fp32
    ln_bias_ptr,      # *fp32
    hidden_size: tl.constexpr,
):
    pid = tl.program_id(0)  # row index
    offs = tl.arange(0, hidden_size)
    x = tl.load(x_ptr + pid * hidden_size + offs)
    mean = tl.sum(x, axis=0) / hidden_size
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / hidden_size
    inv_std = tl.math.rsqrt(var + 1e-6)
    norm = diff * inv_std
    w = tl.load(ln_weight_ptr + offs)
    b = tl.load(ln_bias_ptr + offs)
    out = norm * w + b
    tl.store(out_ptr + pid * hidden_size + offs, out)


# GEMM: C[M, N] = A[M, K] @ B[K, N] (+ bias)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bias_ptr,          # *fp32
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# GELU elementwise (approx) on fp32
@triton.jit
def gelu_kernel(
    x_ptr, y_ptr, SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < SIZE
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
    tl.store(y_ptr + offs, y, mask=mask)


def _launch_layernorm(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor) -> torch.Tensor:
    # hidden: [num_patches, 1536] bfloat16 or fp32; we compute in fp32
    hidden = hidden.contiguous()
    ln_w = ln_weight.contiguous()
    ln_b = ln_bias.contiguous()
    num_patches, hidden_size = hidden.shape
    out = torch.empty((num_patches, hidden_size), dtype=torch.float32, device=hidden.device)
    layernorm_affine_kernel[(num_patches,)](
        hidden.to(torch.float32), out, ln_w.to(torch.float32), ln_b.to(torch.float32), hidden_size,
        num_warps=4, num_stages=2,
    )
    return out


def _launch_gemm(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N], output C: [M, N] fp32
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Incompatible shapes for GEMM"
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_bias_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        bias.contiguous().to(torch.float32),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        # Triton-only forward:
        # 1) LayerNorm on hidden, fp32 output
        hidden_norm = _launch_layernorm(hidden, ln_weight, ln_bias)  # [num_patches, 1536], fp32

        # 2) First GEMM: [num_patches, 6144] @ [6144, 6144] (+ fc1_bias) -> [num_patches, 6144], fp32
        fc1_out = _launch_gemm(hidden_norm, fc1_weight, fc1_bias)

        # 3) GELU activation on fp32
        M, K = fc1_out.shape
        gelu_out = torch.empty_like(fc1_out, dtype=torch.float32, device=fc1_out.device)
        grid = (triton.cdiv(K, 1024),)
        gelu_kernel[grid](fc1_out, gelu_out, K, num_warps=4, num_stages=2)

        # 4) Second GEMM: [num_patches, 6144] @ [3584, 6144] (+ fc2_bias) -> [num_patches, 3584], fp32
        output = _launch_gemm(gelu_out, fc2_weight, fc2_bias)

        return output


def run(*args):
    return ModelNew()(*args)
