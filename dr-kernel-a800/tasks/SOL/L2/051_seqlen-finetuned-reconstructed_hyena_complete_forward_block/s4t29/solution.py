import torch
import triton
import triton.language as tl


# 1) Triton LayerNorm (affine) over last dim for a [M, N] tensor (M = B*S, N = D).
# Kernel 1: compute per-row mean and variance.
@triton.jit
def _layernorm_mean_var_kernel(
    X_ptr,             # *fp32, input [M, N]
    SUM_ptr,           # *fp32, per-row sum [M]
    SUMSQ_ptr,         # *fp32, per-row sum of squares [M]
    M, N,
    stride_xm, stride_xn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return
    acc = 0.0
    acc_sq = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + pid * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
        acc_sq += tl.sum(x * x, axis=0)
    tl.store(SUM_ptr + pid, acc)
    tl.store(SUMSQ_ptr + pid, acc_sq)


# Kernel 2: normalize using computed sum and sumsq, apply affine weight and bias.
@triton.jit
def _layernorm_norm_affine_kernel(
    X_ptr,             # *fp32, input [M, N]
    WEIGHT_ptr,        # *fp32, affine weight [N]
    BIAS_ptr,          # *fp32, affine bias [N]
    Y_ptr,             # *fp32, output [M, N]
    M, N,
    eps,               # fp32
    stride_xm, stride_xn,
    stride_w, stride_b,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return
    mean = tl.load(SUM_ptr + pid) / N
    var = tl.load(SUMSQ_ptr + pid) / N - mean * mean
    inv_std = tl.math.rsqrt(var + eps)
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + pid * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        w = tl.load(WEIGHT_ptr + offs * stride_w, mask=mask, other=1.0)
        b = tl.load(BIAS_ptr + offs * stride_b, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(Y_ptr + pid * stride_ym + offs * stride_yn, y, mask=mask)


# 2) Elementwise 2D add: Y[M, N] = A[M, N] + B[M, N]
@triton.jit
def _add_2d_kernel(
    A_ptr, B_ptr, Y_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        a = tl.load(A_ptr + pid * stride_am + offs * stride_an, mask=mask, other=0.0)
        b = tl.load(B_ptr + pid * stride_bm + offs * stride_bn, mask=mask, other=0.0)
        y = a + b
        tl.store(Y_ptr + pid * stride_ym + offs * stride_yn, y, mask=mask)


# 3) Row-wise linear: C[M, N] = A[M, D] @ W^T[D, N] + bias[N]
@triton.jit
def _linear_rowwise_kernel(
    A_ptr,             # *fp32, input A [M, D]
    W_ptr,             # *fp32, weight W [N, D]
    B_ptr,             # *fp32, bias [N]
    C_ptr,             # *fp32, output C [M, N]
    M, D, N,
    stride_am, stride_ad,
    stride_wn, stride_wd,   # strides for W (n, d)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,  # tile over N
    BLOCK_D: tl.constexpr,  # tile over D
):
    pid = tl.program_id(axis=0)  # program id over rows (M)
    if pid >= M:
        return
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        a = tl.load(A_ptr + pid * stride_am + offs_d * stride_ad, mask=offs_d < D, other=0.0)  # [BLOCK_D]
        # load W tiles for these n offsets
        w = tl.load(W_ptr + offs_n[:, None] * stride_wn + offs_d[None, :] * stride_wd,
                    mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)  # [BLOCK_N, BLOCK_D]
        # accumulate: acc[n] += sum_j a[j] * w[n, j]
        acc += tl.sum(w * a[None, :], axis=1)
    # add bias
    bias = tl.load(B_ptr + offs_n * stride_cn, mask=offs_n < N, other=0.0)
    acc = acc + bias
    tl.store(C_ptr + pid * stride_cm + offs_n * stride_cn, acc, mask=offs_n < N)


# Helper to run Triton LayerNorm (affine) over last dim for 2D [M, N]
def _run_triton_layer_norm_2d(x_2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    M, N = x_2d.shape
    x = x_2d.contiguous()
    # allocate sum and sumsq
    sum_ = torch.empty(M, device=x.device, dtype=torch.float32)
    sumsq = torch.empty(M, device=x.device, dtype=torch.float32)
    # launch mean/var kernel
    BLOCK_N = 128
    grid = (M,)
    _layernorm_mean_var_kernel[grid](
        x, sum_, sumsq,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N,
    )
    # allocate output
    y = torch.empty_like(x)
    # launch norm+affine kernel
    _layernorm_norm_affine_kernel[grid](
        x, weight, bias, y,
        M, N,
        eps,
        x.stride(0), x.stride(1),
        weight.stride(0), bias.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return y


# Helper to run Triton elementwise 2D add
def _run_triton_add_2d(a_2d: torch.Tensor, b_2d: torch.Tensor) -> torch.Tensor:
    M, N = a_2d.shape
    a = a_2d.contiguous()
    b = b_2d.contiguous()
    y = torch.empty_like(a)
    BLOCK_N = 128
    grid = (M,)
    _add_2d_kernel[grid](
        a, b, y,
        M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return y


# Helper to run Triton linear row-wise: C[M, N] = A[M, D] @ W^T[D, N] + bias[N]
def _run_triton_linear_rowwise(a_2d: torch.Tensor, w_2d: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, D = a_2d.shape
    N = w_2d.shape[0]
    a = a_2d.contiguous()
    w = w_2d.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (M,)
    _linear_rowwise_kernel[grid](
        a, w, bias, c,
        M, D, N,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_N=128, BLOCK_D=64,
    )
    return c


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we allocate/launch Triton kernels in forward.

    def forward(self, hidden_states: torch.Tensor,
                norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor,
                mlp_fc1_weight: torch.Tensor, mlp_fc1_bias: torch.Tensor,
                mlp_fc2_weight: torch.Tensor, mlp_fc2_bias: torch.Tensor,
                layernorm_eps: float):
        # Shapes
        B, S, D = hidden_states.shape
        M = B * S

        # 1) First LayerNorm using Triton
        layer1_out = _run_triton_layer_norm_2d(hidden_states, norm1_weight, norm1_bias, layernorm_eps)  # [B, S, D]

        # 2) In-proj linear via Triton: A[M, D] -> [M, inner]
        inner = in_proj_weight.shape[0]
        a_flat = hidden_states.contiguous().view(M, D)
        u_flat = _run_triton_linear_rowwise(a_flat, in_proj_weight, in_proj_bias)  # [M, inner]
        u = u_flat.view(B, S, inner)

        # 3) Out-proj linear via Triton on layer1_out: A[M, D] -> [M, D]
        a_out_flat = layer1_out.contiguous().view(M, D)
        out_flat = _run_triton_linear_rowwise(a_out_flat, out_proj_weight, out_proj_bias)  # [M, D]
        hyena_out = out_flat.view(B, S, D)

        # 4) First residual addition: residual + hyena_out (Triton add)
        out = _run_triton_add_2d(hidden_states.view(M, D), hyena_out.view(M, D))  # [M, D]
        out = out.view(B, S, D)

        # 5) Second LayerNorm using Triton
        out2_norm = _run_triton_layer_norm_2d(out, norm2_weight, norm2_bias, layernorm_eps)  # [B, S, D]

        # 6) First MLP linear via Triton: A[M, D] -> [M, d_inner]
        d_inner = mlp_fc1_weight.shape[0]
        mlp1_in_flat = out2_norm.contiguous().view(M, D)
        mlp1_out_flat = _run_triton_linear_rowwise(mlp1_in_flat, mlp_fc1_weight, mlp_fc1_bias)  # [M, d_inner]

        # 7) Second MLP linear via Triton: A[M, d_inner] -> [M, d_model]
        d_model = mlp_fc2_weight.shape[0]
        mlp2_in_flat = mlp1_out_flat.contiguous()
        mlp2_out_flat = _run_triton_linear_rowwise(mlp2_in_flat, mlp_fc2_weight, mlp_fc2_bias)  # [M, d_model]

        # 8) Final residual addition with MLP output (Triton add)
        final_output = _run_triton_add_2d(out2_norm.view(M, D), mlp2_out_flat)  # [M, D]
        final_output = final_output.view(B, S, D)

        return final_output


def run(*args):
    return ModelNew()(*args)
