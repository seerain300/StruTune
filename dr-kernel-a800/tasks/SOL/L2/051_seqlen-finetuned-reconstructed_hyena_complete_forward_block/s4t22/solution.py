import torch
import triton
import triton.language as tl


# Triton LayerNorm (affine) over last dim for a [M, N] tensor (M = B*S, N = D).
# Kernel 1: compute per-row sum and sum of squares across N.
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
    acc = tl.zeros((), dtype=tl.float32)
    acc_sq = tl.zeros((), dtype=tl.float32)
    # Iterate across N in chunks
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        x = tl.load(X_ptr + pid * stride_xm + offs_n * stride_xn, mask=mask, other=0.0)
        # Sum and sum of squares over valid elements
        acc += tl.sum(x, axis=0)
        acc_sq += tl.sum(x * x, axis=0)
    # Store per-row sum and sumsq
    tl.store(SUM_ptr + pid, acc)
    tl.store(SUMSQ_ptr + pid, acc_sq)


# Kernel 2: normalize using computed mean and variance, then apply affine weight and bias.
@triton.jit
def _layernorm_affine_kernel(
    X_ptr,             # *fp32, input [M, N]
    SUM_ptr,           # *fp32, per-row sum [M]
    SUMSQ_ptr,         # *fp32, per-row sum of squares [M]
    WEIGHT_ptr,        # *fp32, weight [N] (1D)
    BIAS_ptr,          # *fp32, bias [N] (1D)
    Y_ptr,             # *fp32, output [M, N]
    M, N,
    stride_xm, stride_xn,
    stride_w,
    stride_ym, stride_yn,
    eps: tl.constexpr,   # float32 epsilon
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return
    # Load sum and sumsq for this row
    s = tl.load(SUM_ptr + pid)
    ss = tl.load(SUMSQ_ptr + pid)
    mean = s / N
    var = ss / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    # Normalize and apply affine
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        x = tl.load(X_ptr + pid * stride_xm + offs_n * stride_xn, mask=mask, other=0.0)
        w = tl.load(WEIGHT_ptr + offs_n * stride_w, mask=mask, other=1.0)
        b = tl.load(BIAS_ptr + offs_n * stride_w, mask=mask, other=0.0)  # bias 1D contiguous
        y = (x - mean) * rstd
        y = y * w + b
        tl.store(Y_ptr + pid * stride_ym + offs_n * stride_yn, y, mask=mask)


def _run_triton_layer_norm_2d(x_2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """
    x_2d: [M, N] float32 CUDA tensor
    weight, bias: [N] float32 CUDA tensors
    returns y_2d: [M, N]
    """
    assert x_2d.is_cuda and weight.is_cuda and bias.is_cuda
    assert x_2d.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32
    M, N = x_2d.shape
    # Allocate sum and sumsq buffers
    sum_buf = torch.empty((M,), dtype=torch.float32, device=x_2d.device)
    sumsq_buf = torch.empty((M,), dtype=torch.float32, device=x_2d.device)
    # Choose BLOCK_N
    BLOCK_N = 256 if N >= 256 else 128
    grid = (M,)
    # Kernel 1: compute mean and variance
    _layernorm_mean_var_kernel[grid](
        x_2d, sum_buf, sumsq_buf, M, N,
        x_2d.stride(0), x_2d.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    # Allocate output
    y_2d = torch.empty_like(x_2d)
    # Kernel 2: normalize and apply affine
    _layernorm_affine_kernel[grid](
        x_2d, sum_buf, sumsq_buf, weight, bias, y_2d, M, N,
        x_2d.stride(0), x_2d.stride(1),
        1,  # weight/bias are 1D contiguous, stride=1
        y_2d.stride(0), y_2d.stride(1),
        eps=eps,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    return y_2d


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                layer_norm_eps: float):
        """
        Implements two LayerNorms using Triton:
        1) hidden_states -> layer1_out (affine with norm1_weight, norm1_bias)
        2) layer1_out -> final_out (affine with norm2_weight, norm2_bias)
        Returns final_out.
        """
        # Ensure CUDA float32 contiguous
        hidden_states = hidden_states.contiguous().to(torch.float32)
        # First LayerNorm over last dim
        B, S, D = hidden_states.shape
        M = B * S
        x_2d = hidden_states.view(M, D).contiguous()
        layer1_out_2d = _run_triton_layer_norm_2d(x_2d, norm1_weight, norm1_bias, layer_norm_eps)
        # Second LayerNorm over last dim
        final_out_2d = _run_triton_layer_norm_2d(layer1_out_2d, norm2_weight, norm2_bias, layer_norm_eps)
        # Reshape back to [B, S, D]
        final_out = final_out_2d.view(B, S, D)
        return final_out


def run(*args):
    return ModelNew()(*args)
