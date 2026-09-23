import math
import torch
import triton
import triton.language as tl


# Triton LayerNorm (affine) over last dim for a [M, N] tensor (M = B*S, N = D).
# First kernel computes per-row sum and sum of squares; second kernel normalizes and applies affine.

@triton.jit
def _layernorm_mean_var_kernel(
    X_ptr,             # *fp32, input [M, N]
    SUM_ptr,           # *fp32, per-row sum [M]
    SUMSQ_ptr,         # *fp32, per-row sum of squares [M]
    M, N,
    stride_xm, stride_xn,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return
    acc = 0.0
    acc_sq = 0.0
    for j in range(0, N):
        x = tl.load(X_ptr + pid * stride_xm + j * stride_xn)
        acc += x
        acc_sq += x * x
    tl.store(SUM_ptr + pid, acc)
    tl.store(SUMSQ_ptr + pid, acc_sq)

@triton.jit
def _layernorm_norm_affine_kernel(
    X_ptr,             # *fp32, input [M, N]
    SUM_ptr,           # *fp32, per-row sum [M]
    SUMSQ_ptr,         # *fp32, per-row sum of squares [M]
    WEIGHT_ptr,        # *fp32, weight [N]
    BIAS_ptr,          # *fp32, bias [N]
    Y_ptr,             # *fp32, output [M, N]
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return
    sumv = tl.load(SUM_ptr + pid)
    sumsq = tl.load(SUMSQ_ptr + pid)
    mean = sumv / N
    var = sumsq / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    for j in range(0, N):
        x = tl.load(X_ptr + pid * stride_xm + j * stride_xn)
        w = tl.load(WEIGHT_ptr + j)
        b = tl.load(BIAS_ptr + j)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(Y_ptr + pid * stride_ym + j * stride_yn, y)

# Triton elementwise 3D add: out[b, s, d] = a[b, s, d] + b[b, s, d]
@triton.jit
def _add_3d_kernel(
    A_ptr, B_ptr, Y_ptr,
    B, S, D,
    stride_ab, stride_as, stride_ad,
    stride_bb, stride_bs, stride_bd,
    stride_yb, stride_ys, stride_yd,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    d = tl.program_id(axis=2)
    if (b >= B) or (s >= S) or (d >= D):
        return
    a_val = tl.load(A_ptr + b * stride_ab + s * stride_as + d * stride_ad)
    b_val = tl.load(B_ptr + b * stride_bb + s * stride_bs + d * stride_bd)
    y_val = a_val + b_val
    tl.store(Y_ptr + b * stride_yb + s * stride_ys + d * stride_yd, y_val)

# Triton row-wise linear: C[M, N] = A[M, D] @ W^T[D, N] + bias[N]
@triton.jit
def _linear_rowwise_kernel(
    A_ptr,          # *fp32, input A [M, D], contiguous (row-major)
    W_ptr,          # *fp32, weight W [N, D], contiguous (row-major)
    B_ptr,          # *fp32, bias [N]
    C_ptr,          # *fp32, output C [M, N], contiguous (row-major)
    M, D, N,
    stride_am, stride_ad,
    stride_wn, stride_wd,   # strides for W (n, d)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,  # tile over N
    BLOCK_D: tl.constexpr,  # tile over D
):
    pid = tl.program_id(axis=0)  # program id over rows (M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        a = tl.load(A_ptr + pid * stride_am + offs_d * stride_ad, mask=offs_d < D, other=0.0)  # [BLOCK_D]
        w = tl.load(W_ptr + offs_n[:, None] * stride_wn + offs_d[None, :] * stride_wd,
                    mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)  # [BLOCK_N, BLOCK_D]
        acc += tl.sum(a[None, :] * w, axis=1)  # reduce over D, accumulate into BLOCK_N
    # add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b
    tl.store(C_ptr + pid * stride_cm + offs_n * stride_cn, acc, mask=offs_n < N)


class ModelNew(torch.nn.Module):
    def __init__(self, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.layernorm_eps = layer_norm_eps

    def forward(self, hidden_states: torch.Tensor,
        norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
        norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
        in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
        out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor,
        mlp_fc1_weight: torch.Tensor, mlp_fc1_bias: torch.Tensor,
        mlp_fc2_weight: torch.Tensor, mlp_fc2_bias: torch.Tensor,
    ):
        # Ensure CUDA float32 and contiguous
        assert hidden_states.is_cuda and hidden_states.dtype == torch.float32
        B, S, D = hidden_states.shape
        M = B * S

        # 1) First LayerNorm using Triton
        sum_buf = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
        sumsq_buf = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
        grid_layernorm = (M,)
        _layernorm_mean_var_kernel[grid_layernorm](
            hidden_states, sum_buf, sumsq_buf, M, D,
            hidden_states.stride(0), hidden_states.stride(2),
            num_warps=1,
        )
        layer1_out_flat = torch.empty(M, D, dtype=torch.float32, device=hidden_states.device)
        _layernorm_norm_affine_kernel[grid_layernorm](
            hidden_states, sum_buf, sumsq_buf, norm1_weight, norm1_bias, layer1_out_flat, M, D,
            hidden_states.stride(0), hidden_states.stride(2),
            layer1_out_flat.stride(0), layer1_out_flat.stride(1),
            self.layernorm_eps,
            num_warps=1,
        )
        layer1_out = layer1_out_flat.view(B, S, D)

        # 2) In-proj linear via Triton
        a_flat = hidden_states.contiguous().view(M, D)
        inner = in_proj_weight.shape[0]
        u_flat = torch.empty(M, inner, dtype=torch.float32, device=hidden_states.device)
        grid_in = (M,)
        _linear_rowwise_kernel[grid_in](
            a_flat, in_proj_weight, in_proj_bias, u_flat, M, D, inner,
            a_flat.stride(0), a_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            u_flat.stride(0), u_flat.stride(1),
            BLOCK_N=inner, BLOCK_D=128,
            num_warps=1,
        )
        u = u_flat.view(B, S, inner)

        # 3) Out-proj linear via Triton on layer1_out
        a_out_flat = layer1_out.contiguous().view(M, D)
        out_flat = torch.empty(M, D, dtype=torch.float32, device=hidden_states.device)
        grid_out = (M,)
        _linear_rowwise_kernel[grid_out](
            a_out_flat, out_proj_weight, out_proj_bias, out_flat, M, D, D,
            a_out_flat.stride(0), a_out_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_N=D, BLOCK_D=128,
            num_warps=1,
        )
        hyena_out = out_flat.view(B, S, D)

        # 4) First residual addition: residual + hyena_out (Triton add)
        out = torch.empty_like(hidden_states)
        _add_3d_kernel[(B, S, D)](
            hidden_states, hyena_out, out,
            B, S, D,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            hyena_out.stride(0), hyena_out.stride(1), hyena_out.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1,
        )

        # 5) Second LayerNorm using Triton
        sum_buf2 = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
        sumsq_buf2 = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
        _layernorm_mean_var_kernel[grid_layernorm](
            out, sum_buf2, sumsq_buf2, M, D,
            out.stride(0), out.stride(2),
            num_warps=1,
        )
        y2_flat = torch.empty(M, D, dtype=torch.float32, device=hidden_states.device)
        _layernorm_norm_affine_kernel[grid_layernorm](
            out, sum_buf2, sumsq_buf2, norm2_weight, norm2_bias, y2_flat, M, D,
            out.stride(0), out.stride(2),
            y2_flat.stride(0), y2_flat.stride(1),
            self.layernorm_eps,
            num_warps=1,
        )
        out2_norm = y2_flat.view(B, S, D)

        # 6) First MLP linear via Triton
        d_inner = mlp_fc1_weight.shape[0]
        mlp1_in_flat = out2_norm.contiguous().view(M, D)
        mlp1_out_flat = torch.empty(M, d_inner, dtype=torch.float32, device=hidden_states.device)
        grid_mlp1 = (M,)
        _linear_rowwise_kernel[grid_mlp1](
            mlp1_in_flat, mlp_fc1_weight, mlp_fc1_bias, mlp1_out_flat, M, D, d_inner,
            mlp1_in_flat.stride(0), mlp1_in_flat.stride(1),
            mlp_fc1_weight.stride(0), mlp_fc1_weight.stride(1),
            mlp1_out_flat.stride(0), mlp1_out_flat.stride(1),
            BLOCK_N=d_inner, BLOCK_D=128,
            num_warps=1,
        )
        mlp1_out = mlp1_out_flat.view(B, S, d_inner)

        # 7) Second MLP linear via Triton
        d_model = mlp_fc2_weight.shape[0]
        mlp2_in_flat = mlp1_out.contiguous().view(M, d_inner)
        mlp2_out_flat = torch.empty(M, d_model, dtype=torch.float32, device=hidden_states.device)
        grid_mlp2 = (M,)
        _linear_rowwise_kernel[grid_mlp2](
            mlp2_in_flat, mlp_fc2_weight, mlp_fc2_bias, mlp2_out_flat, M, d_inner, d_model,
            mlp2_in_flat.stride(0), mlp2_in_flat.stride(1),
            mlp_fc2_weight.stride(0), mlp_fc2_weight.stride(1),
            mlp2_out_flat.stride(0), mlp2_out_flat.stride(1),
            BLOCK_N=d_model, BLOCK_D=128,
            num_warps=1,
        )
        mlp2_out = mlp2_out_flat.view(B, S, d_model)

        # 8) Final residual addition with MLP output (Triton add)
        final_output = torch.empty_like(hidden_states)
        _add_3d_kernel[(B, S, D)](
            out2_norm, mlp2_out, final_output,
            B, S, D,
            out2_norm.stride(0), out2_norm.stride(1), out2_norm.stride(2),
            mlp2_out.stride(0), mlp2_out.stride(1), mlp2_out.stride(2),
            final_output.stride(0), final_output.stride(1), final_output.stride(2),
            num_warps=1,
        )

        return final_output


def run(*args):
    return ModelNew()(*args)
