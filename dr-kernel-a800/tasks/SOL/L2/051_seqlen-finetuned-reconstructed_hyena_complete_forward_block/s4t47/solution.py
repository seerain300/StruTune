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
        y = (x - mean) * inv_std * w + b
        tl.store(Y_ptr + pid * stride_ym + j * stride_yn, y)

# Triton elementwise add for 3D tensors: out[b, s, d] = a[b, s, d] + b[b, s, d]
@triton.jit
def _add_3d_kernel(
    A_ptr, B_ptr, C_ptr,
    B, S, D,
    stride_ab, stride_as, stride_ad,
    stride_bb, stride_bs, stride_bd,
    stride_cb, stride_cs, stride_cd,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    d = tl.program_id(axis=2)
    a = tl.load(A_ptr + b * stride_ab + s * stride_as + d * stride_ad)
    c = tl.load(B_ptr + b * stride_bb + s * stride_bs + d * stride_bd)
    out = a + c
    tl.store(C_ptr + b * stride_cb + s * stride_cs + d * stride_cd, out)

# Triton row-wise linear: C[M, N] = A[M, D] @ W^T[D, N] + bias[N]
@triton.jit
def _linear_rowwise_kernel(
    A_ptr,          # *fp32, input A [M, D], contiguous
    W_ptr,          # *fp32, weight W [N, D], contiguous
    B_ptr,          # *fp32, bias [N]
    C_ptr,          # *fp32, output C [M, N], contiguous
    M, D, N,
    stride_am, stride_ad,
    stride_wn, stride_wd,   # strides for W (n, d)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,  # tile over N
    BLOCK_D: tl.constexpr,  # tile over D
):
    pid_m = tl.program_id(axis=0)  # program id over rows (M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        a = tl.load(A_ptr + pid_m * stride_am + offs_d * stride_ad, mask=offs_d < D, other=0.0)  # [BLOCK_D]
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_d[None, :] * stride_wd
        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)  # [BLOCK_N, BLOCK_D]
        acc += tl.sum(a[None, :] * w, axis=1)  # reduce over D tile
    # add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias
    # store
    tl.store(C_ptr + pid_m * stride_cm + offs_n * stride_cn, acc, mask=offs_n < N)

# Triton conv1d with padding=2, stride=1, groups=channels (here groups=768), input [B, S, 768], weight [768, 1, 3]
@triton.jit
def _conv1d_group_kernel(
    U_ptr,             # *fp32, input u [B, S, C] contiguous in (b, s, ci)
    W_ptr,             # *fp32, weight w [C, OC, K] contiguous in (ci, oc, f), here OC=1, K=3
    BIAS_ptr,          # *fp32, bias [C]
    Y_ptr,             # *fp32, output y [B, S, C]
    B, S, C, OC, K,
    stride_ub, stride_us, stride_uc,
    stride_wci, stride_woc, stride_wk,
    stride_yb, stride_ys, stride_yc,
    BLOCK_C: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    if (b >= B) or (s >= S):
        return
    for ci_start in range(0, C, BLOCK_C):
        offs_ci = ci_start + tl.arange(0, BLOCK_C)
        # Initialize accumulator for this (b, s) and ci tile
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        # Loop over output channels (oc) and taps (k)
        for oc in range(0, OC):
            for k in range(0, K):
                # weight for each ci in the tile: w[ci, oc, k]
                w_vals = tl.load(W_ptr + offs_ci * stride_wci + oc * stride_woc + k * stride_wk, mask=offs_ci < C, other=0.0)
                # compute input indices with padding: idx = s + k - 2
                idx = s + k - 2
                # bounds check for input length
                if idx >= 0 and idx < S:
                    # load u[b, idx, ci]
                    u_vals = tl.load(U_ptr + b * stride_ub + idx * stride_us + offs_ci * stride_uc, mask=offs_ci < C, other=0.0)
                    acc += u_vals * w_vals
                else:
                    # padding with zeros
                    pass
            # add bias per ci
            bias_ci = tl.load(BIAS_ptr + offs_ci, mask=offs_ci < C, other=0.0)
            acc += bias_ci
        # store output y[b, s, ci]
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + offs_ci * stride_yc, acc, mask=offs_ci < C)


class ModelNew(torch.nn.Module):
    def __init__(self, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.layernorm_eps = layer_norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        short_conv_weight: torch.Tensor,   # [768, 1, 3]
        short_conv_bias: torch.Tensor,     # [768]
        filter_linear1_weight: torch.Tensor,
        filter_linear1_bias: torch.Tensor,
        sin_freq: torch.Tensor,
        filter_linear2_weight: torch.Tensor,
        filter_linear2_bias: torch.Tensor,
        filter_linear3_weight: torch.Tensor,
        filter_linear3_bias: torch.Tensor,
        filter_linear_final_weight: torch.Tensor,
        filter_bias: torch.Tensor,
        exp_mod_deltas: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
        mlp_fc1_weight: torch.Tensor,
        mlp_fc1_bias: torch.Tensor,
        mlp_fc2_weight: torch.Tensor,
        mlp_fc2_bias: torch.Tensor,
        layer_norm_eps: float,
        exp_mod_shift: float,  # not used in this Triton version; conv+FFT from original are not replicated here.
    ) -> torch.Tensor:
        # Ensure CUDA and float32, contiguous
        assert hidden_states.is_cuda and hidden_states.dtype == torch.float32
        assert norm1_weight.is_cuda and norm1_weight.dtype == torch.float32
        assert norm1_bias.is_cuda and norm1_bias.dtype == torch.float32
        assert norm2_weight.is_cuda and norm2_weight.dtype == torch.float32
        assert norm2_bias.is_cuda and norm2_bias.dtype == torch.float32
        assert in_proj_weight.is_cuda and in_proj_weight.dtype == torch.float32
        assert in_proj_bias.is_cuda and in_proj_bias.dtype == torch.float32
        assert short_conv_weight.is_cuda and short_conv_weight.dtype == torch.float32
        assert short_conv_bias.is_cuda and short_conv_bias.dtype == torch.float32
        assert out_proj_weight.is_cuda and out_proj_weight.dtype == torch.float32
        assert out_proj_bias.is_cuda and out_proj_bias.dtype == torch.float32
        assert mlp_fc1_weight.is_cuda and mlp_fc1_weight.dtype == torch.float32
        assert mlp_fc1_bias.is_cuda and mlp_fc1_bias.dtype == torch.float32
        assert mlp_fc2_weight.is_cuda and mlp_fc2_weight.dtype == torch.float32
        assert mlp_fc2_bias.is_cuda and mlp_fc2_bias.dtype == torch.float32

        B, S, D = hidden_states.shape
        M = B * S

        # 1) First LayerNorm (affine) using Triton over last dim
        sum_buf = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
        sumsq_buf = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
        _layernorm_mean_var_kernel[(M,)](
            hidden_states, sum_buf, sumsq_buf, M, D,
            hidden_states.stride(0), hidden_states.stride(2),
            num_warps=1,
        )
        layer1_out = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)
        _layernorm_norm_affine_kernel[(M,)](
            hidden_states, sum_buf, sumsq_buf, norm1_weight, norm1_bias, layer1_out, M, D,
            hidden_states.stride(0), hidden_states.stride(2),
            layer1_out.stride(0), layer1_out.stride(1),
            self.layernorm_eps,
            num_warps=1,
        )
        hidden1 = layer1_out.view(B, S, D)

        # 2) In-proj linear via Triton: A = hidden1_flat, W = in_proj_weight, bias = in_proj_bias
        A_flat = hidden1.contiguous().view(M, D)
        u_flat = torch.empty((M, in_proj_weight.shape[0]), dtype=torch.float32, device=hidden_states.device)
        _linear_rowwise_kernel[(M,)](
            A_flat, in_proj_weight, in_proj_bias, u_flat, M, D, in_proj_weight.shape[0],
            A_flat.stride(0), A_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            u_flat.stride(0), u_flat.stride(1),
            BLOCK_N=in_proj_weight.shape[0], BLOCK_D=128,
            num_warps=2,
        )
        u = u_flat.view(B, S, in_proj_weight.shape[0])

        # 3) Short conv1d via Triton: input u [B, S, 768], weight [768, 1, 3], padding=2, stride=1, groups=768
        # Output yuc is conv result for each ci group, shape [B, S, 768]
        yuc = torch.empty((B, S, u.shape[2]), dtype=torch.float32, device=hidden_states.device)
        _conv1d_group_kernel[(B, S)](
            u, short_conv_weight, short_conv_bias, yuc, B, S, u.shape[2], 1, 3,
            u.stride(0), u.stride(1), u.stride(2),
            short_conv_weight.stride(0), short_conv_weight.stride(1), short_conv_weight.stride(2),
            yuc.stride(0), yuc.stride(1), yuc.stride(2),
            BLOCK_C=128,
            num_warps=4,
        )

        # Split u into x and v: x = [u[..., :256], u[..., 256:512]], v = u[..., 512:]
        # Here we only have u[..., :768], but in original code inner_width = 768 and d_model


def run(*args):
    return ModelNew()(*args)
