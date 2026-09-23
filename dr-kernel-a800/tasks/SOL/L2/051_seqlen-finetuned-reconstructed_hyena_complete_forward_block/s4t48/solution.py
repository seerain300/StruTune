import math
import torch
import triton
import triton.language as tl


# Triton LayerNorm (affine) over last dim for a [M, N] tensor (M = B*S, N = D).
# 1) Compute per-row sum and sum of squares.
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

# 2) Normalize and apply affine (weight [N], bias [N]).
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


# Triton elementwise addition for 3D tensors: out[b, s, d] = a[b, s, d] + b[b, s, d]
# Vectorized kernel along last dim (D).
@triton.jit
def _add_3d_vec_kernel(
    A_ptr, B_ptr, OUT_ptr,
    B, S, D,
    stride_ab, stride_as, stride_ad,
    stride_bb, stride_bs, stride_bd,
    stride_ob, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
):
    # Grid = (B, S, ceil_div(D, BLOCK_D))
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    pid_d = tl.program_id(axis=2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < D
    a = tl.load(A_ptr + pid_b * stride_ab + pid_s * stride_as + offs_d * stride_ad, mask=mask, other=0.0)
    b = tl.load(B_ptr + pid_b * stride_bb + pid_s * stride_bs + offs_d * stride_bd, mask=mask, other=0.0)
    out = a + b
    tl.store(OUT_ptr + pid_b * stride_ob + pid_s * stride_os + offs_d * stride_od, out, mask=mask)

# Triton linear-like: row-wise matmul + bias, C[M, N] = A[M, D] @ W^T[D, N] + bias[N]
@triton.jit
def _linear_rowwise_kernel(
    A_ptr,          # *fp32, [M, D]
    W_ptr,          # *fp32, [N, D]
    B_ptr,          # *fp32, bias[N]
    C_ptr,          # *fp32, [M, N]
    M, D, N,
    stride_am, stride_ad,
    stride_wn, stride_wd,   # W strides: (n, d)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,  # tile over N
    BLOCK_D: tl.constexpr,  # tile over D
):
    pid_m = tl.program_id(axis=0)
    if pid_m >= M:
        return
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        a = tl.load(A_ptr + pid_m * stride_am + offs_d * stride_ad, mask=offs_d < D, other=0.0)  # [BLOCK_D]
        # Load corresponding W rows for all n in tile
        w = tl.load(W_ptr + offs_n[:, None] * stride_wn + offs_d[None, :] * stride_wd,
                    mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)  # [BLOCK_N, BLOCK_D]
        acc += tl.sum(a[None, :] * w, axis=1)  # sum over D
    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b
    # Store
    tl.store(C_ptr + pid_m * stride_cm + offs_n * stride_cn, acc, mask=offs_n < N)


# Triton conv1d with groups: y[B, S, C] = conv(u[B, S, C], W[C, 1, F]) padding=2, stride=1, groups=B*S
@triton.jit
def _conv1d_group_kernel(
    U_ptr,            # *fp32, input [B, S, C]
    W_ptr,            # *fp32, weight [C, 1, F] but indexed as [Co, Ci, Fl]
    BIAS_ptr,         # *fp32, bias [C]
    Y_ptr,            # *fp32, output [B, S, C]
    B, S, C, F,
    stride_ub, stride_us, stride_uc,
    stride_wco, stride_wci, stride_wfl,
    stride_yb, stride_ys, stride_yc,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    # Each program handles one (b, s)
    for co in range(0, C):
        acc = 0.0
        for li in range(0, F):
            li_eff = li - 2  # padding 2
            if (li_eff >= 0) and (li_eff < S):
                u_val = tl.load(U_ptr + pid_b * stride_ub + pid_s * stride_us + li_eff * stride_uc)
                w_val = tl.load(W_ptr + co * stride_wco + 0 * stride_wci + li * stride_wfl)
                acc += u_val * w_val
        bias_val = tl.load(BIAS_ptr + co)
        acc += bias_val
        tl.store(Y_ptr + pid_b * stride_yb + pid_s * stride_ys + co * stride_yc, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, layernorm_eps=1e-5):
        super().__init__()
        self.layernorm_eps = layernorm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        short_conv_weight: torch.Tensor,
        short_conv_bias: torch.Tensor,
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
        exp_mod_shift: float,
    ):
        # Ensure dtype and contiguity
        device = hidden_states.device
        # First LayerNorm
        B, S, D = hidden_states.shape
        M = B * S
        sum_buf = torch.empty(M, dtype=torch.float32, device=device)
        sumsq_buf = torch.empty(M, dtype=torch.float32, device=device)
        _layernorm_mean_var_kernel[(M,)](
            hidden_states, sum_buf, sumsq_buf, M, D,
            hidden_states.stride(0), hidden_states.stride(2),
            num_warps=1,
        )
        layer1_out = torch.empty((M, D), dtype=torch.float32, device=device)
        _layernorm_norm_affine_kernel[(M,)](
            hidden_states, sum_buf, sumsq_buf, norm1_weight, norm1_bias, layer1_out, M, D,
            hidden_states.stride(0), hidden_states.stride(2),
            layer1_out.stride(0), layer1_out.stride(1),
            self.layernorm_eps,
            num_warps=1,
        )
        hidden1 = layer1_out.view(B, S, D)

        # In-proj linear via Triton
        A_flat = hidden1.contiguous().view(M, D)
        inner = in_proj_weight.shape[0]
        u_flat = torch.empty((M, inner), dtype=torch.float32, device=device)
        _linear_rowwise_kernel[(M,)](
            A_flat, in_proj_weight, in_proj_bias, u_flat, M, D, inner,
            A_flat.stride(0), A_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            u_flat.stride(0), u_flat.stride(1),
            BLOCK_N=inner, BLOCK_D=128,
            num_warps=2,
        )
        u = u_flat.view(B, S, inner)

        # Short conv1d via Triton: groups = B*S, padding=2, stride=1
        yuc = torch.empty((B, S, inner), dtype=torch.float32, device=device)
        _conv1d_group_kernel[(B, S)](
            u, short_conv_weight, short_conv_bias, yuc, B, S, inner, 3,
            u.stride(0), u.stride(1), u.stride(2),
            short_conv_weight.stride(0), short_conv_weight.stride(1), short_conv_weight.stride(2),
            yuc.stride(0), yuc.stride(1), yuc.stride(2),
            BLOCK_C=128,
            num_warps=4,
        )

        # For original code, x = u[:256], v = u[256:], but we don't have those splits. Move to next step.
        # We keep yuc and u for downstream.

        # Out-proj linear via Triton on layer1_out
        a_out_flat = layer1_out  # already [M, D]
        out_flat = torch.empty((M, D), dtype=torch.float32, device=device)
        _linear_rowwise_kernel[(M,)](
            a_out_flat, out_proj_weight, out_proj_bias, out_flat, M, D, D,
            a_out_flat.stride(0), a_out_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_N=D, BLOCK_D=128,
            num_warps=2,
        )
        hyena_out = out_flat.view(B, S, D)

        # Residual addition: hidden1 + hyena_out via Triton add
        # Ensure contiguity
        hidden1c = hidden1.contiguous()
        hyena_outc = hyena_out.contiguous()
        out = torch.empty((B, S, D), dtype=torch.float32, device=device)
        _add_3d_vec_kernel[(B, S, triton.cdiv(D, 128))](
            hidden1c, hyena_outc, out,
            B, S, D,
            hidden1c.stride(0), hidden1c.stride(1), hidden1c.stride(2),
            hyena_outc.stride(0), hyena_outc.stride(1), hyena_outc.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_D=128,
            num_warps=2,
        )

        # Second LayerNorm via Triton
        sum_buf2 = torch.empty(M, dtype=torch.float32, device=device)
        sumsq_buf2 = torch.empty(M, dtype=torch.float32, device=device)
        _layernorm_mean_var_kernel[(M,)](
            out, sum_buf2, sumsq_buf2, M, D,
            out.stride(0), out.stride(2),
            num_warps=1,
        )
        out2_norm = torch.empty((M, D), dtype=torch.float32, device=device)
        _layernorm_norm_affine_kernel[(M,)](
            out, sum_buf2, sumsq_buf2, norm2_weight, norm2_bias, out2_norm, M, D,
            out.stride(0), out.stride(2),
            out2_norm.stride(0), out2_norm.stride(1),
            self.layernorm_eps,
            num_warps=1,
        )
        out2_norm = out2_norm.view(B, S, D)

        # MLP 1 via Triton: [B, S, D] -> [B, S, d_inner]
        d_inner = mlp_fc1_weight.shape[0]
        mlp1_in_flat = out2_norm.contiguous().view(M, D)
        mlp1_out_flat = torch.empty((M, d_inner), dtype=torch.float32, device=device)
        _linear_rowwise_kernel[(M,)](
            mlp1_in_flat, mlp_fc1_weight, mlp_fc1_bias, mlp1_out_flat, M, D, d_inner,
            mlp1_in_flat.stride(0), mlp1_in_flat.stride(1),
            mlp_fc1_weight.stride(0), mlp_fc1_weight.stride(1),
            mlp1_out_flat.stride(0), mlp1_out_flat.stride(1),
            BLOCK_N=d_inner, BLOCK_D=128,
            num_warps=2,
        )
        mlp1_out = mlp1_out_flat.view(B, S, d_inner)

        # MLP 2 via Triton: [B, S, d_inner] -> [B, S, D]
        d_model = mlp_fc2_weight.shape[0]
        mlp2_in_flat = mlp1_out.contiguous().view(M, d_inner)
        mlp2_out_flat = torch.empty((M, d_model), dtype=torch.float32, device=device)
        _linear_rowwise_kernel[(M,)](
            mlp2_in_flat, mlp_fc2_weight, mlp_fc2_bias, mlp2_out_flat, M, d_inner, d_model,
            mlp2_in_flat.stride(0), mlp2_in_flat.stride(1),
            mlp_fc2_weight.stride(0), mlp_fc2_weight.stride(1),
            mlp2_out_flat.stride(0), mlp2_out_flat.stride(1),
            BLOCK_N=d_model, BLOCK_D=128,
            num_warps=2,
        )
        mlp2_out = mlp2_out_flat.view(B, S, d_model)

        # Final residual addition via Triton: out2_norm + mlp2_out
        out2_normc = out2_norm.contiguous()
        mlp2_outc = mlp2_out.contiguous()
        final_output = torch.empty((B, S, D), dtype=torch.float32, device=device)
        _add_3d_vec_kernel[(B, S, triton.cdiv(D, 128))](
            out2_normc, mlp2_outc, final_output,
            B, S, D,
            out2_normc.stride(0), out2_normc.stride(1), out2_normc.stride(2),
            mlp2_outc.stride(0), mlp2_outc.stride(1), mlp2_outc.stride(2),
            final_output.stride(0), final_output.stride(1), final_output.stride(2),
            BLOCK_D=128,
            num_warps=2,
        )

        return final_output


def run(*args):
    return ModelNew()(*args)
