import torch
import triton
import triton.language as tl


# Triton LayerNorm (affine) for a 3D tensor [B, S, D].
# 1) Compute per-row mean and variance over D.
# 2) Normalize and apply affine (weight and bias of shape [D]).
@triton.jit
def _layernorm_mean_3d_kernel(
    X_ptr,             # *fp32, input [B, S, D], contiguous
    SUM_ptr,           # *fp32, per-row sum [B*S]
    SUMSQ_ptr,         # *fp32, per-row sum of squares [B*S]
    B, S, D,
    stride_xb, stride_xs, stride_xd,
):
    pid = tl.program_id(axis=0)
    # Map pid to (b, s)
    b = pid // S
    s = pid % S
    if (b >= B) or (s >= S):
        return
    # Accumulate sum and sum of squares over D
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)
    for d in range(0, D):
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + d * stride_xd)
        sum_val += x
        sumsq_val += x * x
    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


@triton.jit
def _layernorm_affine_3d_kernel(
    X_ptr,       # *fp32, input [B, S, D], contiguous
    Weight_ptr,  # *fp32, weight [D]
    Bias_ptr,    # *fp32, bias [D]
    Out_ptr,     # *fp32, output [B, S, D], contiguous
    B, S, D,
    stride_xb, stride_xs, stride_xd,
    stride_ob, stride_os, stride_od,
    eps: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    if (b >= B) or (s >= S):
        return
    # Load sum and sumsq for this (b, s) row
    pid = b * S + s
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    mean = sum_val / D
    var = sumsq_val / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    # Normalize and apply affine
    for d in range(0, D):
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + d * stride_xd)
        w = tl.load(Weight_ptr + d)
        b_bias = tl.load(Bias_ptr + d)
        y = (x - mean) * inv_std
        y = y * w + b_bias
        tl.store(Out_ptr + b * stride_ob + s * stride_os + d * stride_od, y)


# Triton elementwise add for 3D tensors: C = A + B
@triton.jit
def _add_3d_kernel(
    A_ptr, B_ptr, Out_ptr,
    B, S, D,
    stride_ab, stride_as, stride_ad,
    stride_bb, stride_bs, stride_bd,
    stride_ob, stride_os, stride_od,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    if (b >= B) or (s >= S):
        return
    for d in range(0, D):
        a = tl.load(A_ptr + b * stride_ab + s * stride_as + d * stride_ad)
        b_val = tl.load(B_ptr + b * stride_bb + s * stride_bs + d * stride_bd)
        c = a + b_val
        tl.store(Out_ptr + b * stride_ob + s * stride_os + d * stride_od, c)


# Triton row-wise linear: computes C[M, N] = A[M, D] @ W^T[D, N] + bias[N]
@triton.jit
def _linear_rowwise_kernel(
    A_ptr,          # *fp32, input A [M, D], contiguous (row-major)
    W_ptr,          # *fp32, weight W [N, D], contiguous (row-major)
    B_ptr,          # *fp32, bias [N] (can be None; pass zeros if no bias)
    C_ptr,          # *fp32, output C [M, N], contiguous (row-major)
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
    # Accumulator for this row
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Loop over D in tiles
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        # Load A row chunk
        a = tl.load(A_ptr + pid * stride_am + offs_d * stride_ad, mask=offs_d < D, other=0.0)  # [BLOCK_D]
        # Load W tile [BLOCK_N, BLOCK_D]
        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + offs_d[None, :] * stride_wd,
            mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0,
        )  # [BLOCK_N, BLOCK_D]
        # Compute dot product: [BLOCK_N] += sum_{k in BLOCK_D} w[:, k] * a[k]
        # Using broadcasting and reduction:
        acc += tl.sum(w * a[None, :], axis=1)
    # Add bias if provided
    if B_ptr is not None:
        b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += b
    # Store results for this row
    tl.store(C_ptr + pid * stride_cm + offs_n * stride_cn, acc, mask=offs_n < N)


# Helper to run Triton LayerNorm over [B, S, D]
def _run_triton_layernorm_3d(x_3d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    """
    x_3d: [B, S, D], float32, contiguous CUDA
    weight, bias: [D], float32, contiguous CUDA
    Returns: y_3d [B, S, D]
    """
    assert x_3d.is_cuda and x_3d.dtype == torch.float32, "x_3d must be CUDA float32"
    assert weight.is_cuda and weight.dtype == torch.float32, "weight must be CUDA float32"
    assert bias.is_cuda and bias.dtype == torch.float32, "bias must be CUDA float32"
    B, S, D = x_3d.shape
    y = torch.empty_like(x_3d)
    # SUM and SUMSQ buffers
    M = B * S
    sum_buf = torch.empty(M, dtype=torch.float32, device=x_3d.device)
    sumsq_buf = torch.empty(M, dtype=torch.float32, device=x_3d.device)

    # Grid: one program per (b, s) row
    grid = (B, S)
    _layernorm_mean_3d_kernel[grid](x_3d, sum_buf, sumsq_buf, B, S, D, x_3d.stride(0), x_3d.stride(1), x_3d.stride(2), num_warps=4)
    _layernorm_affine_3d_kernel[grid](x_3d, weight, bias, y, B, S, D, x_3d.stride(0), x_3d.stride(1), x_3d.stride(2), y.stride(0), y.stride(1), y.stride(2), eps=1e-5, num_warps=4)
    return y


# Helper to run Triton linear: C[M, N] = A[M, D] @ W^T[D, N] + bias[N]
def _run_triton_linear(a_flat: torch.Tensor, w: torch.Tensor, bias: torch.Tensor):
    """
    a_flat: [M, D] contiguous CUDA float32
    w: [N, D] contiguous CUDA float32
    bias: [N] contiguous CUDA float32 or None
    Returns: c_flat [M, N] float32
    """
    assert a_flat.is_cuda and a_flat.dtype == torch.float32, "a_flat must be CUDA float32"
    assert w.is_cuda and w.dtype == torch.float32, "weight must be CUDA float32"
    M, D = a_flat.shape
    N = w.shape[0]
    c = torch.empty((M, N), dtype=torch.float32, device=a_flat.device)
    # Choose block sizes (small to reduce compilation issues)
    BLOCK_N = 64
    BLOCK_D = 64
    grid = (M,)
    # Pass bias as None if not provided (bias will be zeros)
    if bias is None:
        _linear_rowwise_kernel[grid](
            a_flat, w, None, c, M, D, N,
            a_flat.stride(0), a_flat.stride(1),
            w.stride(0), w.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4
        )
    else:
        _linear_rowwise_kernel[grid](
            a_flat, w, bias, c, M, D, N,
            a_flat.stride(0), a_flat.stride(1),
            w.stride(0), w.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4
        )
    return c


# Helper to run Triton add for 3D tensors
def _run_triton_add_3d(a_3d: torch.Tensor, b_3d: torch.Tensor):
    """
    a_3d, b_3d: [B, S, D], float32, contiguous CUDA
    Returns: c_3d [B, S, D]
    """
    assert a_3d.is_cuda and b_3d.is_cuda and a_3d.dtype == torch.float32 and b_3d.dtype == torch.float32, "inputs must be CUDA float32"
    B, S, D = a_3d.shape
    c = torch.empty_like(a_3d)
    grid = (B, S)
    _add_3d_kernel[grid](a_3d, b_3d, c, B, S, D, a_3d.stride(0), a_3d.stride(1), a_3d.stride(2), b_3d.stride(0), b_3d.stride(1), b_3d.stride(2), c.stride(0), c.stride(1), c.stride(2), num_warps=4)
    return c


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor,
                mlp_fc1_weight: torch.Tensor, mlp_fc1_bias: torch.Tensor,
                mlp_fc2_weight: torch.Tensor, mlp_fc2_bias: torch.Tensor,
                layer_norm_eps: float = 1e-5,
                exp_mod_shift: float = 0.05):
        """
        We intentionally avoid using PyTorch computations in forward.
        This ModelNew performs the following steps with Triton kernels:
        1) First LayerNorm (affine) over hidden_states [B, S, D]
        2) In-projection: hidden_states -> u [B, S, inner]
        3) Out-projection: hidden_states (after LayerNorm1) -> hyena_out [B, S, D]
        4) Residual: hidden_states + hyena_out
        5) Second LayerNorm (affine)
        6) Two MLP linears: norm2_out -> mlp_out [B, S, D]
        7) Final output: mlp_out
        """
        assert hidden_states.is_cuda and hidden_states.dtype == torch.float32, "hidden_states must be CUDA float32"
        B, S, D = hidden_states.shape
        # Ensure parameters are contiguous CUDA float32
        norm1_weight = norm1_weight.contiguous().to(torch.float32).to(hidden_states.device)
        norm1_bias = norm1_bias.contiguous().to(torch.float32).to(hidden_states.device)
        norm2_weight = norm2_weight.contiguous().to(torch.float32).to(hidden_states.device)
        norm2_bias = norm2_bias.contiguous().to(torch.float32).to(hidden_states.device)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32).to(hidden_states.device)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32).to(hidden_states.device)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32).to(hidden_states.device)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32).to(hidden_states.device)
        mlp_fc1_weight = mlp_fc1_weight.contiguous().to(torch.float32).to(hidden_states.device)
        mlp_fc1_bias = mlp_fc1_bias.contiguous().to(torch.float32).to(hidden_states.device)
        mlp_fc2_weight = mlp_fc2_weight.contiguous().to(torch.float32).to(hidden_states.device)
        mlp_fc2_bias = mlp_fc2_bias.contiguous().to(torch.float32).to(hidden_states.device)

        # 1) First LayerNorm (affine)
        norm1_out = _run_triton_layernorm_3d(hidden_states, norm1_weight, norm1_bias, layer_norm_eps)

        # 2) In-proj linear
        inner = in_proj_weight.shape[0]
        a_flat = hidden_states.contiguous().view(B * S, D)
        u_flat = _run_triton_linear(a_flat, in_proj_weight, in_proj_bias)  # [B*S, inner]
        u = u_flat.view(B, S, inner)

        # 3) Out-proj linear on hidden_states (after LayerNorm1) — we use hidden_states for A
        a_out_flat = norm1_out.contiguous().view(B * S, D)
        out_flat = _run_triton_linear(a_out_flat, out_proj_weight, out_proj_bias)  # [B*S, D]
        hyena_out = out_flat.view(B, S, D)

        # 4) Residual: hidden_states + hyena_out
        residual = _run_triton_add_3d(hidden_states, hyena_out)  # [B, S, D]

        # 5) Second LayerNorm (affine)
        norm2_out = _run_triton_layernorm_3d(residual, norm2_weight, norm2_bias, layer_norm_eps)  # [B, S, D]

        # 6) MLP: two linears
        M = B * S
        d_inner = mlp_fc1_weight.shape[0]
        mlp1_in_flat = norm2_out.contiguous().view(M, D)
        mlp1_out_flat = _run_triton_linear(mlp1_in_flat, mlp_fc1_weight, mlp_fc1_bias)  # [M, d_inner]

        d_model = mlp_fc2_weight.shape[0]
        mlp2_in_flat = mlp1_out_flat.contiguous()
        mlp2_out_flat = _run_triton_linear(mlp2_in_flat, mlp_fc2_weight, mlp_fc2_bias)  # [M, d_model]

        # 7) Final output
        final_out = mlp2_out_flat.view(B, S, D)
        return final_out


def run(*args):
    return ModelNew()(*args)
