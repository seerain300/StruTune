import math
import torch
import triton
import triton.language as tl


# ---------- Triton kernels ----------

@triton.jit
def ln_forward_kernel(x_ptr, weight_ptr, bias_ptr, y_ptr,
                       M, D, eps,
                       BLOCK_SIZE: tl.constexpr):
    """
    LayerNorm forward for rows of a 2D tensor [M, D].
    - x_ptr: input flattened to [M*D], float32
    - weight_ptr, bias_ptr: [D] float32
    - y_ptr: output flattened to [M*D], float32
    - M: number of rows = B*S
    - D: number of columns/features = d_model
    Launch: one program per row
    """
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    for start in range(0, D, BLOCK_SIZE):
        col = start + offs
        mask = col < D
        base = row * D + col
        x = tl.load(x_ptr + base, mask=mask, other=0.0)
        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        mean = sum_x / D
        var = sum_x2 / D - mean * mean
        inv_std = 1.0 / tl.sqrt(var + eps)
        y = (x - mean) * inv_std
        w = tl.load(weight_ptr + col, mask=mask, other=1.0)
        b = tl.load(bias_ptr + col, mask=mask, other=0.0)
        y = y * w + b
        tl.store(y_ptr + base, y, mask=mask)


@triton.jit
def matmul_bias_kernel(a_ptr, b_ptr, bias_ptr, c_ptr,
                        M, N, K,
                        stride_am, stride_ak,
                        stride_bk, stride_bn,
                        stride_cm, stride_cn,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C = A @ B + bias, where bias is [N] added elementwise
    A[M, K], B[K, N], C[M, N]
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b)

    # Add bias: broadcast over rows
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def conv1d_per_channel_kernel(u_ptr, w_ptr, b_ptr, y_ptr,
                              B, C, L_in, F,
                              stride_u, stride_w,
                              stride_y,
                              BLOCK: tl.constexpr):
    """
    Per-channel 1D convolution with zero padding on both sides by 2:
    - u_ptr: input [B, C, L_in], float32
    - w_ptr: weight per channel [C, F], float32 (since groups=C, each channel has its own kernel)
    - b_ptr: bias [C], float32
    - y_ptr: output [B, C, L_out], float32, L_out = L_in - F + 1
    Launch: grid (B, C, L_out) so each program computes one output element.
    """
    b_id = tl.program_id(axis=0)
    c_id = tl.program_id(axis=1)
    l_out_id = tl.program_id(axis=2)
    if l_out_id >= (L_in - F + 1):
        return
    # output position index
    t = l_out_id
    # accumulation
    acc = tl.zeros((), dtype=tl.float32)
    # For each filter tap
    for f in tl.static_range(0, F):
        s = t + f  # input index (zero padding already accounted in L_out)
        # bounds check: s in [0, L_in-1]
        # Address: u[b, c, s]
        u_offset = b_id * stride_u + c_id * stride_u + s * stride_u
        w_offset = c_id * F + f  # since w is [C, F], stride_w = 1 in second dim
        val = tl.load(u_ptr + u_offset)
        wval = tl.load(w_ptr + w_offset)
        acc += val * wval
    # add bias
    bval = tl.load(b_ptr + c_id)
    acc += bval
    # store: y[b, c, t]
    y_offset = b_id * stride_y + c_id * stride_y + t * stride_y
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def exp_mod_kernel(h_ptr, delta_ptr, shift, out_ptr, size, BLOCK: tl.constexpr):
    """
    Elementwise: out = h * (exp(-|delta|) + shift)
    - h_ptr: [size] float32
    - delta_ptr: [size] float32
    - shift: float32 scalar
    - out_ptr: [size] float32
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    h = tl.load(h_ptr + offs, mask=mask, other=0.0)
    delta = tl.load(delta_ptr + offs, mask=mask, other=0.0)
    # Match original intent: exp(-|delta|) + shift
    exp_arg = -tl.abs(delta) + shift
    h_mod = h * tl.exp(exp_arg)
    tl.store(out_ptr + offs, h_mod, mask=mask)


# ---------- ModelNew forward ----------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                hidden_states: torch.Tensor,
                norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                short_conv_weight: torch.Tensor, short_conv_bias: torch.Tensor,
                filter_linear1_weight: torch.Tensor, filter_linear1_bias: torch.Tensor,
                sin_freq: torch.Tensor,  # unused
                filter_linear2_weight: torch.Tensor, filter_linear2_bias: torch.Tensor,
                filter_linear3_weight: torch.Tensor, filter_linear3_bias: torch.Tensor,
                filter_linear_final_weight: torch.Tensor, filter_bias: torch.Tensor,
                exp_mod_deltas: torch.Tensor,  # [1, 1, d_model]
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor,
                mlp_fc1_weight: torch.Tensor, mlp_fc1_bias: torch.Tensor,
                mlp_fc2_weight: torch.Tensor, mlp_fc2_bias: torch.Tensor,
                layer_norm_eps: float,
                exp_mod_shift: float):
        """
        Triton-only forward: all heavy ops are computed by Triton kernels.
        The iterative FFT-like loop remains in PyTorch (Triton lacks complex FFT).
        """
        B, S, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) LayerNorm 1 on hidden_states
        x1 = hidden_states
        M = B * S
        x1_flat = x1.reshape(M, D).contiguous()
        y1_flat = torch.empty_like(x1_flat)
        # Launch ln_forward_kernel: one program per row
        BLOCK_SIZE = 256
        grid_ln = (M,)
        ln_forward_kernel[grid_ln](
            x1_flat, norm1_weight, norm1_bias, y1_flat,
            M, D, layer_norm_eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        residual = y1_flat.reshape(B, S, D)

        # 2) Input projection: u = F.linear(residual, in_proj_weight, in_proj_bias) -> [B, inner_width, S]
        #    In Triton: A = residual [B*S, D], B = in_proj_weight [inner_width, D], C = [B*S, inner_width]
        inner_width = D * (2 + 1)  # order=2, so inner_width = d_model * 3 = 768
        A = residual.transpose(1, 2).reshape(B * D, S).transpose(0, 1).reshape(B * S, D).contiguous()
        # Create B_t [D, inner_width] by transpose of in_proj_weight
        w_t = in_proj_weight.transpose(0, 1).contiguous()  # [D, inner_width]
        M_mat = B * S
        K = D
        N_mat = inner_width


def run(*args):
    return ModelNew()(*args)
