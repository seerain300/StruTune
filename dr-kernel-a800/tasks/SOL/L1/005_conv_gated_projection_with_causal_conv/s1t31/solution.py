import triton
import triton.language as tl


# Triton kernel: linear projection Y[B, S, M] = sum over H of X[B, S, h] * W[M, h] + bias[M]
# Assumes W is of shape (M, H) and X is (B, S, H).
@triton.jit
def TritonLinearProjKernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,    # X strides: (B, S, H)
    stride_w_m, stride_w_h,                # W strides: (M, H)
    stride_y_b, stride_y_s, stride_y_m,    # Y strides: (B, S, M)
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # BLOCK_M defaults to 1 in launch
    mask_s = s_offsets < S
    mask_m = m_offsets < M  # BLOCK_M set to M at launch

    # accumulator (S, M)
    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # loop over H
    for h in range(0, H):
        x_ptrs = X_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + h * stride_x_h
        w_ptrs = W_ptr + m_offsets[None, :] * stride_w_m + h * stride_w_h

        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0)  # (BLOCK_S, 1)
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0)  # (1, BLOCK_M)

        acc += x_vals * w_vals  # broadcast over M

    # add bias (M)
    bias_vals = tl.load(Bias_ptr + m_offsets, mask=mask_m, other=0.0)  # (BLOCK_M,)
    acc += bias_vals[None, :]  # broadcast over S

    # store to Y[b, s, m]
    y_ptrs = Y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + m_offsets[None, :] * stride_y_m
    tl.store(y_ptrs, acc, mask=mask_s[:, None] & mask_m[None, :])


# Triton kernel: element-wise multiply Z = A * B over (B, S, H), where A is (B, S, H), B is (B, S, H)
@triton.jit
def TritonMulKernel(
    A_ptr, B_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    a_ptrs = A_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    b_ptrs = B_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]

    a_vals = tl.load(a_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = a_vals * b_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: left-pad along sequence by PAD for Bx, producing Bx_pad[B, H, S + PAD]
# Inputs: Bx [B, H, S], output: out_pad [B, H, S + PAD], PAD=3
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_ob_b, stride_ob_h, stride_ob_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sp = tl.program_id(2)  # tiling over S_out = S + PAD

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_offsets = pid_sp * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # write zeros at the first PAD columns
    for i in range(0, PAD):
        out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s
        tl.store(out_ptr_pos, 0.0)

    # copy from Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_offsets[i]
        if s_in < S:
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
            tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s, val)


# Triton kernel: Grouped causal 1D convolution with groups=H, kernel_size=4
# Input: Bx_pad of shape (B, H, S+3); conv_weight of shape (H, 1, 4); conv_bias (H)
# Output: conv_out of shape (B, H, S)
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S, PAD,  # sizes
    stride_bx_b, stride_bx_h, stride_bx_s,  # strides for Bx_pad
    stride_w_h, stride_w_k,                 # strides for conv_weight[h, 0, k]
    stride_out_b, stride_out_h, stride_out_s,  # strides for out
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S_out

    # accumulator for output over S (vectorized)
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # reduce over kernel_size=4
    for k in range(4):
        in_s = s_offsets + k  # vector
        valid = in_s < S_out
        vals = tl.load(Bx_pad_ptr + b * stride_bx_b + h * stride_bx_h + in_s * stride_bx_s, mask=valid, other=0.0)
        w = tl.load(conv_weight_ptr + h * stride_w_h + k * stride_w_k)  # scalar
        acc += vals * w

    # add bias
    bias = tl.load(conv_bias_ptr + h)
    acc += bias

    # store to out[b, h, s]
    out_ptrs = out_ptr + b * stride_out_b + h * stride_out_h + s_offsets * stride_out_s
    tl.store(out_ptrs, acc, mask=mask_s)


# Triton kernel: final linear projection
# Z[B, S, H] = sum over H of Y[B, S, m] * out_w[m, h] + out_bias[h]
@triton.jit
def TritonFinalLinearKernel(
    Y_ptr, OutW_ptr, OutBias_ptr, Z_ptr,
    B, S, H, M,  # M = hidden_size (H)
    stride_y_b, stride_y_s, stride_y_m,  # Y strides: (B, S, M)
    stride_w_m, stride_w_h,              # OutW strides: (M, H)
    stride_z_b, stride_z_s, stride_z_h,  # Z strides: (B, S, H)
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    # accumulator for Z over H
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # reduce over M (== H)
    for m in range(0, M):
        y_ptrs = Y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + m * stride_y_m  # (S, 1)
        outw_ptrs = OutW_ptr + m * stride_w_m + h_offsets[None, :] * stride_w_h  # (1, H)

        y_vals = tl.load(y_ptrs, mask=mask_s[:, None], other=0.0)          # (S, 1)
        outw_vals = tl.load(outw_ptrs, mask=mask_h[None, :], other=0.0)    # (1, H)

        acc += y_vals * outw_vals  # broadcast over H

    # add bias
    bias = tl.load(OutBias_ptr + h_offsets, mask=mask_h, other=0.0)  # (H,)
    acc += bias[None, :]  # broadcast over S

    # store Z
    z_ptrs = Z_ptr + pid_b * stride_z_b + s_offsets[:, None] * stride_z_s + h_offsets[None, :] * stride_z_h
    tl.store(z_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        # x: (B, S, H), ensure contiguous and float32 for Triton
        B, S, H = x.shape
        x = x.contiguous().to(torch.float32)

        # 1) Three linear projections using Triton
        # B = x @ in_proj_weight[:H, :].T + bias[:H]
        W_B = in_proj_weight[:H, :].contiguous().to(torch.float32)  # (H, H)
        Bias_B = in_proj_bias[:H].contiguous().to(torch.float32)    # (H,)
        B_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        TritonLinearProjKernel


def run(*args):
    return ModelNew()(*args)
